"""Reproducible, point-in-time causal sample materialization (no model fitting)."""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MATRIX_KEY = ["Temporada", "Ronda", "DriverNumber", "LapNumber"]
KEY = ["Temporada", "Ronda", "DriverNumber", "prediction_lap"]
STATIC = ["Driver", "Team", "GranPremio"]
NUMERIC = [
    "LapNumber", "TyreLife_last_eligible", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
    "Position", "Telemetry_Speed_mean", "Telemetry_Speed_max", "Telemetry_RPM_mean", "Telemetry_RPM_max",
    "Telemetry_Throttle_valid_mean", "Telemetry_Throttle_valid_max", "Telemetry_Brake_mean",
    "Weather_AirTemp", "Weather_Humidity", "Weather_Pressure", "Weather_Rainfall", "Weather_TrackTemp",
    "Weather_WindDirection_sin", "Weather_WindDirection_cos", "Weather_WindSpeed",
]
CATEGORICAL = ["Compound_asof", "track_phase_asof"]
FEATURES = NUMERIC + CATEGORICAL
LAP_METRICS = [
    "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "Position", "Telemetry_Speed_mean",
    "Telemetry_Speed_max", "Telemetry_RPM_mean", "Telemetry_RPM_max", "Telemetry_Throttle_valid_mean",
    "Telemetry_Throttle_valid_max", "Telemetry_Brake_mean",
]
FORBIDDEN = {
    "PitInTime", "PitOutTime", "PitWithin3Laps", "LabelReason", "LabelBasis", "RuleVersion",
    "EligibleForSupervisedLearning", "Stint_asof", "FreshTyre_asof", "Telemetry_Samples",
    "Telemetry_Throttle_valid_samples", "Telemetry_Throttle_valid_fraction",
}
MASK_REASONS = {
    0: "present", 1: "padding", 2: "nominal_lap_row_missing",
    3: "temporally_unavailable", 4: "feature_missing", 5: "source_event_unavailable",
}


class ContractError(ValueError):
    """Input or output violates the causal sample contract."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_fastf1_cache(path: Path) -> Any:
    with path.open("rb") as f:
        wrapped = pickle.load(f)
    return wrapped.get("data", wrapped) if isinstance(wrapped, dict) else wrapped


def seconds(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    return value.total_seconds() if hasattr(value, "total_seconds") else float(value)


def event_frame(data: dict[str, Any]) -> pd.DataFrame:
    columns = {k: v for k, v in data.items() if isinstance(v, (list, tuple, np.ndarray, pd.Series))}
    if not columns:
        return pd.DataFrame()
    n = min(map(len, columns.values()))
    return pd.DataFrame({k: list(v)[:n] for k, v in columns.items()})


def track_phase(status: Any) -> str:
    return {
        "1": "GREEN", "2": "YELLOW", "4": "SAFETY_CAR", "5": "RED",
        "6": "VSC", "7": "VSC_ENDING",
    }.get(str(status), "UNKNOWN")


def nominal_window_laps(prediction_lap: int, width: int) -> list[int]:
    """Return fixed-width historical positions, retaining nominal pre-race positions."""
    if width < 1:
        raise ContractError("window_length debe ser positivo")
    return list(range(prediction_lap - width + 1, prediction_lap + 1))


def lap_metrics_available(
    historical_lap: int, prediction_lap: int, lap_close_seconds: float,
    decision_seconds: float, delay_seconds: float = 0.0, generated: bool = False,
) -> bool:
    """A completed-lap aggregate cannot enter its own start-of-lap decision."""
    return (
        historical_lap < prediction_lap
        and pd.notna(lap_close_seconds)
        and lap_close_seconds + delay_seconds <= decision_seconds
        and not generated
    )


def metric_delay_key(feature: str) -> str:
    if feature == "Position":
        return "matrix_position"
    if feature.startswith("Telemetry_"):
        return "matrix_telemetry"
    return "matrix_lap_metrics"


def active_timing_stint(eligible_updates: pd.DataFrame) -> Any:
    """Timing cache stint ids are monotone; ignore late updates tagged to older stints."""
    if eligible_updates.empty or "Stint" not in eligible_updates:
        return None
    stints = pd.to_numeric(eligible_updates["Stint"], errors="coerce").dropna()
    return None if stints.empty else stints.max()


def matrix_stint_for_timing_stint(timing_stint: Any, matrix_rows: pd.DataFrame) -> float | None:
    """FastF1 timing_app_data uses zero-based stints; consolidated laps use one-based stints."""
    if pd.isna(timing_stint):
        return None
    value = pd.to_numeric(pd.Series([timing_stint]), errors="coerce").iloc[0]
    if pd.isna(value) or not float(value).is_integer():
        return None
    mapped = float(value) + 1.0
    matrix_stints = pd.to_numeric(matrix_rows.get("Stint", pd.Series(dtype=float)), errors="coerce")
    return mapped if matrix_stints.eq(mapped).any() else None


def validate_inputs(matrix: pd.DataFrame, labels: pd.DataFrame) -> None:
    required = set(MATRIX_KEY + ["LapStartTime", "Time", "Driver", "Team", "GranPremio", "Compound", "TyreLife"])
    missing = sorted(required - set(matrix.columns))
    if missing:
        raise ContractError(f"Faltan columnas requeridas en matriz: {missing}")
    if matrix[MATRIX_KEY].isna().any().any() or matrix.duplicated(MATRIX_KEY).any():
        raise ContractError("Clave completa de matriz nula o duplicada")
    if not set(MATRIX_KEY).issubset(labels.columns):
        raise ContractError("Etiquetas sin clave completa")
    if labels[MATRIX_KEY].isna().any().any() or labels.duplicated(MATRIX_KEY).any():
        raise ContractError("Clave completa de etiquetas nula o duplicada")
    if not {"PitWithin3Laps", "EligibleForSupervisedLearning"}.issubset(labels.columns):
        raise ContractError("Esquema de etiquetas incompatible con pit_entry_v2")
    if matrix.LapStartTime.isna().any():
        raise ContractError("LapStartTime tiene nulos")


_RACE_CACHE_INDEX: dict[Path, dict[tuple[int, str], list[Path]]] = {}


def race_cache(cache_root: Path, year: int, grand_prix: str) -> Path:
    cache_root = cache_root.resolve()
    if cache_root not in _RACE_CACHE_INDEX:
        index: dict[tuple[int, str], list[Path]] = {}
        for p in cache_root.glob("*/*/*_Race/session_info.ff1pkl"):
            try:
                x = load_fastf1_cache(p)
                info = x[0] if isinstance(x, list) else x
                key = (int(info["StartDate"].year), str(info["Meeting"]["Name"]).casefold())
                index.setdefault(key, []).append(p.parent)
            except Exception:
                continue
        _RACE_CACHE_INDEX[cache_root] = index
    matches = _RACE_CACHE_INDEX[cache_root].get((year, grand_prix.casefold()), [])
    if len(matches) != 1:
        raise ContractError(f"No hay caché Race única para {year}/{grand_prix}: {matches}")
    return matches[0]


def load_race(cache_dir: Path, year: int, gp: str) -> tuple[dict[str, Any], dict[str, str]]:
    names = ["session_info", "timing_app_data", "track_status_data", "weather_data", "session_status_data"]
    data, hashes = {}, {}
    for name in names:
        path = cache_dir / f"{name}.ff1pkl"
        if not path.exists():
            raise ContractError(f"Falta caché necesaria: {path}")
        hashes[str(path)] = sha256(path)
        data[name] = load_fastf1_cache(path)
    info_raw = data["session_info"]
    info = info_raw[0] if isinstance(info_raw, list) else info_raw
    if (str(info.get("Meeting", {}).get("Name", "")).casefold() != gp.casefold()
            or getattr(info.get("StartDate"), "year", None) != year
            or str(info.get("Name", info.get("Type", ""))).casefold() != "race"):
        raise ContractError(f"Identidad de matriz y session_info incompatible: {year}/{gp}, {info}")
    return data, hashes


def _asof_event(frame: pd.DataFrame, decision: float, delay: float = 0.0) -> pd.Series | None:
    if frame.empty or "Time" not in frame:
        return None
    f = frame.copy()
    f["_seconds"] = f.Time.map(seconds)
    f = f.loc[f._seconds + delay <= decision].sort_values("_seconds", kind="stable")
    return None if f.empty else f.iloc[-1]


_STATE_CACHE: dict[int, dict[str, Any]] = {}


def _prepared_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if frame.empty:
        return frame, np.array([], dtype=float)
    result = frame.copy()
    result["_seconds"] = result.Time.map(seconds)
    result = result.sort_values("_seconds", kind="stable").reset_index(drop=True)
    return result, result["_seconds"].to_numpy(dtype=float)


def state_asof(row: pd.Series, all_driver_rows: pd.DataFrame, sources: dict[str, Any],
               delays: dict[str, float], weather_tolerance: float) -> tuple[dict[str, Any], dict[str, str], bool]:
    decision = seconds(row.LapStartTime)
    vals: dict[str, Any] = {"LapNumber": int(row.LapNumber), "Compound_asof": "UNKNOWN", "track_phase_asof": "UNKNOWN"}
    refs: dict[str, str] = {}
    cache_key = id(sources)
    prepared = _STATE_CACHE.get(cache_key)
    if prepared is None:
        timing = sources["timing_app_data"]
        if not isinstance(timing, pd.DataFrame):
            timing = pd.DataFrame()
        track, track_seconds = _prepared_frame(event_frame(sources["track_status_data"]))
        climate, climate_seconds = _prepared_frame(event_frame(sources["weather_data"]))
        session, session_seconds = _prepared_frame(event_frame(sources["session_status_data"]))
        prepared = {
            "sources": sources,
            "timing": timing,
            "track": track, "track_seconds": track_seconds,
            "weather": climate, "weather_seconds": climate_seconds,
            "session": session, "session_seconds": session_seconds,
            "tyre_rows": {},
        }
        _STATE_CACHE[cache_key] = prepared
    timing = prepared["timing"]
    if not timing.empty and {"Driver", "Time", "Stint", "Compound"}.issubset(timing.columns):
        timing_by_driver = prepared.setdefault("timing_by_driver", {})
        driver_key = str(row.DriverNumber)
        td = timing_by_driver.get(driver_key)
        if td is None:
            td = timing.loc[timing.Driver.astype(str) == driver_key].copy()
            if not td.empty:
                td["_seconds"] = td.Time.map(seconds)
                td = td.sort_values("_seconds", kind="stable").reset_index(drop=True)
            timing_by_driver[driver_key] = td
        if not td.empty:
            count = np.searchsorted(td["_seconds"].to_numpy(), decision - delays.get("timing_app_data", 0.0), side="right")
            eligible = td.iloc[:count]
        else:
            eligible = td
        if not eligible.empty:
            active_stint = active_timing_stint(eligible)
            stint_updates = eligible.loc[eligible.Stint.eq(active_stint)] if pd.notna(active_stint) else eligible.iloc[:0]
            # None/NA in a partial message means omitted: keep the last known value in this stint.
            known = stint_updates.loc[stint_updates.Compound.notna() & stint_updates.Compound.astype(str).str.strip().ne("")]
            if not known.empty:
                vals["Compound_asof"] = str(known.iloc[-1].Compound).upper()
                refs["Compound_asof"] = f"timing_app_data:{known.index[-1]}"
            driver_rows_cache = prepared["tyre_rows"].get(id(all_driver_rows))
            if driver_rows_cache is None:
                driver_rows_cache = all_driver_rows.copy()
                driver_rows_cache["_seconds"] = driver_rows_cache.Time.map(seconds)
                driver_rows_cache = driver_rows_cache.sort_values("_seconds", kind="stable").reset_index(drop=True)
                prepared["tyre_rows"][id(all_driver_rows)] = driver_rows_cache
            matrix_stint = matrix_stint_for_timing_stint(active_stint, driver_rows_cache)
            prior = driver_rows_cache.loc[
                driver_rows_cache.Stint.eq(matrix_stint) & driver_rows_cache.TyreLife.notna()
                & ~driver_rows_cache.get("FastF1Generated", pd.Series(False, index=driver_rows_cache.index))
                    .fillna(True).astype(bool)
            ] if matrix_stint is not None else driver_rows_cache.iloc[:0]
            if not prior.empty:
                eligible_count = np.searchsorted(
                    prior["_seconds"].to_numpy(), decision - delays.get("matrix_lap_metrics", 0.0), side="right"
                )
                prior = prior.iloc[:eligible_count]
            if not prior.empty:
                last = prior.sort_values("_seconds", kind="stable").iloc[-1]
                vals["TyreLife_last_eligible"] = float(last.TyreLife)
                refs["TyreLife_last_eligible"] = f"matrix_lap:{int(last.LapNumber)}:TyreLife"
    track, track_seconds = prepared["track"], prepared["track_seconds"]
    track_index = np.searchsorted(track_seconds, decision - delays.get("track_status_data", 0.0), side="right") - 1
    current_track = None if track_index < 0 else track.iloc[track_index]
    if current_track is not None:
        vals["track_phase_asof"] = track_phase(current_track.Status)
        refs["track_phase_asof"] = f"track_status_data:{current_track.name}"
    session, session_seconds = prepared["session"], prepared["session_seconds"]
    session_index = np.searchsorted(session_seconds, decision - delays.get("session_status_data", 0.0), side="right") - 1
    current_session = None if session_index < 0 else session.iloc[session_index]
    # No status event yet is unknown and blocks prediction eligibility.
    aborted = current_session is None or str(current_session.Status).casefold() == "aborted"
    climate, climate_seconds = prepared["weather"], prepared["weather_seconds"]
    weather_index = np.searchsorted(climate_seconds, decision - delays.get("weather_data", 0.0), side="right") - 1
    weather = None if weather_index < 0 else climate.iloc[weather_index]
    if weather is not None and decision - seconds(weather.Time) <= weather_tolerance:
        for src, dst in {
            "AirTemp": "Weather_AirTemp", "Humidity": "Weather_Humidity", "Pressure": "Weather_Pressure",
            "Rainfall": "Weather_Rainfall", "TrackTemp": "Weather_TrackTemp", "WindSpeed": "Weather_WindSpeed",
        }.items():
            if src in weather and pd.notna(weather[src]):
                vals[dst] = float(bool(weather[src])) if src == "Rainfall" else float(weather[src])
                refs[dst] = f"weather_data:{weather.name}"
        if pd.notna(weather.get("WindDirection")):
            angle = math.radians(float(weather.WindDirection))
            vals["Weather_WindDirection_sin"] = math.sin(angle)
            vals["Weather_WindDirection_cos"] = math.cos(angle)
            refs["Weather_WindDirection_sin"] = refs["Weather_WindDirection_cos"] = f"weather_data:{weather.name}"
    return vals, refs, aborted


def build_samples(config_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise ContractError(f"Salida existente; usa una carpeta nueva: {output_dir}")
    root = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    matrix_path, labels_path = root / config["matrix"], root / config["labels"]
    quality_dir = root / config["quality_register"]
    quality_manifest_path = quality_dir / "manifest.json"
    norm_manifest_path = matrix_path.parent / "manifest.json"
    label_manifest_path = labels_path.parent / "manifest.json"
    quality_manifest = json.loads(quality_manifest_path.read_text())
    norm_manifest = json.loads(norm_manifest_path.read_text())
    label_manifest = json.loads(label_manifest_path.read_text())
    matrix_hash = sha256(matrix_path)
    if quality_manifest.get("input", {}).get("sha256") != matrix_hash:
        raise ContractError("Hash de matriz no coincide con registro de calidad v2")
    if quality_manifest.get("quality_register_version") != "historical_quality_contract_v2":
        raise ContractError("Se requiere registro_calidad_historica_v2")
    if quality_manifest.get("input", {}).get("normalization_manifest_sha256") != sha256(norm_manifest_path):
        raise ContractError("Manifiesto normalizado no coincide con registro de calidad v2")
    if norm_manifest.get("normalization_version") != "compound_null_markers_v1":
        raise ContractError("Se requiere matriz Compound normalizada v1")
    if label_manifest.get("rule_version") != "pit_entry_v2" or label_manifest.get("horizon_laps") != 3:
        raise ContractError("Se requiere etiquetado pit_entry_v2 con horizonte de tres vueltas")
    if label_manifest.get("source_sha256") != norm_manifest.get("source", {}).get("sha256_before"):
        raise ContractError("Etiquetas y matriz no proceden de la misma fuente maestra")
    matrix, labels = pd.read_pickle(matrix_path), pd.read_pickle(labels_path)
    validate_inputs(matrix, labels)
    if len(matrix) != norm_manifest["source"]["rows"] or matrix.shape[1] != norm_manifest["source"]["columns"]:
        raise ContractError("Forma de matriz no coincide con manifiesto")
    matrix = matrix.copy()
    matrix["prediction_lap"] = pd.to_numeric(matrix.LapNumber, errors="raise").astype(int)
    labels = labels.rename(columns={"LapNumber": "prediction_lap"})
    if matrix.duplicated(KEY).any() or labels.duplicated(KEY).any():
        raise ContractError("Clave de muestra duplicada")
    labels_idx = labels.set_index(KEY)

    grouped = {(int(y), int(r), str(d)): g.sort_values("LapNumber", kind="stable").copy()
               for (y, r, d), g in matrix.groupby(["Temporada", "Ronda", "DriverNumber"], sort=False)}
    race_data, cache_hashes = {}, {}
    for (year, rnd), race in matrix.groupby(["Temporada", "Ronda"], sort=False):
        gp_names = set(race.GranPremio.astype(str))
        if len(gp_names) != 1:
            raise ContractError(f"GranPremio no constante para {year}/{rnd}")
        gp = next(iter(gp_names))
        cache_path = race_cache(root / "EDA/dataset/cache_fastf1_carreras", int(year), gp)
        data, hashes = load_race(cache_path, int(year), gp)
        race_data[(int(year), int(rnd))] = data
        cache_hashes.update(hashes)

    # Precompute state at each lap start, then evaluate lap-complete values at each decision.
    states, refs_by_key, aborted_by_key = {}, {}, {}
    for group_number, ((year, rnd, driver), group) in enumerate(grouped.items(), start=1):
        if group_number == 1 or group_number % 100 == 0:
            print(f"Reconstruyendo estados temporales: secuencia {group_number}/{len(grouped)}", flush=True)
        sources = race_data[(year, rnd)]
        for _, row in group.iterrows():
            k = (year, rnd, driver, int(row.LapNumber))
            state, refs, aborted = state_asof(
                row, group, sources, config.get("source_delays_seconds", {}),
                float(config.get("weather_tolerance_seconds", 120)),
            )
            states[k], refs_by_key[k], aborted_by_key[k] = state, refs, aborted

    n, w = len(matrix), int(config["window_length"])
    if w < 1:
        raise ContractError("window_length debe ser positivo")
    numeric = np.zeros((n, w, len(NUMERIC)), dtype=np.float32)
    categorical = np.full((n, w, len(CATEGORICAL)), "", dtype="<U32")
    window_laps = np.zeros((n, w), dtype=np.int16)
    padding = np.zeros((n, w), bool)
    slot_exists = np.zeros((n, w), bool)
    row_missing = np.zeros((n, w), bool)
    temporal = np.zeros((n, w, len(FEATURES)), bool)
    feature_missing = np.zeros((n, w, len(FEATURES)), bool)
    reason_code = np.ones((n, w, len(FEATURES)), dtype=np.uint8)
    lineage = np.full((n, w, len(FEATURES)), -1, dtype=np.int32)
    lineage_dict, lineage_rows = {}, []
    sample_rows, static_rows, label_rows = [], [], []
    ordered = matrix.sort_values(["Temporada", "Ronda", "DriverNumber", "LapNumber"], kind="stable").reset_index(drop=True)
    delays = config.get("source_delays_seconds", {})
    lap_maps = {
        key: {int(r.LapNumber): r for _, r in group.iterrows()}
        for key, group in grouped.items()
    }
    for i, sample in ordered.iterrows():
        year, rnd, driver, lap = int(sample.Temporada), int(sample.Ronda), str(sample.DriverNumber), int(sample.LapNumber)
        sample_key = (year, rnd, driver, lap)
        decision = seconds(sample.LapStartTime)
        laps_map = lap_maps[(year, rnd, driver)]
        requested = nominal_window_laps(lap, w)
        first_slot = w - len(requested)
        ineligible_reasons = []
        if aborted_by_key[sample_key]:
            ineligible_reasons.append("session_aborted_or_unknown")
        if states[sample_key].get("track_phase_asof") == "RED":
            ineligible_reasons.append("track_red")
        if bool(sample.get("FastF1Generated", False)):
            ineligible_reasons.append("current_lap_generated")
        for slot, hist_lap in enumerate(requested, start=first_slot):
            if hist_lap < 1:
                window_laps[i, slot] = 0
                padding[i, slot] = True
                reason_code[i, slot, :] = 1
                continue
            window_laps[i, slot] = hist_lap
            slot_exists[i, slot] = True
            hist = laps_map.get(hist_lap)
            if hist is None:
                row_missing[i, slot] = True
                reason_code[i, slot, :] = 2
                continue
            hk = (year, rnd, driver, hist_lap)
            vals = {**states[hk]}
            refs = dict(refs_by_key[hk])
            codes = {f: (0 if f == "LapNumber" else 4) for f in FEATURES}
            codes["LapNumber"] = 0
            close = seconds(hist.Time)
            for field in LAP_METRICS:
                available = lap_metrics_available(
                    hist_lap, lap, close, decision,
                    delays.get(metric_delay_key(field), 0),
                    bool(hist.get("FastF1Generated", False)),
                )
                if not available:
                    vals[field] = np.nan
                    codes[field] = 3
                elif pd.isna(hist.get(field)):
                    vals[field] = np.nan
                    codes[field] = 4
                else:
                    val = hist[field]
                    vals[field] = seconds(val) if field in {"LapTime", "Sector1Time", "Sector2Time", "Sector3Time"} else float(val)
                    codes[field] = 0
                    refs[field] = f"matrix_lap:{hist_lap}:{field}"
            for f in NUMERIC:
                if f not in vals or pd.isna(vals[f]):
                    if f not in codes or codes[f] == 0:
                        codes[f] = 4
                else:
                    codes[f] = 0
            for f in CATEGORICAL:
                vals.setdefault(f, "UNKNOWN")
                codes[f] = 0 if vals[f] != "UNKNOWN" else 5
            for fi, field in enumerate(NUMERIC):
                value, code = vals.get(field, np.nan), codes.get(field, 4)
                if pd.notna(value):
                    numeric[i, slot, fi] = float(value)
                reason_code[i, slot, fi] = code
                temporal[i, slot, fi] = code == 3
                feature_missing[i, slot, fi] = code in (2, 4, 5)
                ref = refs.get(field)
                if ref:
                    ident = (year, rnd, driver, ref, field)
                    if ident not in lineage_dict:
                        lineage_dict[ident] = len(lineage_rows) + 1
                        lineage_rows.append({"lineage_id": lineage_dict[ident], "source_reference": ref,
                                             "Temporada": year, "Ronda": rnd, "DriverNumber": driver,
                                             "feature": field, "rule_version": "causal_sample_v1"})
                    lineage[i, slot, fi] = lineage_dict[ident]
            for ci, field in enumerate(CATEGORICAL):
                categorical[i, slot, ci] = str(vals.get(field, "UNKNOWN"))[:32]
                fi = len(NUMERIC) + ci
                code = codes.get(field, 5)
                reason_code[i, slot, fi] = code
                feature_missing[i, slot, fi] = code in (2, 4, 5)
                ref = refs.get(field)
                if ref:
                    ident = (year, rnd, driver, ref, field)
                    if ident not in lineage_dict:
                        lineage_dict[ident] = len(lineage_rows) + 1
                        lineage_rows.append({"lineage_id": lineage_dict[ident], "source_reference": ref,
                                             "Temporada": year, "Ronda": rnd, "DriverNumber": driver,
                                             "feature": field, "rule_version": "causal_sample_v1"})
                    lineage[i, slot, fi] = lineage_dict[ident]
        eligible = not ineligible_reasons
        sample_rows.append({
            "sample_id": i, "Temporada": year, "Ronda": rnd, "DriverNumber": driver, "prediction_lap": lap,
            "decision_time_seconds": decision, "prediction_eligible": eligible,
            "prediction_ineligible_reason": ";".join(ineligible_reasons),
        })
        static_rows.append({"sample_id": i, "Temporada": year, "Ronda": rnd, "DriverNumber": driver,
                            **{field: str(sample[field]) for field in STATIC}})
        if sample_key in labels_idx.index:
            label = labels_idx.loc[sample_key]
            y = label.PitWithin3Laps
            label_status = "published"
            label_reason = "" if pd.isna(label.get("LabelReason")) else str(label.get("LabelReason"))
            supervised = bool(label.get("EligibleForSupervisedLearning", False)) and pd.notna(y)
        else:
            y, label_status, label_reason, supervised = pd.NA, "not_published", "no_published_label_for_key", False
        label_rows.append({
            "sample_id": i, **dict(zip(KEY, sample_key)), "PitWithin3Laps": y, "label_status": label_status,
            "LabelReason": label_reason, "EligibleForSupervisedLearning": bool(
                label.get("EligibleForSupervisedLearning", False)) if label_status == "published" else False,
            "supervised_eligible": supervised,
        })

    # Create only a new output directory; inputs remain read-only.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    sample_df, static_df, label_df = map(pd.DataFrame, (sample_rows, static_rows, label_rows))
    sample_df.to_csv(output_dir / "sample_index.csv.gz", index=False, compression="gzip")
    static_df.to_csv(output_dir / "static_features.csv.gz", index=False, compression="gzip")
    label_df.to_csv(output_dir / "labels_aligned.csv.gz", index=False, compression="gzip")
    pd.DataFrame(lineage_rows).to_csv(output_dir / "feature_sources.csv.gz", index=False, compression="gzip")
    np.savez_compressed(
        output_dir / "windows.npz", X_numeric=numeric, X_categorical=categorical,
        numeric_feature_names=np.array(NUMERIC, dtype="U64"), categorical_feature_names=np.array(CATEGORICAL, dtype="U64"),
        window_lap_numbers=window_laps, padding_mask=padding, slot_exists_mask=slot_exists,
        slot_row_missing_mask=row_missing, temporal_unavailable_mask=temporal,
        feature_missing_mask=feature_missing, mask_reason_codes=reason_code, lineage_ids=lineage,
    )
    (output_dir / "mask_codes.json").write_text(json.dumps(MASK_REASONS, indent=2) + "\n")
    registry = {
        "static": STATIC, "numeric_sequence": NUMERIC, "categorical_sequence": CATEGORICAL,
        "forbidden_from_X": sorted(FORBIDDEN), "rule_version": "causal_sample_v1",
        "units": {"LapTime": "seconds", "Sector1Time": "seconds", "Sector2Time": "seconds",
                  "Sector3Time": "seconds", "Position": "rank", "TyreLife_last_eligible": "laps",
                  "Weather_AirTemp": "degC", "Weather_Humidity": "percent", "Weather_Pressure": "mbar",
                  "Weather_Rainfall": "boolean_as_0_1", "Weather_TrackTemp": "degC",
                  "Weather_WindDirection_sin": "unitless", "Weather_WindDirection_cos": "unitless",
                  "Weather_WindSpeed": "m/s"},
    }
    (output_dir / "feature_registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    coverage = {
        "samples": len(sample_df), "window_length": w, "labels_by_status": label_df.label_status.value_counts().to_dict(),
        "labels_known": int(label_df.PitWithin3Laps.notna().sum()),
        "positives": int(label_df.PitWithin3Laps.eq(True).sum()),
        "prediction_eligible": int(sample_df.prediction_eligible.sum()),
        "prediction_ineligible": int((~sample_df.prediction_eligible).sum()),
        "padding_slots": int(padding.sum()), "missing_nominal_slots": int(row_missing.sum()),
        "temporally_unavailable_cells": int(temporal.sum()), "feature_missing_cells": int(feature_missing.sum()),
    }
    coverage["prediction_ineligible_reasons"] = sample_df.loc[
        ~sample_df.prediction_eligible, "prediction_ineligible_reason"
    ].value_counts().to_dict()
    coverage["label_reasons"] = label_df.loc[
        label_df.PitWithin3Laps.isna(), "LabelReason"
    ].value_counts(dropna=False).to_dict()
    coverage["labels_by_season"] = {
        str(year): {"rows": int(len(group)), "published": int(group.PitWithin3Laps.notna().sum()),
                    "unknown": int(group.PitWithin3Laps.isna().sum())}
        for year, group in label_df.groupby("Temporada")
    }
    (output_dir / "coverage_report.json").write_text(json.dumps(coverage, indent=2, default=str) + "\n")
    (output_dir / "example_load.py").write_text(
        "from pathlib import Path\nimport pandas as pd\nimport numpy as np\np=Path('muestras_causales_v1')\n"
        "index=pd.read_csv(p/'sample_index.csv.gz')\nstatic=pd.read_csv(p/'static_features.csv.gz')\n"
        "labels=pd.read_csv(p/'labels_aligned.csv.gz')\nwith np.load(p/'windows.npz',allow_pickle=False) as z:\n"
        "    X_numeric=z['X_numeric']; X_categorical=z['X_categorical']; masks=z['feature_missing_mask']\n"
        "print(index.shape, static.shape, labels.shape, X_numeric.shape, X_categorical.shape)\n",
        encoding="utf-8",
    )
    manifest_inputs = [matrix_path, labels_path, norm_manifest_path, quality_manifest_path, label_manifest_path, config_path]
    manifest = {
        "artifact_version": "causal_samples_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_version": config["contract_version"],
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True).stdout.strip(),
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "inputs": {str(p): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in manifest_inputs},
        "cache_sources": cache_hashes, "normalization_version": norm_manifest["normalization_version"],
        "quality_register_version": quality_manifest["quality_register_version"],
        "label_rule_version": label_manifest["rule_version"],
        "time_assumption": "FastF1 event-time historical replay with zero assumed latency; not live receipt latency",
        "rows": len(matrix), "window_length": w,
        "horizon_laps": 3, "source_delays_seconds": delays,
        "specification_sha256": sha256(root / "docs/specs/09_construccion_muestras_causales.md"),
        "builder_sha256": sha256(Path(__file__)),
    }
    manifest["output_files_sha256"] = {
        p.name: sha256(p) for p in output_dir.iterdir() if p.is_file() and p.name != "manifest.json"
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "report.md").write_text(
        f"# Muestras causales v1\n\n- Muestras conservadas: {len(matrix):,}.\n"
        f"- W={w}; horizonte fijo N–N+2.\n- Etiquetas publicadas: {int(label_df.label_status.eq('published').sum()):,}; "
        f"sin etiqueta publicada: {int(label_df.label_status.eq('not_published').sum()):,}; "
        f"publicadas desconocidas: {int((label_df.label_status.eq('published') & label_df.PitWithin3Laps.isna()).sum()):,}.\n"
        "- Replay FastF1 con supuesto de latencia cero; no equivale a latencia de recepción en vivo.\n"
        "- Salida candidata de preparación; no es una matriz aprobada para entrenar.\n"
    )
    # Reopen the written data and ensure row alignment and arrays survive serialization.
    reloaded_index = pd.read_csv(output_dir / "sample_index.csv.gz")
    reloaded_labels = pd.read_csv(output_dir / "labels_aligned.csv.gz", low_memory=False)
    with np.load(output_dir / "windows.npz", allow_pickle=False) as arrays:
        if arrays["X_numeric"].shape != (len(matrix), w, len(NUMERIC)):
            raise ContractError("La forma de X no sobrevivió la recarga")
    if len(reloaded_index) != len(matrix) or len(reloaded_labels) != len(matrix):
        raise ContractError("El número de filas cambió al recargar")
    return output_dir

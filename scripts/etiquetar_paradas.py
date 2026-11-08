"""Build the pit-visit target from FastF1 PitIn/PitOut timing events."""
import hashlib
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "EDA/dataset/datos_carreras_completos_2022_2025"
SOURCE = DATA / "vueltas_2022_2025.pkl"
AUDIT = DATA / "etiquetado_candidate_v1/casos_revision_usuario.csv"
OUTPUT = DATA / "etiquetado_pit_entry_v2_pair_required"
RULE_VERSION = "pit_entry_v2"
SEASONS = [2022, 2023]
SUPPORTED_SEASONS = [2022, 2023, 2024, 2025]
DEFAULT_CONFIG = ROOT / "configs/etiquetado_pit_entry_v3_2022_2025.json"
LEGACY_LABELS = DATA / "etiquetado_pit_entry_v2_pair_required/labels.pkl"
LEGACY_LABELS_SHA256 = "01ad18dd5033b7bb91de29eff5bdcbb8f979b30257c1dd09ec491b1621929750"
MASTER_SOURCE_SHA256 = "9f8eb3f40d66869778c30d75f0a1ebc45196b8ad03c3a4258cc8f5e335dab65c"
KEY = ["Temporada", "Ronda", "DriverNumber", "LapNumber"]
GROUP = ["Temporada", "Ronda", "DriverNumber"]
OFFICIAL_STOP_CONFIRMED = {
    "FIA_stop_recorded_compounds_not_reported",
    "FIA_stop_lap_confirmed_compounds_not_reported",
    "FIA_pit_stop_summary_confirms_stop",
    "FIA_race_report_confirms_pit_entry",
    "FIA_race_lap_analysis_marks_pit_entry",
    "official_report_confirms_pit_entry",
}
OFFICIAL_NO_PIT_VISIT = {
    "FIA_race_lap_analysis_confirms_no_pit_entry_at_candidate_lap",
}
CROSS_SOURCE_STOP_CONFIRMED = {
    "cross_source_confirms_pit_lane_entry",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(source_frame=None, seasons=SEASONS):
    frame = pd.read_pickle(SOURCE) if source_frame is None else source_frame
    frame = frame.loc[frame["Temporada"].isin(seasons)].copy()
    required = KEY + ["GranPremio", "PitInTime", "PitOutTime", "LapStartTime", "Time", "FastF1Generated"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError("Null or duplicated sample keys")
    frame["DriverNumber"] = frame["DriverNumber"].astype(str)
    frame["LapNumber"] = frame["LapNumber"].astype(int)
    return frame.sort_values(GROUP + ["LapNumber"], kind="stable").reset_index(drop=True)


def assert_legacy_invariance(new_labels, old_labels, seasons=(2022, 2023)):
    """Require exact old-split keys, targets, eligibility, reasons and bases."""
    compare_columns = KEY + [
        "PitWithin3Laps", "LabelReason", "LabelBasis", "RuleVersion",
        "EligibleForSupervisedLearning",
    ]
    absent = sorted(set(compare_columns) - set(new_labels.columns) | set(compare_columns) - set(old_labels.columns))
    if absent:
        raise ValueError(f"Cannot compare legacy labels; missing fields: {absent}")
    new = new_labels.loc[new_labels.Temporada.isin(seasons), compare_columns].copy()
    old = old_labels.loc[old_labels.Temporada.isin(seasons), compare_columns].copy()
    new["DriverNumber"] = new.DriverNumber.astype(str)
    old["DriverNumber"] = old.DriverNumber.astype(str)
    new["LapNumber"] = new.LapNumber.astype(int)
    old["LapNumber"] = old.LapNumber.astype(int)
    sort_key = ["Temporada", "Ronda", "DriverNumber", "LapNumber"]
    new = new.sort_values(sort_key, kind="stable").reset_index(drop=True)
    old = old.sort_values(sort_key, kind="stable").reset_index(drop=True)
    if not new[KEY].equals(old[KEY]):
        raise ValueError(
            f"Legacy label key coverage changed: new={len(new)}, legacy={len(old)}; "
            "investigate before publication."
        )
    # Pandas nullable booleans preserve NA equality in assert_frame_equal.
    try:
        pd.testing.assert_frame_equal(new, old, check_dtype=False, check_like=False)
    except AssertionError as exc:
        raise ValueError(f"2022–2023 labels differ from published pit_entry_v2: {exc}") from exc
    return {"seasons": list(seasons), "rows": len(old), "identical": True}


def load_fia_audit(audit_path=AUDIT):
    if not audit_path.exists():
        raise FileNotFoundError(f"Official event audit file not found: {audit_path}")
    audit = pd.read_csv(audit_path, dtype={"case_id": str})
    result = {}
    for row in audit.to_dict("records"):
        if row["manual_review_result"] in OFFICIAL_STOP_CONFIRMED:
            result[row["case_id"]] = "external_entry_claim_confirmed"
        elif row["manual_review_result"] in OFFICIAL_NO_PIT_VISIT:
            result[row["case_id"]] = "external_no_entry_claim"
        elif row["manual_review_result"] in CROSS_SOURCE_STOP_CONFIRMED:
            result[row["case_id"]] = "cross_source_entry_claim"
        else:
            result[row["case_id"]] = "candidate_not_confirmed"
    return audit, result


def _event_id(year, round_number, driver, kind, lap):
    return f"{year}-{int(round_number):02d}-{driver}-{kind}-{int(lap)}"


def review_source_type(url):
    host = urlparse(str(url or "")).hostname or ""
    host = host.lower().removeprefix("www.")
    if host == "fia.com" or host.endswith(".fia.com"):
        return "FIA"
    if host == "formula1.com" or host.endswith(".formula1.com"):
        return "Formula1"
    if not host:
        return "unknown"
    return "other_source"


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def extract_events(frame, fia_audit):
    events, orphan_exits = [], []
    for group_key, group in frame.groupby(GROUP, sort=True, dropna=False):
        entries = group.loc[group["PitInTime"].notna()].sort_values("PitInTime")
        exits = group.loc[group["PitOutTime"].notna()].sort_values("PitOutTime")
        used_exit = set()
        for _, entry in entries.iterrows():
            later = exits.loc[(exits["PitOutTime"] > entry["PitInTime"]) & (~exits.index.isin(used_exit))]
            next_entry = entries.loc[entries["PitInTime"] > entry["PitInTime"], "PitInTime"].min()
            if pd.notna(next_entry):
                later = later.loc[later["PitOutTime"] < next_entry]
            exit_row = later.iloc[0] if len(later) else None
            if exit_row is not None:
                used_exit.add(exit_row.name)

            lap = int(entry["LapNumber"])
            event_id = _event_id(*group_key, "IN", lap)
            official = fia_audit.get(event_id)
            generated = bool(entry["FastF1Generated"])
            if generated:
                status, basis, review, reason = "unknown", "unresolved", "not_verifiable", "generated_pit_in"
            elif exit_row is None or bool(exit_row["FastF1Generated"]):
                # Official evidence can audit an entry, but SPEC-LABEL-002
                # requires a later, paired PitOutTime for a positive label.
                status, basis, review, reason = "unknown", "unresolved", "not_verifiable", "pit_in_without_valid_pit_out"
            elif official == "external_no_entry_claim":
                # The source and the recorded pair disagree; retain for review.
                status, basis, review, reason = "unknown", "unresolved", "not_verifiable", "FIA_feed_pair_conflict"
            else:
                status, basis, review, reason = "pit_stop", "telemetry_event", "not_required", "pit_in_pit_out_pair"

            before = group.loc[group["LapNumber"] == lap - 1]
            after = group.loc[group["LapNumber"] == lap + 1]
            events.append({
                "event_id": event_id, "Temporada": int(group_key[0]), "Ronda": int(group_key[1]),
                "GranPremio": str(entry["GranPremio"]), "DriverNumber": group_key[2],
                "event_lap": lap, "entry_lap": lap, "entry_time": entry["PitInTime"],
                "exit_lap": None if exit_row is None else int(exit_row["LapNumber"]),
                "exit_time": None if exit_row is None else str(exit_row["PitOutTime"]),
                "stint_before": None if before.empty else before.iloc[0].get("Stint"),
                "stint_after": None if after.empty else after.iloc[0].get("Stint"),
                "event_status": status, "evidence_basis": basis, "review_status": review,
                "reason": reason, "rule_version": RULE_VERSION,
                "external_review_status": official or "no_review_record",
            })

        for _, exit_row in exits.iterrows():
            if exit_row.name in used_exit:
                continue
            lap = int(exit_row["LapNumber"])
            orphan_exits.append({
                "Temporada": int(group_key[0]), "Ronda": int(group_key[1]),
                "DriverNumber": group_key[2], "lap": lap,
                "time": exit_row["PitOutTime"],
                "reason": "race_start_release" if lap == 1 else "unpaired_pit_out",
            })
            if lap <= 1:
                continue
            events.append({
                "event_id": _event_id(*group_key, "OUT", lap),
                "Temporada": int(group_key[0]), "Ronda": int(group_key[1]),
                "GranPremio": str(exit_row["GranPremio"]), "DriverNumber": group_key[2],
                # An exit has no defensible entry lap; keep it in the review
                # ledger without assigning it to a prediction horizon.
                "event_lap": pd.NA, "entry_lap": pd.NA, "entry_time": None,
                "exit_lap": lap, "exit_time": str(exit_row["PitOutTime"]),
                "stint_before": None, "stint_after": None,
                "event_status": "unknown",
                "evidence_basis": "unresolved",
                "review_status": "not_verifiable",
                "reason": "orphan_pit_out_has_no_entry_lap",
                "rule_version": RULE_VERSION,
                "external_review_status": "no_review_record",
            })
    event_columns = [
        "event_id", "Temporada", "Ronda", "GranPremio", "DriverNumber", "event_lap",
        "entry_lap", "entry_time", "exit_lap", "exit_time", "stint_before", "stint_after",
        "event_status", "evidence_basis", "review_status", "reason", "rule_version",
        "external_review_status",
    ]
    orphan_columns = ["Temporada", "Ronda", "DriverNumber", "lap", "time", "reason"]
    return (pd.DataFrame.from_records(events, columns=event_columns),
            pd.DataFrame.from_records(orphan_exits, columns=orphan_columns))


def label_horizons(frame, events):
    labels = []
    for group_key, group in frame.groupby(GROUP, sort=False, dropna=False):
        indexed = group.set_index("LapNumber", drop=False)
        driver_events = events.loc[
            (events["Temporada"] == group_key[0])
            & (events["Ronda"] == group_key[1])
            & (events["DriverNumber"] == group_key[2])
        ]
        for _, sample in group.iterrows():
            lap = int(sample["LapNumber"])
            current_ok = pd.notna(sample["LapStartTime"]) and not bool(sample["FastF1Generated"])
            target, reason, basis, eligible = pd.NA, "invalid_current_sample", "unresolved", False
            if current_ok:
                horizon = {lap, lap + 1, lap + 2}
                relevant = driver_events.loc[driver_events["event_lap"].isin(horizon)]
                positive = relevant.loc[relevant["event_status"] == "pit_stop"].copy()
                # Every entry in the horizon must occur after the decision
                # boundary, even if its lap number is later (clock anomalies
                # must not turn an earlier event into a future positive).
                if not positive.empty:
                    entry_times = pd.to_timedelta(positive["entry_time"], errors="coerce")
                    after_decision = entry_times.gt(
                        pd.to_timedelta(sample["LapStartTime"], errors="coerce")
                    )
                    positive = positive.loc[after_decision]
                unresolved = (relevant["event_status"] == "unknown").any()
                if not positive.empty:
                    chosen = positive.iloc[0]
                    target, reason, basis, eligible = True, "pit_stop_in_horizon", chosen["evidence_basis"], True
                else:
                    complete = True
                    for future_lap in sorted(horizon):
                        if future_lap not in indexed.index:
                            complete = False
                            break
                        row = indexed.loc[future_lap]
                        if (isinstance(row, pd.DataFrame) or bool(row["FastF1Generated"])
                                or pd.isna(row["LapStartTime"]) or pd.isna(row["Time"])
                                or row["Time"] < row["LapStartTime"]):
                            complete = False
                            break
                    if unresolved:
                        reason = "unresolved_pit_event"
                    elif complete:
                        target, reason, basis, eligible = False, "complete_horizon_no_pit_stop", "source_coverage", True
                    else:
                        reason = "incomplete_horizon"
            labels.append({
                "Temporada": int(group_key[0]), "Ronda": int(group_key[1]),
                "DriverNumber": group_key[2], "LapNumber": lap,
                "PitWithin3Laps": target, "LabelReason": reason,
                "LabelBasis": basis, "RuleVersion": RULE_VERSION,
                "EligibleForSupervisedLearning": eligible,
            })
    result = pd.DataFrame(labels)
    result["PitWithin3Laps"] = pd.array(result["PitWithin3Laps"], dtype="boolean")
    return result


def main(
    output_name="EDA/dataset/datos_carreras_completos_2022_2025/etiquetado_pit_entry_v3_2022_2025",
    audit_path=AUDIT,
    seasons=(2022, 2023, 2024, 2025),
    source_path=SOURCE,
    expected_source_sha256=MASTER_SOURCE_SHA256,
    reference_labels=LEGACY_LABELS,
    expected_reference_sha256=LEGACY_LABELS_SHA256,
    expected_audit_sha256=None,
    reference_manifest=None,
    expected_reference_manifest_sha256=None,
    config_path=DEFAULT_CONFIG,
):
    output = Path(output_name)
    output = output if output.is_absolute() else ROOT / output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output}")
    seasons = [int(x) for x in seasons]
    if not seasons or len(set(seasons)) != len(seasons) or not set(seasons).issubset(SUPPORTED_SEASONS):
        raise ValueError(f"Seasons must be unique members of {SUPPORTED_SEASONS}; got {seasons}")
    source_path = Path(source_path).resolve()
    reference_labels = Path(reference_labels).resolve()
    audit_path = Path(audit_path).resolve()
    observed_source_sha256 = sha256(source_path)
    if expected_source_sha256 and observed_source_sha256 != expected_source_sha256:
        raise ValueError(
            f"Source SHA-256 mismatch for {source_path}: expected {expected_source_sha256}, "
            f"observed {observed_source_sha256}; review/regenerate config evidence before labeling."
        )
    if expected_reference_sha256 and sha256(reference_labels) != expected_reference_sha256:
        raise ValueError(
            f"Reference label SHA-256 mismatch for {reference_labels}; review the 2022–2023 "
            "invariance reference before publishing."
        )
    if expected_audit_sha256 and sha256(audit_path) != expected_audit_sha256:
        raise ValueError(f"Official audit SHA-256 mismatch: {audit_path}; review config evidence.")
    if reference_manifest is not None:
        reference_manifest = Path(reference_manifest).resolve()
        if expected_reference_manifest_sha256 and sha256(reference_manifest) != expected_reference_manifest_sha256:
            raise ValueError(f"Reference manifest SHA-256 mismatch: {reference_manifest}.")
    source_frame = pd.read_pickle(source_path)
    source_rows_all_seasons = len(source_frame)
    source_columns = len(source_frame.columns)
    frame = load_inputs(source_frame, seasons=seasons)
    audit, fia_audit = load_fia_audit(audit_path)
    events, orphan_exits = extract_events(frame, fia_audit)
    labels = label_horizons(frame, events)
    reference_frame = pd.read_pickle(reference_labels)
    historical_invariance = assert_legacy_invariance(labels, reference_frame)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True)
    config = {
        "rule_version": RULE_VERSION, "source": str(source_path.relative_to(ROOT)),
        "source_sha256": observed_source_sha256, "seasons": seasons,
        "audit_path": str(audit_path.relative_to(ROOT)), "reference_labels": str(reference_labels.relative_to(ROOT)),
        "reference_labels_sha256": sha256(reference_labels), "horizon_laps": 3,
        "output_dir": str(output.relative_to(ROOT)),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    events.to_csv(output / "events.csv", index=False)
    orphan_exits.to_csv(output / "orphan_exits.csv", index=False)
    labels.to_pickle(output / "labels.pkl")
    review_queue = events.loc[events["event_status"] == "unknown"].copy()
    diagnostic_columns = [
        "Temporada", "Ronda", "DriverNumber", "LapNumber", "PitInTime", "PitOutTime",
        "FastF1Generated", "IsAccurate", "Stint", "Compound", "TyreLife", "Position",
        "LapStartTime", "Time",
    ]
    raw_diagnostics = frame[diagnostic_columns].copy()
    raw_diagnostics["DriverNumber"] = raw_diagnostics["DriverNumber"].astype(str)
    review_queue["DriverNumber"] = review_queue["DriverNumber"].astype(str)
    review_queue = review_queue.merge(
        raw_diagnostics,
        left_on=["Temporada", "Ronda", "DriverNumber", "event_lap"],
        right_on=["Temporada", "Ronda", "DriverNumber", "LapNumber"],
        how="left",
        validate="one_to_one",
    )
    audit_columns = [
        "case_id", "official_source_url", "source_title", "evidence_summary",
        "manual_review_result", "manual_review_summary", "manual_review_source_url",
    ]
    audit_context = audit[audit_columns].rename(columns={
        "case_id": "event_id",
        "official_source_url": "candidate_source_url",
        "source_title": "candidate_source_title",
        "evidence_summary": "candidate_evidence_summary",
    })
    review_queue = review_queue.merge(audit_context, on="event_id", how="left", validate="one_to_one")
    review_queue.to_csv(output / "review_queue.csv", index=False)
    coverage = labels.groupby(["Temporada", "Ronda"], as_index=False).agg(
        samples=("LapNumber", "size"), known=("PitWithin3Laps", "count"),
        positives=("PitWithin3Laps", lambda s: int((s == True).sum())),
        negatives=("PitWithin3Laps", lambda s: int((s == False).sum())),
        unknown=("PitWithin3Laps", lambda s: int(s.isna().sum())),
    )
    coverage.to_csv(output / "coverage.csv", index=False)
    season_coverage = labels.groupby("Temporada", as_index=False).agg(
        samples=("LapNumber", "size"), known=("PitWithin3Laps", "count"),
        positives=("PitWithin3Laps", lambda s: int((s == True).sum())),
        negatives=("PitWithin3Laps", lambda s: int((s == False).sum())),
        unknown=("PitWithin3Laps", lambda s: int(s.isna().sum())),
    )
    season_coverage.to_csv(output / "coverage_by_season.csv", index=False)
    decisions = []
    for row in audit.to_dict("records"):
        confirmed = row["manual_review_result"] in OFFICIAL_STOP_CONFIRMED
        cross_confirmed = row["manual_review_result"] in CROSS_SOURCE_STOP_CONFIRMED
        no_visit = row["manual_review_result"] in OFFICIAL_NO_PIT_VISIT
        source = row.get("manual_review_source_url")
        if pd.isna(source) or not source:
            source = row.get("official_source_url")
        decisions.append({
            "event_id": row["case_id"],
            "review_claim": "pit_entry_reported" if confirmed or cross_confirmed else (
                "no_entry_reported" if no_visit else "not_confirmed"
            ),
            "source_type": review_source_type(source),
            "evidence_basis": "cross_source_review" if cross_confirmed else (
                "external_review" if confirmed or no_visit else "unresolved"
            ),
            "review_status": "entry_claim_confirmed" if confirmed else (
                "entry_claim_cross_checked" if cross_confirmed else (
                    "no_entry_claim_confirmed" if no_visit else "not_verifiable"
                )
            ),
            "label_effect": "audit_only; does not replace a valid PitInTime/PitOutTime pair",
            "reason": row["manual_review_result"],
            "source_url": source,
            "source_title": row["source_title"],
            "evidence_summary": row["evidence_summary"],
        })
    (output / "review_decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "rule_version": RULE_VERSION, "target_definition": "pit_visit_recorded_by_paired_pit_in_and_pit_out",
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_anchor": "PitInTime determines entry lap; only a later paired PitOutTime makes a positive visit",
        "initial_lap_pit_out": "ignored_as_race_start_release",
        "unpaired_pit_in": "unknown_even_if_external_source_reports_entry",
        "orphan_pit_out": "review_only_without_assigned_entry_lap",
        "time_assumption": "FastF1 event Time supports historical replay only; does not establish live receipt latency.",
        "horizon_laps": 3, "prediction_instant": "lap_start", "seasons": seasons,
        "source": str(source_path.relative_to(ROOT)), "source_sha256": observed_source_sha256,
        "config": str(Path(config_path).resolve().relative_to(ROOT)) if config_path else None,
        "config_sha256": sha256(Path(config_path).resolve()) if config_path and Path(config_path).exists() else None,
        "reference_labels": str(reference_labels.relative_to(ROOT)),
        "reference_labels_sha256": sha256(reference_labels),
        "reference_manifest": str(reference_manifest.relative_to(ROOT)) if reference_manifest else None,
        "reference_manifest_sha256": sha256(reference_manifest) if reference_manifest else None,
        "historical_2022_2023_invariance": historical_invariance,
        "source_rows_all_seasons": int(source_rows_all_seasons),
        "source_columns": int(source_columns),
        "official_event_audit": str(audit_path.relative_to(ROOT)), "official_event_audit_sha256": sha256(audit_path),
        "script_sha256": sha256(Path(__file__).resolve()),
        "spec_sha256": sha256(ROOT / "docs/specs/07_objetivo_entrada_boxes.md"),
        "git_revision": git_revision(),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "events": len(events), "pit_stops": int((events["event_status"] == "pit_stop").sum()),
        "unresolved_events": int((events["event_status"] == "unknown").sum()),
        "review_cases": len(audit),
        "FIA_entry_claims": sum(x["source_type"] == "FIA" and x["review_claim"] == "pit_entry_reported" for x in decisions),
        "other_source_entry_claims": sum(x["source_type"] != "FIA" and x["review_claim"] == "pit_entry_reported" for x in decisions),
        "cross_source_entry_claims": sum(x["review_status"] == "entry_claim_cross_checked" for x in decisions),
        "review_queue_rows": len(review_queue), "review_queue_sha256": sha256(output / "review_queue.csv"),
        "labels": len(labels), "positives": int((labels["PitWithin3Laps"] == True).sum()),
        "negatives": int((labels["PitWithin3Laps"] == False).sum()),
        "unknown_labels": int(labels["PitWithin3Laps"].isna().sum()),
        "unknown_events_by_reason": {
            str(k): int(v) for k, v in events.loc[
                events.event_status.eq("unknown"), "reason"
            ].value_counts().items()
        },
        "labels_by_season": {
            str(int(row.Temporada)): {
                "samples": int(row.samples), "known": int(row.known), "positives": int(row.positives),
                "negatives": int(row.negatives), "unknown": int(row.unknown),
            }
            for row in season_coverage.itertuples(index=False)
        },
        "training_status": "candidate_for_preprocessing_not_model_approved",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = (
        "# Etiquetado candidato: entrada a boxes\n\n"
        "La etiqueta predice si se registra una visita a boxes dentro de las tres vueltas "
        "incluyendo la actual. No exige comprobar un cambio de neumáticos. `PitInTime` "
        "ancla el evento; `PitOutTime` lo empareja. Una entrada sin salida queda desconocida "
        "aunque una fuente externa la reporte; esas fuentes auditan discrepancias y no sustituyen la pareja.\n\n"
        f"- Eventos detectados: {manifest['events']} ({manifest['pit_stops']} clasificados; "
        f"{manifest['unresolved_events']} sin resolver).\n"
        f"- Casos revisados: {manifest['review_cases']}; reclamos de entrada respaldados por FIA: {manifest['FIA_entry_claims']}; "
        f"por otras fuentes: {manifest['other_source_entry_claims']}; cruce de fuentes: {manifest['cross_source_entry_claims']}.\n"
        f"- Casos pendientes con diagnóstico local: {manifest['review_queue_rows']} (`review_queue.csv`).\n"
        f"- Etiquetas: {manifest['positives']} positivas, {manifest['negatives']} negativas, "
        f"{manifest['unknown_labels']} desconocidas.\n"
        f"- Temporadas cubiertas: {', '.join(map(str, seasons))}.\n"
        f"- Invariancia 2022–2023 frente a pit_entry_v2: {historical_invariance['rows']} claves idénticas.\n"
        "- No se entrenó ningún modelo ni se ajustaron preprocesadores.\n"
    )
    (output / "report.md").write_text(report)
    (output / "reconciliation.json").write_text(json.dumps({
        "historical_invariance": historical_invariance,
        "source_rows_selected": len(frame), "labels_published": len(labels),
        "missing_label_keys": 0, "duplicate_label_keys": int(labels.duplicated(KEY).sum()),
    }, indent=2) + "\n")
    manifest["output_files_sha256"] = {
        p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(report)
    print(f"Artefactos: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    main(
        output_name=config["output_dir"], audit_path=ROOT / config["audit_path"],
        seasons=config["seasons"], source_path=ROOT / config["source"],
        expected_source_sha256=config["source_sha256"],
        reference_labels=ROOT / config["reference_labels"],
        expected_reference_sha256=config["reference_labels_sha256"], config_path=config_path,
        expected_audit_sha256=config.get("audit_sha256"),
        reference_manifest=ROOT / config["reference_manifest"] if config.get("reference_manifest") else None,
        expected_reference_manifest_sha256=config.get("reference_manifest_sha256"),
    )

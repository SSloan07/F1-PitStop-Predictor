"""Serve a causal, lap-timed replay of 2024 validation predictions."""
from __future__ import annotations
import argparse
from bisect import bisect_right
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import pickle
import threading

import numpy as np
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse
from zipfile import ZipFile
from io import BytesIO

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from normalizar_compound import load_normalized_matrix

DATA = ROOT / "EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025_compound_normalizado_v1/vueltas_2022_2025_compound_normalizado.pkl"
DATA_SHA256 = "02c15d7b2d766e1d01604cf143bb288a2e584370f0f422766683866a10ca00e0"
DATA_MANIFEST = DATA.parent / "manifest.json"
DATA_MANIFEST_SHA256 = "7d982955db05afaacbaf03d467d7dd6d9c9e06987327e24a0f247c7002acea2d"
EVENTS = ROOT / "EDA/dataset/datos_carreras_completos_2022_2025/etiquetado_pit_entry_v3_2022_2025/events.csv"
EVENTS_SHA256 = "ada92a965c77f8705dfdedca3459b7d9fd6d01313166dfc407ca4951fcb5d516"
SAMPLE_INDEX = ROOT / "EDA/dataset/muestras_causales_v1_3/sample_index.csv.gz"
SAMPLE_INDEX_SHA256 = "263155d033e65c0042d98383aa77877067c8a125adecfe4ea757a6296da817eb"
APP = ROOT / "Frontend/app"
CATALOG_PATH = ROOT / "configs/simulador_modelos_2024.json"
CACHE_ROOT = ROOT / "EDA/dataset/cache_fastf1_carreras/2024"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seconds(value):
    if pd.isna(value):
        return None
    if isinstance(value, timedelta):
        return value.total_seconds()
    return pd.to_timedelta(value).total_seconds()


def clean_driver(value) -> str:
    return str(value).removesuffix(".0")


def finite(value):
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not (-float("inf") < value < float("inf")):
        return None
    return value




class TelemetryReplay:
    """Lazy, point-in-time coordinates and speed from local FastF1 race caches."""
    def __init__(self, cache_dirs):
        self.cache_dirs = cache_dirs
        self._cache = {}
        self._lock = threading.RLock()

    @staticmethod
    def _read(path):
        with path.open("rb") as stream:
            wrapped = pickle.load(stream)
        return wrapped.get("data", wrapped) if isinstance(wrapped, dict) else wrapped

    @staticmethod
    def _arrays(frame, fields, anchor=None):
        if not {"Date", "Time", *fields}.issubset(frame.columns):
            return None
        date = pd.to_datetime(frame["Date"], errors="coerce")
        date_valid = date.notna().to_numpy()
        date_epoch = date.astype("int64").to_numpy(dtype=np.float64) / 1e9
        source_time = pd.to_timedelta(frame["Time"], errors="coerce").dt.total_seconds().to_numpy(dtype=np.float64)
        if anchor is None:
            offsets_mask = date_valid & np.isfinite(source_time)
            return date_epoch, source_time, offsets_mask
        session_time = date_epoch - anchor
        result = {"time": session_time.astype(np.float32)}
        for name in fields:
            result[name.lower()] = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float32)
        valid = date_valid & np.isfinite(session_time)
        for name in fields:
            valid &= np.isfinite(result[name.lower()])
        if "X" in fields and "Y" in fields:
            valid &= (result["x"] != 0) | (result["y"] != 0)
        for name in result:
            result[name] = result[name][valid]
        if not len(result["time"]):
            return None
        order = np.argsort(result["time"], kind="stable")
        for name in result:
            result[name] = result[name][order]
        return result

    def _load(self, round_number, race_rows, events):
        with self._lock:
            return self._load_unlocked(round_number, race_rows, events)

    def _load_unlocked(self, round_number, race_rows, events):
        if round_number in self._cache:
            return self._cache[round_number]
        folder = self.cache_dirs.get(round_number)
        if folder is None:
            raise RuntimeError(f"No local FastF1 telemetry cache for round {round_number}")
        position_frames = self._read(folder / "position_data.ff1pkl")
        car_frames = self._read(folder / "car_data.ff1pkl")
        offset_samples = []
        for frames in (position_frames, car_frames):
            for frame in frames.values():
                if not isinstance(frame, pd.DataFrame) or not {"Date", "Time"}.issubset(frame.columns):
                    continue
                date = pd.to_datetime(frame["Date"], errors="coerce")
                date_valid = date.notna().to_numpy()
                epoch = date.astype("int64").to_numpy(dtype=np.float64) / 1e9
                session_time = pd.to_timedelta(frame["Time"], errors="coerce").dt.total_seconds().to_numpy(dtype=np.float64)
                valid = date_valid & np.isfinite(session_time)
                indexes = np.flatnonzero(valid)
                if len(indexes):
                    stride = max(1, len(indexes) // 250)
                    offset_samples.extend((epoch[indexes[::stride]] - session_time[indexes[::stride]]).tolist())
        if not offset_samples:
            raise RuntimeError(f"No timestamps in telemetry cache for round {round_number}")
        anchor = float(np.median(np.asarray(offset_samples, dtype=np.float64)))
        positions = {}
        speeds = {}
        for driver, frame in position_frames.items():
            if isinstance(frame, pd.DataFrame):
                arrays = self._arrays(frame, ("X", "Y"), anchor)
                if arrays is not None:
                    positions[clean_driver(driver)] = arrays
        for driver, frame in car_frames.items():
            if isinstance(frame, pd.DataFrame):
                arrays = self._arrays(frame, ("Speed",), anchor)
                if arrays is not None:
                    speeds[clean_driver(driver)] = arrays
        metadata = {}
        for row in race_rows:
            driver = clean_driver(row["DriverNumber"])
            metadata.setdefault(driver, {"driver": str(row["Driver"]), "team": str(row.get("Team") or "")})
        track_points = []
        for row in sorted(race_rows, key=lambda x: (x["lap"], x["DriverNumber"])):
            duration = seconds(row.get("Time")) - seconds(row.get("LapStartTime"))
            if not 40 <= duration <= 180:
                continue
            series = positions.get(clean_driver(row["DriverNumber"]))
            if series is None:
                continue
            start = seconds(row.get("LapStartTime"))
            end = seconds(row.get("Time"))
            left = int(np.searchsorted(series["time"], start, side="left"))
            right = int(np.searchsorted(series["time"], end, side="right"))
            if right - left < 80:
                continue
            track_points = self._decimate(series["x"][left:right], series["y"][left:right], 700)
            if len(track_points) >= 40:
                break
        pit_lane = []
        for event in events:
            driver = clean_driver(event["DriverNumber"])
            series = positions.get(driver)
            entry = finite(event.get("entry_seconds"))
            exit_time = finite(event.get("exit_seconds"))
            if series is None or entry is None or exit_time is None or exit_time <= entry:
                continue
            left = int(np.searchsorted(series["time"], max(0.0, entry - 4.0), side="left"))
            right = int(np.searchsorted(series["time"], exit_time + 4.0, side="right"))
            if right - left >= 25:
                pit_lane = self._decimate(series["x"][left:right], series["y"][left:right], 180)
                break
        stops = {}
        for event in events:
            driver = clean_driver(event["DriverNumber"])
            series = speeds.get(driver)
            entry = finite(event.get("entry_seconds"))
            exit_time = finite(event.get("exit_seconds"))
            if series is None or entry is None or exit_time is None or exit_time <= entry:
                continue
            left = int(np.searchsorted(series["time"], entry, side="left"))
            right = int(np.searchsorted(series["time"], exit_time, side="right"))
            if right <= left:
                continue
            local_speeds = series["speed"][left:right]
            stopped = np.flatnonzero(local_speeds <= 2.0)
            if not len(stopped):
                continue
            start_index = left + int(stopped[0])
            start_time = float(series["time"][start_index])
            moving_after = np.flatnonzero((series["time"][start_index + 1:right] > start_time) & (series["speed"][start_index + 1:right] > 2.0))
            end_time = float(series["time"][start_index + 1 + int(moving_after[0])]) if len(moving_after) else min(exit_time, float(series["time"][right - 1]))
            stops[str(event["event_id"])] = {
                "driver": driver, "entry_lap": int(event["entry_lap"]), "entry": entry,
                "exit": exit_time, "start": start_time, "end": max(start_time, end_time),
            }
        loaded = {"positions": positions, "speeds": speeds, "metadata": metadata,
                  "track": {"points": track_points, "pit_lane": pit_lane}, "stops": stops}
        self._cache = {round_number: loaded}
        return loaded

    @staticmethod
    def _decimate(xs, ys, maximum):
        if len(xs) <= maximum:
            indexes = np.arange(len(xs))
        else:
            indexes = np.linspace(0, len(xs) - 1, maximum).astype(np.int64)
        return [[float(xs[i]), float(ys[i])] for i in indexes]

    def track(self, round_number, race_rows, events):
        data = self._load(round_number, race_rows, events)
        return {"round": round_number, **data["track"]}

    def snapshot(self, round_number, now, since, race_rows, events, risks, event_matches):
        data = self._load(round_number, race_rows, events)
        leaderboard = {clean_driver(row["DriverNumber"]): row for row in race_rows if float(row["time_seconds"]) <= now}
        risk_by_driver = {risk["driver"]: risk for risk in risks}
        actual_by_driver = {}
        for event in events:
            entry = finite(event.get("entry_seconds"))
            exit_time = finite(event.get("exit_seconds"))
            if entry is not None and exit_time is not None and entry <= now <= exit_time:
                actual_by_driver[clean_driver(event["DriverNumber"])] = event
        cars = []
        for driver, series in data["positions"].items():
            index = bisect_right(series["time"], now) - 1
            if index < 0 or now - float(series["time"][index]) > 5.0:
                continue
            x = float(series["x"][index]); y = float(series["y"][index])
            speed = data["speeds"].get(driver)
            speed_value = None
            if speed is not None:
                speed_index = bisect_right(speed["time"], now) - 1
                if speed_index >= 0 and now - float(speed["time"][speed_index]) <= 5.0:
                    speed_value = float(speed["speed"][speed_index])
            meta = data["metadata"].get(driver, {"driver": driver, "team": ""})
            lap_row = leaderboard.get(driver)
            risk = risk_by_driver.get(driver, {})
            actual = actual_by_driver.get(driver)
            stopped = any(stop["driver"] == driver and stop["start"] <= now <= stop["end"] for stop in data["stops"].values())
            cars.append({
                "driver": meta["driver"], "driver_number": driver, "team": meta["team"],
                "x": x, "y": y, "position": None if lap_row is None else finite(lap_row.get("position")),
                "lap": None if lap_row is None else int(lap_row["lap"]), "speed_kph": speed_value,
                "alert": bool(risk.get("alert", False)), "probability": finite(risk.get("probability")),
                "in_pit": actual is not None, "stopped": stopped,
                "pit_lap": None if actual is None else int(actual["entry_lap"]),
            })
        cars.sort(key=lambda item: (float("inf") if item["position"] is None else item["position"], item["driver_number"]))
        new_stops = []
        for event_id, stop in data["stops"].items():
            if since < stop["start"] <= now:
                new_stops.append({
                    "event_id": event_id, "driver": data["metadata"].get(stop["driver"], {"driver": stop["driver"]})["driver"],
                    "entry_lap": stop["entry_lap"], "stop_seconds": stop["start"],
                    "duration_seconds": max(0.0, stop["end"] - stop["start"]),
                    "predicted": event_matches.get(event_id, False),
                })
        new_stops.sort(key=lambda item: item["stop_seconds"])
        return {"cars": cars, "new_actual_stops": new_stops}


class ReplayStore:
    def __init__(self):
        for path, digest in ((DATA, DATA_SHA256), (DATA_MANIFEST, DATA_MANIFEST_SHA256), (EVENTS, EVENTS_SHA256), (SAMPLE_INDEX, SAMPLE_INDEX_SHA256)):
            if not path.is_file() or sha256(path) != digest:
                raise RuntimeError(f"Missing or changed pinned replay source: {path.relative_to(ROOT)}")
        self.models = {}
        catalog = json.loads(CATALOG_PATH.read_text())
        self.model_public = []
        expected_sample_ids = None
        sample_index = pd.read_csv(SAMPLE_INDEX, dtype={"DriverNumber": str})
        sample_index = sample_index.loc[sample_index.Temporada.eq(2024), ["sample_id", "decision_time_seconds"]]
        time_by_sample = dict(zip(sample_index.sample_id.astype(int), sample_index.decision_time_seconds.astype(float)))

        for entry in catalog["models"]:
            self._load_model(entry, time_by_sample)
            ids = set(self.models[entry["id"]]["predictions_by_sample"])
            if expected_sample_ids is None:
                expected_sample_ids = ids
            elif ids != expected_sample_ids:
                raise RuntimeError("Model prediction populations differ; refusing incomparable replay")
            self.model_public.append(self.models[entry["id"]]["public"])

        laps = load_normalized_matrix(DATA)
        laps = laps.loc[laps.Temporada.eq(2024)].copy()
        if laps.duplicated(["Temporada", "Ronda", "DriverNumber", "LapNumber"]).any():
            raise RuntimeError("Replay position source has duplicate driver-lap keys")
        laps["DriverNumber"] = laps.DriverNumber.map(clean_driver)
        laps["lap"] = pd.to_numeric(laps.LapNumber, errors="coerce")
        laps["time_seconds"] = laps.Time.map(seconds)
        laps["position"] = pd.to_numeric(laps.Position, errors="coerce")
        laps["tyre_life"] = pd.to_numeric(laps.TyreLife, errors="coerce")
        self.race_rows = {}
        for round_number, frame in laps.groupby("Ronda", sort=True):
            frame = frame.loc[frame.time_seconds.notna() & frame.lap.notna()].copy()
            frame.sort_values(["time_seconds", "DriverNumber", "lap"], inplace=True)
            self.race_rows[int(round_number)] = frame.to_dict("records")

        events = pd.read_csv(EVENTS, dtype={"DriverNumber": str}, keep_default_na=False)
        events = events.loc[events.Temporada.eq(2024) & events.event_status.eq("pit_stop")].copy()
        events["DriverNumber"] = events.DriverNumber.map(clean_driver)
        events["entry_seconds"] = events.entry_time.map(seconds)
        events["exit_seconds"] = events.exit_time.map(seconds)
        self.race_events = {}
        for round_number, frame in events.groupby("Ronda", sort=True):
            rows = frame.sort_values("entry_seconds").to_dict("records")
            self.race_events[int(round_number)] = rows

        self.races = []
        for round_number, frame in laps.groupby("Ronda", sort=True):
            round_number = int(round_number)
            rows = self.race_rows[round_number]
            times = [float(row["time_seconds"]) for row in rows]
            total_laps = int(pd.to_numeric(frame.LapNumber, errors="coerce").max())
            self.races.append({
                "round": round_number,
                "name": str(frame.GranPremio.dropna().iloc[0]),
                "total_laps": total_laps,
                "start_seconds": 0.0,
                "end_seconds": max(times),
                "evaluable_events": len(self.race_events.get(round_number, [])),
            })
        if len(self.races) != 24:
            raise RuntimeError(f"Expected 24 validation races in 2024; got {len(self.races)}")
        self.race_by_round = {race["round"]: race for race in self.races}
        race_round_by_name = {race["name"].casefold(): race["round"] for race in self.races}
        cache_dirs = {}
        for info_path in CACHE_ROOT.glob("*/*_Race/session_info.ff1pkl"):
            try:
                info = self._load_fastf1_cache(info_path)
                name = str(info["Meeting"]["Name"]).casefold()
                if name in race_round_by_name:
                    cache_dirs[race_round_by_name[name]] = info_path.parent
            except Exception as exc:
                raise RuntimeError(f"Invalid FastF1 session cache: {info_path.relative_to(ROOT)}") from exc
        if len(cache_dirs) != 24:
            raise RuntimeError(f"Expected telemetry caches for 24 validation races; got {len(cache_dirs)}")
        self.telemetry = TelemetryReplay(cache_dirs)

    @staticmethod
    def _load_fastf1_cache(path):
        with path.open("rb") as stream:
            cached = pickle.load(stream)
        return cached.get("data", cached) if isinstance(cached, dict) else cached

    def _load_model(self, entry, time_by_sample):
        zip_path = ROOT / entry["zip"]
        if not zip_path.is_file() or sha256(zip_path) != entry["sha256"]:
            raise RuntimeError(f"Missing or changed model artifact: {entry['zip']}")
        child = entry["child"]
        with ZipFile(zip_path) as archive:
            names = set(archive.namelist())
            if "pair_manifest.json" not in names or "pair_summary.json" not in names:
                raise RuntimeError(f"Model artifact is not a completed pair: {entry['zip']}")
            pair_manifest = json.loads(archive.read("pair_manifest.json"))
            pair_summary = json.loads(archive.read("pair_summary.json"))
            if pair_manifest.get("test_used") is not False or pair_manifest.get("comparison_authorized") is not True:
                raise RuntimeError(f"Model pair failed its selection/test-use gate: {entry['id']}")
            child_manifest = json.loads(archive.read(f"{child}/manifest.json"))
            child_status = json.loads(archive.read(f"{child}/status.json"))
            if child_status.get("status") != "complete" or child_manifest.get("test_used") is not False:
                raise RuntimeError(f"Incomplete or test-contaminated model child: {entry['id']}")
            for relative, digest in child_manifest["files_sha256"].items():
                member = f"{child}/{relative}"
                if member not in names or hashlib.sha256(archive.read(member)).hexdigest() != digest:
                    raise RuntimeError(f"Child artifact hash mismatch: {entry['id']}/{relative}")
            metrics_doc = json.loads(archive.read(f"{child}/metrics.json"))
            selected = metrics_doc["selected_threshold"]
            selected_metrics = selected.get("metrics", selected)
            threshold = float(json.loads(archive.read(f"{child}/threshold.json"))["threshold"])
            predictions = pd.read_csv(BytesIO(archive.read(f"{child}/validation_predictions.csv.gz")), compression="gzip", dtype={"DriverNumber": str})
            if not predictions.Temporada.eq(2024).all() or predictions.sample_id.duplicated().any():
                raise RuntimeError(f"Invalid validation prediction population: {entry['id']}")
            if len(predictions) != 25635:
                raise RuntimeError(f"Unexpected validation row count for {entry['id']}: {len(predictions)}")
            rows = []
            by_sample = {}
            by_race_driver = {}
            for row in predictions[["sample_id", "Ronda", "DriverNumber", "prediction_lap", "probability", "alert"]].to_dict("records"):
                sample_id = int(row["sample_id"])
                if sample_id not in time_by_sample:
                    raise RuntimeError(f"No causal timestamp for sample {sample_id}")
                item = {
                    "sample_id": sample_id,
                    "round": int(row["Ronda"]),
                    "driver": clean_driver(row["DriverNumber"]),
                    "lap": int(row["prediction_lap"]),
                    "decision_seconds": float(time_by_sample[sample_id]),
                    "probability": float(row["probability"]),
                    "alert": float(row["probability"]) >= threshold,
                }
                rows.append(item)
                by_sample[sample_id] = item
                by_race_driver.setdefault((item["round"], item["driver"]), []).append(item)
            for key in by_race_driver:
                by_race_driver[key].sort(key=lambda item: (item["decision_seconds"], item["lap"]))
            model_spec = json.loads(archive.read(f"{child}/model_spec.json"))
            public = {
                "id": entry["id"], "name": entry["name"], "family": entry["family"],
                "architecture": entry["architecture"], "objective": entry["objective"],
                "threshold": threshold, "seed": int(child_manifest.get("seed", pair_manifest.get("seed", 42))),
                "f1": float(selected_metrics["f1"]), "f2": float(selected_metrics["f2"]),
                "precision": float(selected_metrics["precision"]), "recall": float(selected_metrics["recall"]),
                "average_precision": float(selected_metrics["average_precision"]),
                "parameter_count": int(child_manifest["parameter_count"]),
                "best_epoch": int(child_manifest["best_epoch"]),
                "run_id": pair_summary.get("pair_id", entry["id"]),
                "artifact_sha256": entry["sha256"],
                "validation_scope": "2024 · development validation",
                "model_spec": model_spec,
            }
            self.models[entry["id"]] = {
                "public": public, "predictions": rows, "predictions_by_sample": by_sample,
                "by_race_driver": by_race_driver, "events_cache": {},
            }

    @staticmethod
    def _public_position(row):
        return {
            "driver": str(row["Driver"]), "driver_number": clean_driver(row["DriverNumber"]),
            "position": finite(row["position"]), "lap": int(row["lap"]),
            "time_seconds": float(row["time_seconds"]), "team": str(row.get("Team") or ""),
            "compound": str(row.get("Compound") or "UNKNOWN"), "tyre_life": finite(row.get("tyre_life")),
            "track_status": str(row.get("TrackStatus") or ""),
        }

    def snapshot(self, model_id, round_number, now, since):
        if model_id not in self.models or round_number not in self.race_by_round:
            raise ValueError("Unknown model or race")
        race = self.race_by_round[round_number]
        now = max(0.0, min(float(now), float(race["end_seconds"])))
        since = max(-1.0, min(float(since), now))
        all_rows = self.race_rows[round_number]
        visible_laps = [row for row in all_rows if float(row["time_seconds"]) <= now]
        latest_by_driver = {}
        for row in visible_laps:
            latest_by_driver[clean_driver(row["DriverNumber"])] = row
        leaderboard = [self._public_position(row) for row in latest_by_driver.values()]
        leaderboard.sort(key=lambda item: (float("inf") if item["position"] is None else item["position"], item["driver"]))
        current_lap = max((item["lap"] for item in leaderboard), default=0)

        model = self.models[model_id]
        current_risks = []
        new_alerts = []
        for (rnd, driver), rows in model["by_race_driver"].items():
            if rnd != round_number:
                continue
            times = [row["decision_seconds"] for row in rows]
            index = bisect_right(times, now) - 1
            if index < 0:
                continue
            prediction = rows[index]
            state = latest_by_driver.get(driver)
            driver_lap = current_lap if state is None else state["lap"]
            active = bool(prediction["alert"] and driver_lap <= prediction["lap"] + 2)
            current_risks.append({
                "driver": driver, "prediction_lap": prediction["lap"],
                "probability": prediction["probability"], "threshold": model["public"]["threshold"],
                "alert": active, "decision_seconds": prediction["decision_seconds"],
                "horizon_end_lap": prediction["lap"] + 2,
            })
            start_index = bisect_right(times, since)
            for candidate_index, candidate in enumerate(rows[start_index:index + 1], start=start_index):
                was_alert = candidate_index > 0 and rows[candidate_index - 1]["alert"]
                if candidate["alert"] and not was_alert:
                    new_alerts.append({
                        "driver": driver, "prediction_lap": candidate["lap"],
                        "probability": candidate["probability"], "threshold": model["public"]["threshold"],
                        "horizon_end_lap": candidate["lap"] + 2,
                        "decision_seconds": candidate["decision_seconds"],
                    })
        current_risks.sort(key=lambda item: (-item["probability"], item["driver"]))
        new_alerts.sort(key=lambda item: item["decision_seconds"])

        all_events = self.race_events.get(round_number, [])
        due_events = [event for event in all_events if since < float(event["entry_seconds"]) <= now]
        visible_events = [event for event in all_events if float(event["entry_seconds"]) <= now]
        for event in due_events:
            event["predicted"] = self._event_covered(model, round_number, event)
        public_events = [{
            "event_id": str(event["event_id"]), "driver": str(event["DriverNumber"]),
            "entry_lap": int(event["entry_lap"]), "entry_seconds": float(event["entry_seconds"]),
            "predicted": self._event_covered(model, round_number, event),
        } for event in visible_events[-20:]]
        summary = None
        if now >= race["end_seconds"]:
            events = all_events
            evaluable = []
            covered = 0
            for event in events:
                opportunities = [p for p in model["by_race_driver"].get((round_number, clean_driver(event["DriverNumber"])), [])
                    if p["lap"] <= int(event["entry_lap"]) <= p["lap"] + 2
                    and p["decision_seconds"] < float(event["entry_seconds"])]
                if opportunities:
                    evaluable.append(event)
                    if any(p["alert"] for p in opportunities):
                        covered += 1
            summary = {"confirmed_pit_events": len(events), "evaluable_events": len(evaluable),
                       "covered_events": covered, "event_recall": None if not evaluable else covered / len(evaluable),
                       "scope": "2024 validation; development result, not an independent test"}
        event_matches = {str(event["event_id"]): self._event_covered(model, round_number, event)
                         for event in all_events if float(event["entry_seconds"]) <= now}
        telemetry_snapshot = self.telemetry.snapshot(
            round_number, now, since, all_rows, all_events, current_risks, event_matches,
        )
        return {
            "model_id": model_id, "race": race, "time_seconds": now, "current_lap": current_lap,
            "leaderboard": leaderboard, "risks": current_risks, "new_alerts": new_alerts,
            "actual_pits": public_events,
            "new_actual_pits": [{"event_id": str(e["event_id"]), "driver": str(e["DriverNumber"]),
                                 "entry_lap": int(e["entry_lap"]), "entry_seconds": float(e["entry_seconds"]),
                                 "predicted": bool(e["predicted"])} for e in due_events],
            "summary": summary, "cars": telemetry_snapshot["cars"],
            "new_actual_stops": telemetry_snapshot["new_actual_stops"],
        }

    def _event_covered(self, model, round_number, event):
        event_lap = int(event["entry_lap"])
        candidates = [p for p in model["by_race_driver"].get((round_number, clean_driver(event["DriverNumber"])), [])
            if p["lap"] <= event_lap <= p["lap"] + 2 and p["decision_seconds"] < float(event["entry_seconds"])]
        return any(p["alert"] for p in candidates)


class Handler(BaseHTTPRequestHandler):
    store = None
    server_version = "PitwallReplay/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._api(parsed.path, parse_qs(parsed.query))
            return
        self._static(parsed.path)

    def _api(self, route, query):
        try:
            if route == "/api/bootstrap":
                self._json({"season": 2024, "scope": "development validation", "models": self.store.model_public,
                            "races": self.store.races, "default_round": 11,
                            "notice": "Predicciones de validación reproducidas a su decision_time. La temporada 2025 queda reservada y no participa en el replay."})
            elif route == "/api/track":
                round_number = int(query.get("round", ["11"])[0])
                if round_number not in self.store.race_by_round:
                    raise ValueError("Unknown race")
                track = self.store.telemetry.track(
                    round_number, self.store.race_rows[round_number],
                    self.store.race_events.get(round_number, []),
                )
                self._json(track)
            elif route == "/api/snapshot":
                model_id = query.get("model_id", [""])[0]
                round_number = int(query.get("round", ["11"])[0])
                now = float(query.get("time", ["0"])[0])
                since = float(query.get("since", ["-1"])[0])
                self._json(self.store.snapshot(model_id, round_number, now, since))
            else:
                self._json({"error": "Not found"}, status=404)
        except (ValueError, KeyError) as exc:
            self._json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._json({"error": f"Replay error: {type(exc).__name__}: {exc}"}, status=500)

    def _json(self, obj, status=200):
        payload = json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _static(self, route):
        relative = "index.html" if route in ("", "/") else route.removeprefix("/")
        target = (APP / relative).resolve()
        if not target.is_relative_to(APP.resolve()) or not target.is_file():
            self.send_error(404)
            return
        content = target.read_bytes()
        mime = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8"}.get(target.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    store = ReplayStore()
    Handler.store = store
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"Replay 2024: http://{args.host}:{args.port}")
    print(f"Modelos disponibles: {len(store.model_public)} · carreras: {len(store.races)} · 2025: reservada, sin inferencia ni evaluación")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

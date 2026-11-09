"""Serve a causal, lap-timed replay of 2024 validation predictions."""
from __future__ import annotations
import argparse
from bisect import bisect_right
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
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
        return {
            "model_id": model_id, "race": race, "time_seconds": now, "current_lap": current_lap,
            "leaderboard": leaderboard, "risks": current_risks, "new_alerts": new_alerts,
            "actual_pits": public_events,
            "new_actual_pits": [{"event_id": str(e["event_id"]), "driver": str(e["DriverNumber"]),
                                 "entry_lap": int(e["entry_lap"]), "entry_seconds": float(e["entry_seconds"]),
                                 "predicted": bool(e["predicted"])} for e in due_events],
            "summary": summary,
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

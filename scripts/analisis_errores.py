"""Error analysis of the best validation-2024 models (3-seed LightGBM and the 50/50 ensemble).

No model is trained or re-selected: the script reads frozen validation
predictions and the thresholds already chosen in docs/aclaraciones/evidencia_ensamble_v1.
It answers:

1. False positives: how far is the next real pit entry? (near misses vs pure false alarms)
2. False negatives / missed events: track status, stop order, stint length, lead time.
3. Where the model fails: per race and by situation.
4. A complementary tolerance metric: alerts whose next entry is 1-2 laps beyond
   the 3-lap horizon are counted as early (not false) alarms.

2025 is never loaded.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_pitstop.experiment_metrics import binary_metrics
from f1_pitstop.rnn_data import read_json, sha256, write_json
from f1_pitstop.rnn_experiment import attach_decision_times, event_coverage
from f1_pitstop.strategy_features import median_stint_length_train

ENSEMBLE = ROOT / "docs/aclaraciones/evidencia_ensamble_v1/resultados"
STRATEGY = ROOT / "docs/aclaraciones/evidencia_gbm_v1/variables_estrategia_validacion.csv.gz"
EVENTS = ROOT / "EDA/dataset/datos_carreras_completos_2022_2025/etiquetado_pit_entry_v3_2022_2025/events.csv"
LAPS = ROOT / "EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025_compound_normalizado_v1/vueltas_2022_2025_compound_normalizado.pkl"
SAMPLE_INDEX = ROOT / "EDA/dataset/muestras_causales_v1_3/sample_index.csv.gz"
MODELS = ("gbm_media_3", "ensamble_50_50")
KEYS = ["sample_id", "Temporada", "Ronda", "DriverNumber", "prediction_lap"]
SC_CODES = ("4", "6", "7")


def _driver(series):
    return series.astype(str).str.removesuffix(".0")


def next_entry_distance(predictions, events):
    """Laps from the predicted lap N to the next pit entry (any status) with entry_lap >= N."""
    entries = events[["Ronda", "DriverNumber", "entry_lap", "event_status"]].copy()
    entries["DriverNumber"] = _driver(entries.DriverNumber)
    distance = np.full(len(predictions), np.nan)
    status = np.full(len(predictions), "none", dtype=object)
    grouped = {key: g.sort_values("entry_lap") for key, g in entries.groupby(["Ronda", "DriverNumber"])}
    for i, row in enumerate(predictions[["Ronda", "DriverNumber", "prediction_lap"]].itertuples(index=False)):
        group = grouped.get((row.Ronda, row.DriverNumber))
        if group is None:
            continue
        later = group.loc[group.entry_lap >= row.prediction_lap]
        if len(later):
            distance[i] = later.entry_lap.iloc[0] - row.prediction_lap
            status[i] = later.event_status.iloc[0]
    return distance, status


def bucket(distance):
    labels = np.full(len(distance), "sin_parada_posterior", dtype=object)
    finite = np.isfinite(distance)
    for low, high, name in ((0, 2, "0-2 (horizonte)"), (3, 3, "+3"), (4, 4, "+4"), (5, 6, "+5 a +6"),
                            (7, 10, "+7 a +10"), (11, 10_000, "+11 o más")):
        labels[finite & (distance >= low) & (distance <= high)] = name
    return labels


def event_context(events, laps, stint_median):
    """Per-event descriptors: track status on the in-lap, stop order, stint length relative to typical."""
    laps = laps.loc[laps.Temporada == 2024, ["Ronda", "DriverNumber", "LapNumber", "TrackStatus", "TyreLife",
                                             "Compound", "PitInTime"]].copy()
    laps["DriverNumber"] = _driver(laps.DriverNumber)
    laps["LapNumber"] = laps.LapNumber.astype(int)
    indexed = laps.set_index(["Ronda", "DriverNumber", "LapNumber"])
    rows = []
    stops = events.loc[events.event_status.eq("pit_stop")].copy()
    stops["DriverNumber"] = _driver(stops.DriverNumber)
    stops = stops.sort_values(["Ronda", "DriverNumber", "entry_lap"])
    stops["stop_number"] = stops.groupby(["Ronda", "DriverNumber"]).cumcount() + 1
    stops["stops_total"] = stops.groupby(["Ronda", "DriverNumber"]).entry_lap.transform("size")
    for row in stops.itertuples(index=False):
        key = (row.Ronda, row.DriverNumber, int(row.entry_lap))
        lap = indexed.loc[key] if key in indexed.index else None
        status = str(lap.TrackStatus) if lap is not None else ""
        tyre = float(lap.TyreLife) if lap is not None and pd.notna(lap.TyreLife) else np.nan
        compound = lap.Compound if lap is not None else None
        typical = stint_median.get(compound, np.nan)
        rows.append({"event_id": row.event_id, "stop_number": int(row.stop_number),
                     "is_last_stop": bool(row.stop_number == row.stops_total),
                     "sc_vsc_on_in_lap": any(code in status for code in SC_CODES),
                     "tyre_life_at_entry": tyre, "compound": compound,
                     "stint_vs_typical": tyre / typical if typical else np.nan,
                     "early_stop": bool(np.isfinite(tyre) and typical and tyre / typical < 0.5)})
    return pd.DataFrame(rows)


def tolerant_metrics(y, alert, distance):
    """FP whose next entry is +3/+4 laps away are counted as early alarms, not false ones."""
    tp = int((alert & (y == 1)).sum())
    fp = alert & (y == 0)
    early = fp & np.isin(distance, [3, 4])
    fp_strict = int(fp.sum())
    fp_tolerant = fp_strict - int(early.sum())
    fn = int((~alert & (y == 1)).sum())
    precision = tp / (tp + fp_tolerant) if tp + fp_tolerant else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return {"early_alarms_plus3_plus4": int(early.sum()), "fp_strict": fp_strict, "fp_tolerant": fp_tolerant,
            "precision_tolerant": precision, "recall": recall, "f2_tolerant": f2}


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    metrics = read_json(ENSEMBLE / "metrics.json")["results"]
    predictions = pd.read_csv(ENSEMBLE / "predicciones_validacion.csv.gz", dtype={"DriverNumber": str})
    predictions["DriverNumber"] = _driver(predictions.DriverNumber)
    strategy = pd.read_csv(STRATEGY, dtype={"DriverNumber": str})
    if not np.array_equal(strategy.sample_id.to_numpy(), predictions.sample_id.to_numpy()):
        raise ValueError("Strategy features are not aligned with predictions")
    events = pd.read_csv(EVENTS, dtype={"DriverNumber": str}, keep_default_na=False)
    events = events.loc[events.Temporada == 2024].copy()
    events["DriverNumber"] = _driver(events.DriverNumber)
    laps = pd.read_pickle(LAPS)
    laps = laps.loc[laps.Temporada.isin([2022, 2023, 2024])]
    stint_median = median_stint_length_train(laps)
    sample_index = pd.read_csv(SAMPLE_INDEX, dtype={"DriverNumber": str})
    distance, next_status = next_entry_distance(predictions, events)
    contexts = event_context(events, laps, stint_median)
    y = predictions.target.to_numpy()
    results = {}
    output.mkdir(parents=True)
    for model in MODELS:
        threshold = metrics[model]["threshold"]
        probability = predictions[model].to_numpy()
        alert = probability >= threshold
        frame = predictions[KEYS + ["target"]].copy()
        frame["probability"], frame["alert"] = probability, alert
        frame["laps_to_next_entry"], frame["next_entry_status"] = distance, next_status
        frame["distance_bucket"] = bucket(distance)
        frame = pd.concat([frame, strategy[["stops_done", "laps_since_last_stop", "tyre_life_vs_typical_stint",
                                            "race_fraction"]]], axis=1)
        fp = frame.loc[frame.alert & (frame.target == 0)]
        fp_buckets = fp.distance_bucket.value_counts().reindex(
            ["+3", "+4", "+5 a +6", "+7 a +10", "+11 o más", "sin_parada_posterior"], fill_value=0)
        recent_stop = int((fp.laps_since_last_stop <= 2).sum())
        # Event level.
        table = attach_decision_times(frame[KEYS + ["target", "probability", "alert"]], sample_index)
        event_table, event_metrics = event_coverage(events, table)
        event_table = event_table.merge(contexts, on="event_id", how="left", validate="one_to_one")

        def coverage(mask):
            subset = event_table.loc[mask & (event_table.evaluable_opportunities > 0)]
            return {"events": int(len(subset)), "covered": int(subset.covered.sum()),
                    "coverage": float(subset.covered.mean()) if len(subset) else None}

        by_situation = {
            "bajo_sc_vsc": coverage(event_table.sc_vsc_on_in_lap), "en_verde": coverage(~event_table.sc_vsc_on_in_lap),
            "primera_parada": coverage(event_table.stop_number == 1),
            "parada_intermedia": coverage((event_table.stop_number > 1) & ~event_table.is_last_stop),
            "ultima_parada_de_varias": coverage((event_table.stop_number > 1) & event_table.is_last_stop),
            "parada_temprana_stint_menor_50pct": coverage(event_table.early_stop),
            "stint_normal_o_largo": coverage(~event_table.early_stop),
        }
        covered = event_table.loc[event_table.covered]
        lead = covered.maximum_lead_laps.value_counts().sort_index().to_dict()
        by_race = []
        for (rnd,), group in frame.groupby(["Ronda"]):
            race_metrics = binary_metrics(group.target, group.probability, threshold)
            race_events = event_table.loc[event_table.Ronda == rnd]
            by_race.append({"Ronda": int(rnd), "samples": len(group), "positives": int(group.target.sum()),
                            "f2": race_metrics["f2"], "precision": race_metrics["precision"],
                            "recall": race_metrics["recall"], "average_precision": race_metrics["average_precision"],
                            "events": int(len(race_events)), "covered_events": int(race_events.covered.sum())})
        gp = laps.loc[laps.Temporada == 2024].groupby("Ronda").GranPremio.first()
        by_race = pd.DataFrame(by_race)
        by_race.insert(1, "GranPremio", by_race.Ronda.map(gp))
        by_race.sort_values("f2").to_csv(output / f"{model}_por_carrera.csv", index=False)
        event_table.to_csv(output / f"{model}_eventos.csv", index=False)
        results[model] = {
            "threshold": threshold, "strict": {k: metrics[model][k] for k in ("f2", "precision", "recall", "fp", "fn")},
            "false_positives": {"total": int(len(fp)), "by_laps_to_next_entry": fp_buckets.to_dict(),
                                "share_plus3_plus4": float(fp_buckets[["+3", "+4"]].sum() / len(fp)),
                                "share_within_plus6": float(fp_buckets[["+3", "+4", "+5 a +6"]].sum() / len(fp)),
                                "within_2_laps_after_own_stop": recent_stop,
                                "next_entry_unknown_status": int((fp.next_entry_status == "unknown").sum())},
            "tolerance_plus2_laps": tolerant_metrics(y, alert, distance),
            "events": {**event_metrics, "by_situation": by_situation,
                       "maximum_lead_laps_of_covered": {int(k): int(v) for k, v in lead.items()}},
            "worst_races_by_f2": by_race.nsmallest(4, "f2")[["GranPremio", "f2", "recall", "covered_events", "events"]].to_dict("records"),
            "best_races_by_f2": by_race.nlargest(4, "f2")[["GranPremio", "f2", "recall", "covered_events", "events"]].to_dict("records"),
        }
        print(model, "FP:", fp_buckets.to_dict(), "| tolerante F2",
              round(results[model]["tolerance_plus2_laps"]["f2_tolerant"], 4), flush=True)
    write_json(output / "metrics.json", {"models": results,
                                         "note": "Frozen validation-2024 predictions and thresholds; no retraining or re-selection."})
    write_json(output / "manifest.json", {
        "artifact_version": "error_analysis_v1",
        "inputs_sha256": {"predictions": sha256(ENSEMBLE / "predicciones_validacion.csv.gz"),
                          "ensemble_metrics": sha256(ENSEMBLE / "metrics.json"), "strategy": sha256(STRATEGY),
                          "events": sha256(EVENTS), "laps": sha256(LAPS), "sample_index": sha256(SAMPLE_INDEX),
                          "script": sha256(Path(__file__))},
        "files_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    })
    print(f"Resultados: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/aclaraciones/evidencia_errores_v1")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()

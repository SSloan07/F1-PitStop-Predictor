"""Evaluate the frozen final LightGBM model, following docs/specs/22.

Two modes share exactly the same code path:

* ``--split validation`` (dry run, 2024): rebuilds the inputs from the raw
  causal windows with the frozen preprocessor, recomputes the strategy
  features, predicts with the frozen boosters and **requires** that the
  transformed arrays equal the published validation package and that the
  probabilities equal the frozen validation predictions. It then writes the
  same report that the test run will write.
* ``--split test`` (2025, single run): same pipeline on the reserved season.
  It requires ``--confirmar-evaluacion-unica`` and refuses an existing output
  directory. Nothing is re-fitted: no preprocessing statistics, threshold,
  parameters or features are learned from 2025.

Before anything else the script checks that the frozen model files and every
source file recorded in the freeze manifest are byte-identical.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from analisis_errores import bucket, event_context, next_entry_distance, tolerant_metrics
from diagnostico_gbm import CACHE, LAPS, PACKAGE, PACKAGE_MANIFEST_SHA256, SAMPLE_INDEX, base_features
from f1_pitstop.experiment_metrics import binary_metrics
from f1_pitstop.preprocessing import InputPreprocessor
from f1_pitstop.rnn_data import MODEL_ARRAYS, SplitData, load_split, read_json, sha256, verify_preprocessed, write_json
from f1_pitstop.rnn_experiment import event_coverage
from f1_pitstop.strategy_features import median_stint_length_train, scheduled_laps, strategy_features
from f1_pitstop.strategy_features_r2 import pit_loss_table, strategy_features_round2
from f1_pitstop.strategy_features_r3 import stop_priors, strategy_features_round3

FROZEN = ROOT / "modelos/lightgbm_final_v1"
FROZEN_MANIFEST_SHA256 = "e23908f3b6b0e6a4c077d53d2b2498aa9ee526a81744267ff9db56ea275f5f06"
FEATURES_DIR = ROOT / "EDA/dataset/muestras_causales_v1_3"
PARTITIONS_DIR = ROOT / "EDA/dataset/muestras_causales_v1_3_etiquetas_v3_particiones_v1_1"
EVENTS = ROOT / "EDA/dataset/datos_carreras_completos_2022_2025/etiquetado_pit_entry_v3_2022_2025/events.csv"
SPLITS = {"validation": (2024, "validation"), "test": (2025, "test_final_reserved")}
KEYS = ["sample_id", "Temporada", "Ronda", "DriverNumber", "prediction_lap"]
MASKS = ["padding_mask", "slot_exists_mask", "slot_row_missing_mask",
         "temporal_unavailable_mask", "feature_missing_mask", "mask_reason_codes"]
PARITY_ATOL = 1e-12


def verify_frozen():
    manifest_path = FROZEN / "manifest.json"
    if sha256(manifest_path) != FROZEN_MANIFEST_SHA256:
        raise ValueError("Frozen model manifest differs from the pre-registered one")
    manifest = read_json(manifest_path)
    for name, digest in manifest["files_sha256"].items():
        if sha256(FROZEN / name) != digest:
            raise ValueError(f"Frozen model file changed: {name}")
    for name, digest in manifest["sources_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Frozen source changed since the freeze: {name}")
    return read_json(FROZEN / "model.json"), read_json(FROZEN / "features.json")


def split_inputs(split, state):
    """Transform the raw causal windows of one split with the frozen preprocessor (no fitting)."""
    season, split_name = SPLITS[split]
    preprocessed_manifest = read_json(PACKAGE / "manifest.json")
    if sha256(FEATURES_DIR / "manifest.json") != preprocessed_manifest["source_hashes"]["feature_manifest"]:
        raise ValueError("Causal feature package differs from the one used by the frozen preprocessor")
    if sha256(PARTITIONS_DIR / "manifest.json") != preprocessed_manifest["source_hashes"]["partition_manifest"]:
        raise ValueError("Partition package differs from the one used by the frozen preprocessor")
    preprocessor = InputPreprocessor.from_dict(state)
    sample_index = pd.read_csv(FEATURES_DIR / "sample_index.csv.gz", dtype={"DriverNumber": str})
    partitions = pd.read_csv(PARTITIONS_DIR / "sample_partitions.csv.gz", dtype={"DriverNumber": str})
    aligned = pd.read_csv(PARTITIONS_DIR / "labels_aligned_v3.csv.gz", dtype={"DriverNumber": str},
                          keep_default_na=False, low_memory=False)
    if not (partitions.sample_id.equals(sample_index.sample_id) and aligned.sample_id.equals(sample_index.sample_id)):
        raise ValueError("Partition/label/sample order mismatch")
    rows = partitions.split.eq(split_name).to_numpy()
    if not partitions.loc[rows, "Temporada"].eq(season).all():
        raise ValueError(f"Split {split_name} is not exclusively season {season}")
    ids = partitions.loc[rows, "sample_id"].to_numpy(dtype=np.int64)
    registry = read_json(FEATURES_DIR / "feature_registry.json")
    static_frame = pd.read_csv(FEATURES_DIR / "static_features.csv.gz", low_memory=False)
    static = static_frame.loc[static_frame.sample_id.isin(ids), registry["static"]].to_numpy(dtype=str)
    if not np.array_equal(static_frame.loc[static_frame.sample_id.isin(ids), "sample_id"].to_numpy(), ids):
        raise ValueError("Static context order mismatch")
    with np.load(FEATURES_DIR / "windows.npz", allow_pickle=False) as archive:
        masks = {name: np.asarray(archive[name][ids]) for name in MASKS}
        result = preprocessor.transform(archive["X_numeric"][ids], archive["X_categorical"][ids], static, masks,
                                        state["schema"])
    arrays = {"sample_id": ids, "X_numeric": result["X_numeric"],
              "X_dynamic_categorical_ids": result["X_dynamic_categorical_ids"],
              "X_static_categorical_ids": result["X_static_categorical_ids"],
              **{name: result[name] for name in MASKS}}
    metadata = partitions.loc[rows].reset_index(drop=True)
    labels = aligned.loc[rows].reset_index(drop=True)
    eligible = labels.supervised_eligible.astype(str).str.lower().eq("true").to_numpy()
    if not np.array_equal(eligible, metadata.supervised_eligible.astype(str).str.lower().eq("true").to_numpy()):
        raise ValueError("Eligibility differs between partitions and labels")
    positions = np.flatnonzero(eligible)
    targets_text = labels.iloc[positions].PitWithin3Laps.astype(str).str.lower()
    if not targets_text.isin(["true", "false"]).all():
        raise ValueError("An eligible sample has no binary target")
    targets = targets_text.eq("true").to_numpy(dtype=np.float32)
    data = SplitData(arrays, metadata, labels, positions, targets)
    if split == "validation":
        published = load_split(PACKAGE, "validation", state)
        for name in MODEL_ARRAYS:
            if not np.array_equal(arrays[name], published.arrays[name]):
                raise ValueError(f"Dry run: rebuilt {name} differs from the published validation package")
        if not (np.array_equal(positions, published.positions) and np.array_equal(targets, published.targets)):
            raise ValueError("Dry run: eligible population or targets differ from the published package")
    return data, sample_index


def feature_frame(data, state, sample_index, season, columns):
    samples = data.metadata.iloc[data.positions][KEYS].copy()
    samples["DriverNumber"] = samples.DriverNumber.astype(str).str.removesuffix(".0")
    decision = sample_index[["sample_id", "decision_time_seconds"]]
    samples = samples.merge(decision, on="sample_id", how="left", validate="one_to_one").reset_index(drop=True)
    if samples.decision_time_seconds.isna().any():
        raise ValueError("Missing causal decision times")
    seasons = tuple(range(2022, season + 1))
    laps = pd.read_pickle(LAPS)
    laps = laps.loc[laps.Temporada.isin(seasons)].copy()
    total_laps = scheduled_laps(CACHE, seasons)
    stint_median = median_stint_length_train(laps)
    race_stops, long_stint = stop_priors(laps)
    base, _ = base_features(data, state)
    parts = [base.reset_index(drop=True)]
    for table in (strategy_features(samples, laps, total_laps, stint_median),
                  strategy_features_round2(samples, laps, total_laps, stint_median, pit_loss_table(laps)),
                  strategy_features_round3(samples, laps, total_laps, race_stops, long_stint)):
        if not np.array_equal(table.sample_id.to_numpy(), samples.sample_id.to_numpy()):
            raise ValueError("Strategy features out of sample order")
        parts.append(table.drop(columns="sample_id").reset_index(drop=True))
    frame = pd.concat(parts, axis=1)
    missing = [c for c in columns if c not in frame]
    if missing:
        raise ValueError(f"Frozen features missing from the rebuilt frame: {missing}")
    return frame[columns], samples, laps, stint_median


def report(samples, y, probability, threshold, laps, stint_median, sample_index, season):
    alert = probability >= threshold
    metrics = binary_metrics(y, probability, threshold)
    events = pd.read_csv(EVENTS, dtype={"DriverNumber": str}, keep_default_na=False)
    events = events.loc[events.Temporada == season].copy()
    events["DriverNumber"] = events.DriverNumber.astype(str).str.removesuffix(".0")
    table = samples[KEYS + ["decision_time_seconds"]].copy()
    table["target"], table["probability"], table["alert"] = y.astype(int), probability, alert
    event_table, event_metrics = event_coverage(events, table)
    contexts = event_context(events, laps, stint_median, season=season)
    event_table = event_table.merge(contexts, on="event_id", how="left", validate="one_to_one")

    def coverage(mask):
        subset = event_table.loc[mask & (event_table.evaluable_opportunities > 0)]
        return {"events": int(len(subset)), "covered": int(subset.covered.sum()),
                "coverage": float(subset.covered.mean()) if len(subset) else None}

    sc, early = event_table.sc_vsc_on_in_lap.fillna(False).astype(bool), event_table.early_stop.fillna(False).astype(bool)
    distance, _ = next_entry_distance(table, events)
    fp_buckets = pd.Series(bucket(distance)[alert & (y == 0)]).value_counts().to_dict()
    by_race = []
    for rnd, group in table.groupby("Ronda"):
        race = binary_metrics(group.target, group.probability, threshold)
        by_race.append({"Ronda": int(rnd), "samples": len(group), "positives": int(group.target.sum()),
                        **{k: race[k] for k in ("f2", "precision", "recall", "average_precision")}})
    by_race = pd.DataFrame(by_race)
    gp = laps.loc[laps.Temporada == season].groupby("Ronda").GranPremio.first()
    by_race.insert(1, "GranPremio", by_race.Ronda.map(gp))
    return {
        "threshold": threshold,
        "primary_f2": metrics["f2"],
        "metrics": {k: metrics[k] for k in ("average_precision", "f2", "f1", "precision", "recall", "tp", "fp",
                                            "fn", "tn", "n_samples", "positives")},
        "event_coverage": {**event_metrics, "by_type": {
            "verde_stint_normal": coverage(~sc & ~early), "sc_vsc_stint_normal": coverage(sc & ~early),
            "stint_corto": coverage(early)}},
        "tolerance_plus3_plus4": tolerant_metrics(y.astype(int), alert, distance),
        "false_positives_by_laps_to_next_entry": fp_buckets,
        "per_race_f2": {"min": float(by_race.f2.min()), "median": float(by_race.f2.median()),
                        "max": float(by_race.f2.max())},
    }, table, event_table, by_race


def run(split, output, confirmed):
    import lightgbm as lgb
    output = Path(output)
    if split == "test" and not confirmed:
        raise SystemExit("La evaluación de 2025 es única: añade --confirmar-evaluacion-unica para ejecutarla.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; the evaluation must not be repeated")
    model, features = verify_frozen()
    season = SPLITS[split][0]
    _, state = verify_preprocessed(PACKAGE, PACKAGE_MANIFEST_SHA256)
    data, sample_index = split_inputs(split, state)
    X, samples, laps, stint_median = feature_frame(data, state, sample_index, season, features["order"])
    boosters = [lgb.Booster(model_file=str(FROZEN / f"lightgbm_seed{seed}.txt")) for seed in model["seeds"]]
    probability = np.mean([booster.predict(X) for booster in boosters], axis=0)
    if split == "validation":
        frozen = pd.read_csv(FROZEN / "validation_predictions.csv.gz")
        if not np.array_equal(frozen.sample_id.to_numpy(), samples.sample_id.to_numpy()):
            raise ValueError("Dry run: sample order differs from frozen predictions")
        delta = float(np.max(np.abs(frozen.probability.to_numpy() - probability)))
        if delta > PARITY_ATOL:
            raise ValueError(f"Dry run: probabilities differ from frozen predictions (max {delta})")
        print(f"Ensayo 2024: entradas y probabilidades idénticas a las congeladas (diferencia máx. {delta:.1e})")
    results, table, event_table, by_race = report(samples, data.targets, probability, model["threshold"],
                                                  laps, stint_median, sample_index, season)
    output.mkdir(parents=True)
    table.to_csv(output / "predicciones.csv.gz", index=False, compression="gzip")
    event_table.to_csv(output / "eventos.csv", index=False)
    by_race.to_csv(output / "por_carrera.csv", index=False)
    write_json(output / "metrics.json", {"split": split, "season": season, **results,
                                         "frozen_manifest_sha256": FROZEN_MANIFEST_SHA256,
                                         "protocol": "docs/specs/22_modelo_final_y_protocolo_test_2025.md",
                                         "refit_on_this_split": False})
    write_json(output / "manifest.json", {
        "artifact_version": "final_model_evaluation_v1", "split": split,
        "inputs_sha256": {"frozen_manifest": FROZEN_MANIFEST_SHA256, "laps": sha256(LAPS),
                          "sample_index": sha256(SAMPLE_INDEX), "events": sha256(EVENTS),
                          "script": sha256(Path(__file__))},
        "files_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    })
    m = results["metrics"]
    print(f"{split} {season}: F2 {m['f2']:.4f} AP {m['average_precision']:.4f} precisión {m['precision']:.3f} "
          f"recall {m['recall']:.3f} | eventos {results['event_coverage']['covered_events']}/"
          f"{results['event_coverage']['events_with_evaluable_opportunity']}")
    print(f"Resultados: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confirmar-evaluacion-unica", action="store_true")
    args = parser.parse_args()
    default = {"validation": ROOT / "docs/aclaraciones/evaluacion_final_ensayo_2024",
               "test": ROOT / "docs/aclaraciones/evaluacion_final_2025"}[args.split]
    run(args.split, args.output or default, args.confirmar_evaluacion_unica)


if __name__ == "__main__":
    main()

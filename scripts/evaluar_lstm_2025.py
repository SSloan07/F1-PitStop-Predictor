"""Post-hoc comparison on 2025: the starting LSTM and the frozen strategy LSTM.

Declared after the LightGBM test run (docs/aclaraciones/35). Nothing is fitted
or re-thresholded on 2025:

* ``lstm_partida``: LSTM32 dropout 0.2, BCE pos_weight=2, seed 42, full v1_3
  inputs (runs/lstm32_pos_weight2_f2_cpu_v1_seed42.zip), its own 2024 threshold.
* ``lstm_estrategia``: frozen three-seed strategy LSTM32 (modelos/lstm32_estrategia_v1),
  mean of seeds at the threshold fixed on 2024.

Inputs are rebuilt from the raw causal windows with the frozen v1_3 preprocessor,
the frozen telemetry projection and the frozen strategy standardization. The
same code first rebuilds 2024 and must reproduce the published validation
packages exactly and the exported validation probabilities (parity 1e-5).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from diagnostico_gbm import CACHE, LAPS, PACKAGE, PACKAGE_MANIFEST_SHA256, SAMPLE_INDEX
from evaluar_modelo_final import FEATURES_DIR, KEYS, report, split_inputs
from preparar_entradas_estrategia import PRESENT, TEMPORAL, slot_values
from preparar_entradas_telemetria import derive_arrays, derive_state
from f1_pitstop.rnn_data import MODEL_ARRAYS, SplitData, load_split, read_json, sha256, verify_preprocessed, write_json
from f1_pitstop.rnn_experiment import _build_model, _loader, predict
from f1_pitstop.strategy_features import median_stint_length_train, scheduled_laps, strategy_features

TELEMETRY = ROOT / "EDA/dataset/muestras_preprocesadas_v1_3_telemetria_v1"
STRATEGY = ROOT / "EDA/dataset/muestras_preprocesadas_v1_3_telemetria_estrategia_v1"
FROZEN_LSTM = ROOT / "modelos/lstm32_estrategia_v1"
START_ZIP = ROOT / "runs/lstm32_pos_weight2_f2_cpu_v1_seed42.zip"
START_ZIP_SHA256 = "ca1b2bb5adcc784910c6b56063acf2c8ce1591b157b36b3cf915dc814de6d481"
START_CHILD = "lstm32_pos_weight2"
PARITY_ATOL = 1e-5


def telemetry_split(data, state_v13):
    config = read_json(TELEMETRY / "config.json")
    derived_state, numeric_index, dynamic_index, mask_index = derive_state(state_v13, config)
    _, published_state = verify_preprocessed(TELEMETRY)
    if derived_state != published_state:
        raise ValueError("Telemetry projection differs from the published frozen state")
    arrays = derive_arrays(data.arrays, numeric_index, dynamic_index, mask_index)
    return SplitData(arrays, data.metadata, data.labels, data.positions, data.targets), derived_state


def strategy_split(data, sample_index, season):
    """Append the nine round-1 strategy features per slot with the frozen train standardization."""
    config = read_json(STRATEGY / "config.json")
    names = config["strategy_features"]
    _, state = verify_preprocessed(STRATEGY)
    seasons = tuple(range(2022, season + 1))
    laps = pd.read_pickle(LAPS)
    laps = laps.loc[laps.Temporada.isin(seasons)].copy()
    index = sample_index.loc[sample_index.Temporada.isin(seasons)].copy()
    index["DriverNumber"] = index.DriverNumber.astype(str).str.removesuffix(".0")
    samples = index[["sample_id", "Temporada", "Ronda", "DriverNumber", "prediction_lap", "decision_time_seconds"]]
    samples = samples.loc[samples.Temporada.eq(season)]
    table = strategy_features(samples, laps, scheduled_laps(CACHE, seasons), median_stint_length_train(laps))
    table = samples.merge(table, on="sample_id", how="left", validate="one_to_one")
    table = table[["Temporada", "Ronda", "DriverNumber", "prediction_lap", "decision_time_seconds", *names]].rename(
        columns={"prediction_lap": "slot_lap", "decision_time_seconds": "slot_decision_time"})
    values, reasons = slot_values(data.metadata, data.arrays, table, sample_index, names)
    mean = np.array([state["numeric"]["mean"][n] for n in names])
    scale = np.array([state["numeric"]["scale"][n] for n in names])
    insert_at = data.arrays["X_numeric"].shape[-1]
    arrays = dict(data.arrays)
    arrays["X_numeric"] = np.concatenate(
        [arrays["X_numeric"], np.where(reasons == PRESENT, (np.nan_to_num(values) - mean) / scale, 0.0).astype(np.float32)], axis=-1)
    codes = np.concatenate([arrays["mask_reason_codes"][..., :insert_at], reasons.astype(arrays["mask_reason_codes"].dtype),
                            arrays["mask_reason_codes"][..., insert_at:]], axis=-1)
    arrays["mask_reason_codes"] = codes
    arrays["temporal_unavailable_mask"] = codes == TEMPORAL
    arrays["feature_missing_mask"] = np.isin(codes, [2, 4, 5])
    return SplitData(arrays, data.metadata, data.labels, data.positions, data.targets), state


def check_against_published(data, directory, state, label):
    published = load_split(directory, "validation", state)
    for name in MODEL_ARRAYS:
        if not np.array_equal(data.arrays[name], published.arrays[name]):
            raise ValueError(f"Dry run: rebuilt {label} {name} differs from the published package")


def bundle_probabilities(bundle, data, require_same_source=True):
    """Predict with an exported bundle.

    The starting LSTM was exported before ``lstm_model.py`` gained optional
    extra layers; for it the source hash is not required, and functional
    equivalence is enforced instead by the exact 2024 parity check that
    always runs before 2025.
    """
    manifest = read_json(bundle / "manifest.json")
    for name, digest in manifest["files_sha256"].items():
        if sha256(bundle / name) != digest:
            raise ValueError(f"Bundle hash mismatch: {bundle.name}/{name}")
    if require_same_source and sha256(ROOT / "src/f1_pitstop/lstm_model.py") != manifest["model_source_sha256"]:
        raise ValueError("LSTM source differs from the bundle's frozen source")
    model = _build_model(read_json(bundle / "model_spec.json"))
    model.load_state_dict(torch.load(bundle / "weights.pt", map_location="cpu", weights_only=True))
    config = read_json(bundle / "config.json")
    probabilities, _ = predict(model, _loader(data, config["training"]["batch_size"]), torch.device("cpu"))
    return probabilities


def evaluate(split, start_bundle, sample_index):
    season = 2024 if split == "validation" else 2025
    _, state_v13 = verify_preprocessed(PACKAGE, PACKAGE_MANIFEST_SHA256)
    full, _ = split_inputs(split, state_v13)  # checks 2024 against the published v1_3 package
    with np.load(FEATURES_DIR / "windows.npz", allow_pickle=False) as archive:
        full.arrays["window_lap_numbers"] = np.array(archive["window_lap_numbers"][full.arrays["sample_id"]])
    telemetry, telemetry_state = telemetry_split(full, state_v13)
    strategy, strategy_state = strategy_split(telemetry, sample_index, season)
    if split == "validation":
        check_against_published(telemetry, TELEMETRY, telemetry_state, "telemetry")
        check_against_published(strategy, STRATEGY, strategy_state, "strategy")
    frozen = read_json(FROZEN_LSTM / "model.json")
    seeds = frozen["seeds"]
    by_seed = {s: bundle_probabilities(FROZEN_LSTM / f"seed{s}", strategy) for s in seeds}
    start = bundle_probabilities(start_bundle, full, require_same_source=False)
    if split == "validation":
        for seed, probabilities in by_seed.items():
            exported = pd.read_csv(FROZEN_LSTM / f"seed{seed}" / "validation_predictions.csv.gz").probability.to_numpy()
            if np.max(np.abs(exported - probabilities)) > PARITY_ATOL:
                raise ValueError(f"Dry run: strategy LSTM seed {seed} probabilities differ from its export")
        exported = pd.read_csv(start_bundle / "validation_predictions.csv.gz").probability.to_numpy()
        if np.max(np.abs(exported - start)) > PARITY_ATOL:
            raise ValueError("Dry run: starting LSTM probabilities differ from its export")
    samples = full.metadata.iloc[full.positions][KEYS].copy()
    samples["DriverNumber"] = samples.DriverNumber.astype(str).str.removesuffix(".0")
    samples = samples.merge(sample_index[["sample_id", "decision_time_seconds"]], on="sample_id",
                            how="left", validate="one_to_one").reset_index(drop=True)
    laps = pd.read_pickle(LAPS)
    laps = laps.loc[laps.Temporada.isin(range(2022, season + 1))].copy()
    stint_median = median_stint_length_train(laps)
    candidates = {
        "lstm_partida": (start, read_json(start_bundle / "threshold.json")["threshold"]),
        "lstm_estrategia": (np.mean([by_seed[s] for s in seeds], axis=0), frozen["threshold"]),
        **{f"lstm_estrategia_seed{s}": (by_seed[s], frozen["per_seed"][str(s)]["threshold"]) for s in seeds},
    }
    results, tables = {}, {}
    for name, (probability, threshold) in candidates.items():
        results[name], table, _, by_race = report(samples, full.targets, probability, threshold,
                                                  laps, stint_median, sample_index, season)
        tables[name] = (table, by_race)
        m = results[name]["metrics"]
        print(f"{season} {name:24s} F2 {m['f2']:.4f} AP {m['average_precision']:.4f} "
              f"P {m['precision']:.3f} R {m['recall']:.3f} eventos {results[name]['event_coverage']['covered_events']}",
              flush=True)
    return results, tables


def dry_run():
    """Rebuild 2024 only and check parity; never touches 2025."""
    sample_index = pd.read_csv(SAMPLE_INDEX, dtype={"DriverNumber": str})
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(START_ZIP) as archive:
            archive.extractall(tmp)
        evaluate("validation", Path(tmp) / START_CHILD, sample_index)
    print("Ensayo 2024 superado: entradas y probabilidades reproducidas.")


def run(output, confirmed):
    output = Path(output)
    if not confirmed:
        raise SystemExit("La evaluación de 2025 es única: añade --confirmar-evaluacion-unica para ejecutarla.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if sha256(FROZEN_LSTM / "manifest.json") != EXPECTED_FROZEN_LSTM_MANIFEST:
        raise ValueError("Frozen strategy LSTM manifest differs from the committed one")
    if sha256(START_ZIP) != START_ZIP_SHA256:
        raise ValueError("Starting LSTM ZIP changed")
    sample_index = pd.read_csv(SAMPLE_INDEX, dtype={"DriverNumber": str})
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(START_ZIP) as archive:
            archive.extractall(tmp)
        start_bundle = Path(tmp) / START_CHILD
        validation, _ = evaluate("validation", start_bundle, sample_index)
        print("Ensayo 2024 superado: entradas y probabilidades reproducidas.", flush=True)
        test, tables = evaluate("test", start_bundle, sample_index)
    output.mkdir(parents=True)
    for name, (table, by_race) in tables.items():
        if "seed" in name:
            continue
        table.to_csv(output / f"{name}_predicciones_2025.csv.gz", index=False, compression="gzip")
        by_race.to_csv(output / f"{name}_por_carrera_2025.csv", index=False)
    write_json(output / "metrics.json", {
        "declaration": "post-hoc comparison after the LightGBM 2025 test; no fitting or thresholding on 2025",
        "validation_2024": validation, "test_2025": test,
        "frozen_lstm_manifest_sha256": EXPECTED_FROZEN_LSTM_MANIFEST, "start_zip_sha256": START_ZIP_SHA256,
    })
    write_json(output / "manifest.json", {
        "artifact_version": "lstm_posthoc_2025_v1", "script": sha256(Path(__file__)),
        "files_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    })
    print(f"Resultados: {output}")


EXPECTED_FROZEN_LSTM_MANIFEST = "e5921051d48add1565fa62fd2c1912d925d05086bcea382d9c531b46642dd643"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/aclaraciones/evaluacion_lstm_2025")
    parser.add_argument("--confirmar-evaluacion-unica", action="store_true")
    parser.add_argument("--solo-ensayo-2024", action="store_true", help="Check 2024 parity only; 2025 is not loaded")
    args = parser.parse_args()
    if args.solo_ensayo_2024:
        dry_run()
    else:
        run(args.output, args.confirmar_evaluacion_unica)


if __name__ == "__main__":
    main()

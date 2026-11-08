"""Inspect Compound markers without changing any source or writing a matrix.

Run from any directory with Python and pandas. The output directory must be new.
Only local project pickles are read; no FastF1 API requests are made.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd


KEY = ["Temporada", "Ronda", "DriverNumber", "LapNumber"]
MARKERS = {"None", "nan"}
PHYSICAL = {"HARD", "MEDIUM", "SOFT", "INTERMEDIATE", "WET"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    data = root / "EDA/dataset/datos_carreras_completos_2022_2025"
    source = data / "vueltas_2022_2025.pkl"
    paths = [source, data / "vueltas_2022_2025_compound_reconstruido.pkl",
             data / "vueltas_2022_2025_limpieza_v1.pkl"]
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    hashes = {}

    def read_frame(path, cache=False):
        hashes[str(path.relative_to(root))] = sha256(path)
        if cache:
            with path.open("rb") as stream:
                return pickle.load(stream)["data"]
        return pd.read_pickle(path)

    frame, reconstructed, legacy = [read_frame(path) for path in paths]
    for other in (reconstructed, legacy):
        if not frame[KEY].equals(other[KEY]):
            raise ValueError("Matrices have different keys/order; comparison is unsafe")

    affected = frame.loc[frame["Compound"].isin(MARKERS)].copy()
    records = []
    blocks = []
    for (year, rnd, gp), group in affected.groupby(["Temporada", "Ronda", "GranPremio"]):
        race = read_frame(data / f"{year}_{rnd:02d}.pkl")
        master_race = frame.loc[(frame.Temporada == year) & (frame.Ronda == rnd)]
        if not master_race[KEY + ["Compound"]].reset_index(drop=True).equals(
            race[KEY + ["Compound"]].reset_index(drop=True)
        ):
            raise ValueError(f"Race Compound differs from master: {year}/{rnd}")
        cache_paths = list((root / f"EDA/dataset/cache_fastf1_carreras/{year}").glob(
            f"*{gp.replace(' ', '_')}*/*_Race/timing_app_data.ff1pkl"))
        if len(cache_paths) != 1:
            raise ValueError(f"Expected one cache for {year}/{rnd}, got {cache_paths}")
        timing = read_frame(cache_paths[0], cache=True)
        first_update = timing.groupby("Driver")["Time"].min()
        for _, row in group.iterrows():
            first = first_update.get(str(row.DriverNumber), pd.NaT)
            records.append({
                **{key: row[key] for key in KEY}, "GranPremio": gp,
                "Compound_raw": row.Compound, "Stint": row.Stint,
                "TyreLife": row.TyreLife, "FreshTyre": row.FreshTyre,
                "FastF1Generated": row.FastF1Generated,
                "LapStartTime": str(row.LapStartTime), "Time": str(row.Time),
                "first_driver_timing_update": str(first),
                "ends_before_first_timing_update": bool(pd.notna(first) and row.Time < first),
                "Compound_reconstructed": reconstructed.at[row.name, "Compound"],
                "Compound_legacy": legacy.at[row.name, "Compound"],
            })
        for (driver, stint), block in group.groupby(["DriverNumber", "Stint"], dropna=False):
            src = timing.loc[(timing.Driver.astype(str) == str(driver))
                             & (timing.Stint == stint - 1)]
            labels = src.loc[src.Compound.isin(PHYSICAL)]
            blocks.append({
                "Temporada": int(year), "Ronda": int(rnd), "GranPremio": gp,
                "DriverNumber": str(driver), "Stint": None if pd.isna(stint) else float(stint),
                "rows": len(block), "generated_rows": int(block.FastF1Generated.sum()),
                "raw_values": sorted(block.Compound.unique().tolist()),
                "source_stint_mapping": "raw source Stint = matrix Stint - 1; diagnostic only",
                "source_physical_compounds": sorted(labels.Compound.unique().tolist()),
                "source_physical_updates": [
                    {"Time": str(r.Time), "Compound": r.Compound}
                    for _, r in labels.iterrows()],
            })

    # Proposed representation only: a copy in memory, never a published matrix.
    normalized = frame.copy()
    normalized["Compound"] = normalized.Compound.astype("string").mask(
        normalized.Compound.isin(MARKERS), pd.NA)
    others = [c for c in frame if c != "Compound"]
    if not frame[others].equals(normalized[others]):
        raise ValueError("Unexpected change outside Compound")
    if not frame.loc[~frame.Compound.isin(MARKERS), "Compound"].astype("string").equals(
        normalized.loc[~frame.Compound.isin(MARKERS), "Compound"]
    ):
        raise ValueError("Unexpected change to nonmarker Compound")

    details = pd.DataFrame(records)
    summary = {
        "scope": "representation proposal; no matrix written or approval inferred",
        "rows": len(frame), "markers": affected.Compound.value_counts().to_dict(),
        "races_affected": len(affected[["Temporada", "Ronda"]].drop_duplicates()),
        "by_season": {str(k): int(v) for k, v in affected.groupby("Temporada").size().items()},
        "by_race": affected.groupby(["Temporada", "Ronda", "GranPremio", "Compound"])
            .size().rename("rows").reset_index().to_dict("records"),
        "identified_stints_affected": len(affected.loc[affected.Stint.notna(),
            ["Temporada", "Ronda", "DriverNumber", "Stint"]].drop_duplicates()),
        "driver_race_blocks_without_stint": len(affected.loc[affected.Stint.isna(),
            ["Temporada", "Ronda", "DriverNumber"]].drop_duplicates()),
        "reconstructed_marker_rows": int((frame.Compound.isin(MARKERS)
            & reconstructed.Compound.isin(PHYSICAL)).sum()),
        "reconstruction_changes_nonmarkers": int(((frame.Compound != reconstructed.Compound)
            & ~frame.Compound.isin(MARKERS)).sum()),
        "remaining_reconstructed_markers": reconstructed.loc[
            reconstructed.Compound.isin(MARKERS), "Compound"].value_counts().to_dict(),
        "legacy_null_original_values": frame.loc[legacy.Compound.isna(),
            "Compound"].value_counts().to_dict(),
        "proposed_normalization": {"source": "master", "new_nulls": int(normalized.Compound.isna().sum()),
            "rows_preserved": frame[KEY].equals(normalized[KEY]),
            "other_columns_unchanged": frame[others].equals(normalized[others]),
            "idempotent": normalized.Compound.equals(normalized.Compound.mask(
                normalized.Compound.isin(MARKERS), pd.NA))},
        "miami_nan_rows_before_first_source_update": int(details.loc[
            details.Compound_raw == "nan", "ends_before_first_timing_update"].sum()),
        "pandas_version": pd.__version__,
        "cache_parser_version_note": "Caches contain parser version 15, not acquisition package version",
    }
    for relative, expected in hashes.items():
        if sha256(root / relative) != expected:
            raise RuntimeError(f"Input changed during inspection: {relative}")
    summary["input_sha256"] = hashes
    summary["inputs_unchanged"] = True
    summary["analysis_script_sha256"] = sha256(Path(__file__))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    details.to_csv(args.output_dir / "marker_rows.csv", index=False)
    (args.output_dir / "stint_blocks.json").write_text(
        json.dumps(blocks, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "input_sha256"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Causal race-strategy features rebuilt from the lap matrix.

Every value for a decision point uses only lap rows whose end time and pit
entries whose time are not later than ``decision_time_seconds`` (start of the
predicted lap). The scheduled race distance comes from the FastF1 ``lap_count``
announced before the start, and typical stint lengths are fitted on train
seasons only.
"""
from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

SEASONS = (2022, 2023, 2024)  # 2025 final test stays reserved
TRAIN_SEASONS = (2022, 2023)
DRY = ("SOFT", "MEDIUM", "HARD")

__all__ = ["DRY", "SEASONS", "TRAIN_SEASONS", "median_stint_length_train", "scheduled_laps", "strategy_features"]


def _seconds(series):
    return pd.to_timedelta(series).dt.total_seconds().to_numpy(dtype=float)


def scheduled_laps(cache, seasons=SEASONS):
    """Scheduled race distance announced before the start (FastF1 lap_count, first value)."""
    result = {}
    for season in seasons:
        folders = sorted(p for p in (Path(cache) / str(season)).iterdir() if p.is_dir())
        for round_number, folder in enumerate(folders, start=1):
            files = list(folder.glob("*_Race/lap_count.ff1pkl"))
            if len(files) != 1:
                raise FileNotFoundError(f"lap_count missing for {folder}")
            with files[0].open("rb") as stream:
                data = pickle.load(stream)["data"]
            total = next(value for value in data["TotalLaps"] if value is not None)
            result[(season, round_number, folder.name)] = int(total)
    return result


def median_stint_length_train(laps):
    """Median completed dry-stint length per compound, from train seasons only."""
    train = laps.loc[laps.Temporada.isin(TRAIN_SEASONS) & laps.Compound.isin(DRY)]
    ended = train.groupby(["Temporada", "Ronda", "DriverNumber", "Stint"]).agg(
        compound=("Compound", "last"), length=("TyreLife", "max"),
        pitted=("PitInTime", lambda s: s.notna().any()))
    ended = ended.loc[ended.pitted]
    return ended.groupby("compound").length.median().to_dict()


def strategy_features(samples, laps, total_laps, stint_median):
    """Strategy/race-context features with a hard ``time <= decision_time`` filter."""
    rows = []
    for (season, rnd), group in samples.groupby(["Temporada", "Ronda"], sort=False):
        race = laps.loc[(laps.Temporada == season) & (laps.Ronda == rnd)].copy()
        race["DriverNumber"] = race.DriverNumber.astype(str).str.removesuffix(".0")
        race["LapNumber"] = race.LapNumber.astype(int)
        race["t_end"] = _seconds(race.Time)
        race["t_pit_in"] = _seconds(race.PitInTime)
        race["lap_time"] = _seconds(race.LapTime)
        race["in_out"] = race.PitInTime.notna() | race.PitOutTime.notna()
        race["green"] = race.TrackStatus.astype(str).eq("1")
        names = [key for key in total_laps if key[:2] == (season, rnd)]
        if len(names) != 1 or names[0][2].split("_", 1)[1] != race.GranPremio.iloc[0].replace(" ", "_"):
            raise ValueError(f"Cache folder does not match race {season}-{rnd}: {names}")
        scheduled = total_laps[names[0]]
        by_driver = {d: g.sort_values("LapNumber") for d, g in race.groupby("DriverNumber")}
        pit_times_all = race.loc[race.t_pit_in.notna(), ["DriverNumber", "t_pit_in"]]
        n_drivers = race.DriverNumber.nunique()
        lap_end = race.set_index(["DriverNumber", "LapNumber"])[["t_end", "Position"]]
        for sample in group.itertuples(index=False):
            driver, lap, t = str(sample.DriverNumber), int(sample.prediction_lap), float(sample.decision_time_seconds)
            own = by_driver.get(driver)
            seen = own.loc[own.t_end <= t] if own is not None else race.iloc[:0]
            own_pits = own.loc[own.t_pit_in <= t, ["LapNumber", "t_pit_in"]] if own is not None else race.iloc[:0]
            compounds = set(seen.Compound.dropna())
            row = {"sample_id": sample.sample_id,
                   "race_fraction": (lap - 1) / scheduled, "laps_remaining": scheduled - (lap - 1),
                   "stops_done": len(own_pits),
                   "laps_since_last_stop": lap - (own_pits.LapNumber.max() if len(own_pits) else 0),
                   "dry_compounds_used": len(compounds & set(DRY)),
                   "two_dry_compounds_used": float(len(compounds & set(DRY)) >= 2),
                   "used_wet_tyres": float(bool(compounds & {"INTERMEDIATE", "WET"}))}
            # Current-stint pace (green laps, no in/out laps).
            if len(seen):
                current = seen.loc[seen.Stint == seen.Stint.iloc[-1]]
                compound_now = current.Compound.dropna().iloc[-1] if current.Compound.notna().any() else None
                tyre_now = current.TyreLife.dropna().iloc[-1] if current.TyreLife.notna().any() else np.nan
                median = stint_median.get(compound_now, np.nan)
                row["tyre_life_vs_typical_stint"] = tyre_now / median if median else np.nan
                clean = current.loc[~current.in_out & current.green & np.isfinite(current.lap_time)]
                if len(clean) >= 2:
                    row["pace_vs_stint_best"] = clean.lap_time.iloc[-1] - clean.lap_time.min()
                    recent = clean.tail(5)
                    row["pace_slope_last5"] = (np.polyfit(recent.LapNumber, recent.lap_time, 1)[0]
                                               if len(recent) >= 3 else np.nan)
                last_lap_time = seen.lap_time.iloc[-1] if np.isfinite(seen.lap_time.iloc[-1]) else 100.0
            else:
                last_lap_time = 100.0
            window_start = t - 2 * max(last_lap_time, 60.0)
            others = pit_times_all.loc[pit_times_all.DriverNumber != driver]
            row["field_pit_entries_last2laps"] = int(((others.t_pit_in > window_start) & (others.t_pit_in <= t)).sum())
            stopped = pit_times_all.loc[pit_times_all.t_pit_in <= t].DriverNumber.nunique()
            row["field_share_stopped"] = stopped / n_drivers
            # Neighbours: ahead identified at lap N-1 (crossed before us), behind at lap N-2.
            for side, ref_lap, offset in (("ahead", lap - 1, -1), ("behind", lap - 2, +1)):
                own_ref = lap_end.loc[(driver, ref_lap)] if (driver, ref_lap) in lap_end.index else None
                value_gap, value_pitted = np.nan, np.nan
                if own_ref is not None and np.isfinite(own_ref.Position) and own_ref.t_end <= t:
                    same = race.loc[(race.LapNumber == ref_lap) & (race.Position == own_ref.Position + offset)
                                    & (race.t_end <= t)]
                    if len(same) == 1:
                        other = same.iloc[0]
                        value_gap = abs(own_ref.t_end - other.t_end)
                        other_pits = pit_times_all.loc[pit_times_all.DriverNumber == other.DriverNumber]
                        value_pitted = float(((other_pits.t_pit_in > window_start) & (other_pits.t_pit_in <= t)).any())
                row[f"gap_{side}_s"] = value_gap
                row[f"{side}_pitted_last2laps"] = value_pitted
            rows.append(row)
    return pd.DataFrame(rows)

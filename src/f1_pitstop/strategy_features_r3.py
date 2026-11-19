"""Third round of causal strategy features: how many stops are still expected.

Motivation (docs/aclaraciones/32_analisis_errores.md): almost half of the
false alarms happen when the driver stretches the stint or makes one stop
fewer than the model expects. These features describe the stop plan:

* ``circuit_typical_stops``: median stops per classified driver at the same
  Grand Prix in train races (other races only), else the train median.
* ``expected_stops_remaining``: that prior minus the driver's own stops so far.
* ``stops_vs_field_mean``: own stops minus the mean stops of the field so far.
* ``field_share_done_typical``: share of rivals that already made the typical
  number of stops.
* ``laps_beyond_long_stint``: laps remaining minus what the current tyre can
  still do if stretched to a long stint (90th percentile of train stint
  lengths for that compound). Positive means another stop is needed.

Current-race values only use laps that ended and pit entries made at or before
``decision_time_seconds``; priors come from train seasons only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy_features import DRY, TRAIN_SEASONS

__all__ = ["FEATURES", "stop_priors", "strategy_features_round3"]

FEATURES = ["circuit_typical_stops", "expected_stops_remaining", "stops_vs_field_mean",
            "field_share_done_typical", "laps_beyond_long_stint"]


def _seconds(series):
    return pd.to_timedelta(series).dt.total_seconds().to_numpy(dtype=float)


def stop_priors(laps):
    """Train-only priors: per-race stop counts of classified drivers and long-stint length per compound."""
    train = laps.loc[laps.Temporada.isin(TRAIN_SEASONS)].copy()
    train["DriverNumber"] = train.DriverNumber.astype(str).str.removesuffix(".0")
    race_stops = {}
    for (season, rnd), race in train.groupby(["Temporada", "Ronda"]):
        race_laps = race.LapNumber.max()
        per_driver = race.groupby("DriverNumber").agg(laps=("LapNumber", "max"),
                                                       stops=("PitInTime", lambda s: int(s.notna().sum())))
        classified = per_driver.loc[per_driver.laps >= 0.9 * race_laps]
        race_stops[(int(season), int(rnd))] = (race.GranPremio.iloc[0], classified.stops.tolist())
    stints = train.loc[train.Compound.isin(DRY)].groupby(
        ["Temporada", "Ronda", "DriverNumber", "Stint"]).agg(compound=("Compound", "last"), length=("TyreLife", "max"))
    long_stint = stints.groupby("compound").length.quantile(0.9).to_dict()
    return race_stops, long_stint


def _typical_stops(race_stops, season, rnd, gp):
    same, other = [], []
    for (s, r), (name, stops) in race_stops.items():
        if (s, r) == (season, rnd):
            continue
        (same if name == gp else other).extend(stops)
    if same:
        return float(np.median(same))
    return float(np.median(other)) if other else np.nan


def strategy_features_round3(samples, laps, total_laps, race_stops, long_stint):
    rows = []
    for (season, rnd), group in samples.groupby(["Temporada", "Ronda"], sort=False):
        race = laps.loc[(laps.Temporada == season) & (laps.Ronda == rnd)].copy()
        race["DriverNumber"] = race.DriverNumber.astype(str).str.removesuffix(".0")
        race["t_end"] = _seconds(race.Time)
        race["t_pit_in"] = _seconds(race.PitInTime)
        gp = race.GranPremio.iloc[0]
        scheduled = next(v for k, v in total_laps.items() if k[:2] == (season, rnd))
        typical = _typical_stops(race_stops, int(season), int(rnd), gp)
        pit_times = {d: np.sort(g.t_pit_in.dropna().to_numpy()) for d, g in race.groupby("DriverNumber")}
        first_end = race.groupby("DriverNumber").t_end.min().to_dict()
        by_driver = {d: g.sort_values("LapNumber") for d, g in race.groupby("DriverNumber")}
        for sample in group.itertuples(index=False):
            driver, lap, t = str(sample.DriverNumber), int(sample.prediction_lap), float(sample.decision_time_seconds)
            own_stops = int(np.searchsorted(pit_times.get(driver, np.array([])), t, side="right"))
            rivals = [d for d in pit_times if d != driver and first_end.get(d, np.inf) <= t]
            rival_stops = np.array([np.searchsorted(pit_times[d], t, side="right") for d in rivals], dtype=float)
            row = {"sample_id": sample.sample_id, "circuit_typical_stops": typical,
                   "expected_stops_remaining": max(typical - own_stops, 0.0) if np.isfinite(typical) else np.nan,
                   "stops_vs_field_mean": own_stops - rival_stops.mean() if len(rival_stops) else np.nan,
                   "field_share_done_typical": float((rival_stops >= typical).mean())
                   if len(rival_stops) and np.isfinite(typical) else np.nan}
            own = by_driver.get(driver)
            seen = own.loc[own.t_end <= t] if own is not None else race.iloc[:0]
            value = np.nan
            if len(seen):
                current = seen.loc[seen.Stint == seen.Stint.iloc[-1]]
                compound = current.Compound.dropna().iloc[-1] if current.Compound.notna().any() else None
                tyre = current.TyreLife.dropna().iloc[-1] if current.TyreLife.notna().any() else np.nan
                long = long_stint.get(compound, np.nan)
                value = (scheduled - (lap - 1)) - (long - tyre)
            row["laps_beyond_long_stint"] = value
            rows.append(row)
    return pd.DataFrame(rows)

"""Second round of causal race-strategy features (candidates, not yet in any input package).

Same causal rule as ``strategy_features``: for a decision at the start of lap N
(``decision_time_seconds`` = t), only lap rows that ended at or before t and pit
entries at or before t are used. Pit-loss priors come from train races only and
exclude the race being described, so a train race never sees its own stops.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy_features import DRY, TRAIN_SEASONS

__all__ = ["FEATURE_GROUPS", "pit_loss_table", "strategy_features_round2"]

FEATURE_GROUPS = {
    "llegar_al_final": ["laps_short_to_finish"],
    "companero": ["teammate_pitted_last2laps", "teammate_stops_minus_own"],
    "ventana_parada": ["pit_loss_est_s", "free_stop_margin_s", "cars_within_pit_loss_behind"],
    "undercut": ["nearest_behind_pitted_last2laps", "behind_faster_by_s"],
    "safety_car": ["sc_vsc_last_lap", "cheap_stop_index"],
    "lluvia": ["rain_on_slicks", "dry_on_wets"],
    "pelotón": ["share_field_fresher_tyres"],
}
WET = ("INTERMEDIATE", "WET")
SC_CODES = ("4", "6", "7")


def _seconds(series):
    return pd.to_timedelta(series).dt.total_seconds().to_numpy(dtype=float)


def _prepare_race(race):
    race = race.copy()
    race["DriverNumber"] = race.DriverNumber.astype(str).str.removesuffix(".0")
    race["LapNumber"] = race.LapNumber.astype(int)
    race["t_end"] = _seconds(race.Time)
    race["t_pit_in"] = _seconds(race.PitInTime)
    race["lap_time"] = _seconds(race.LapTime)
    race["in_out"] = race.PitInTime.notna() | race.PitOutTime.notna()
    race["green"] = race.TrackStatus.astype(str).eq("1")
    race["clean"] = ~race.in_out & race.green & np.isfinite(race.lap_time)
    return race.sort_values(["DriverNumber", "LapNumber"])


def _race_stop_losses(race):
    """Green-flag stop losses: in-lap + out-lap minus twice the driver's median clean lap so far.

    Returns (available_time, loss) where available_time is the end of the out-lap.
    """
    rows = []
    for _, laps in race.groupby("DriverNumber"):
        laps = laps.set_index("LapNumber")
        for lap in laps.index[laps.t_pit_in.notna()]:
            if lap + 1 not in laps.index:
                continue
            in_lap, out_lap = laps.loc[lap], laps.loc[lap + 1]
            if not (in_lap.green and out_lap.green and np.isfinite(in_lap.lap_time) and np.isfinite(out_lap.lap_time)):
                continue
            before = laps.loc[(laps.index < lap) & laps.clean, "lap_time"]
            if len(before) < 3:
                continue
            rows.append((float(out_lap.t_end), float(in_lap.lap_time + out_lap.lap_time - 2 * before.median())))
    return rows


def pit_loss_table(laps):
    """Per-race green-flag stop losses for every race in ``laps``: {(season, round): (gp, [(t, loss)])}."""
    table = {}
    for (season, rnd), race in laps.groupby(["Temporada", "Ronda"]):
        table[(int(season), int(rnd))] = (race.GranPremio.iloc[0], _race_stop_losses(_prepare_race(race)))
    return table


def _prior_loss(table, season, rnd, gp):
    """Median train loss at the same Grand Prix (other races), else all other train races."""
    same, other = [], []
    for (s, r), (name, losses) in table.items():
        if s not in TRAIN_SEASONS or (s, r) == (season, rnd):
            continue
        values = [loss for _, loss in losses]
        (same if name == gp else other).extend(values)
    if same:
        return float(np.median(same))
    return float(np.median(other)) if other else np.nan


def strategy_features_round2(samples, laps, total_laps, stint_median, loss_table):
    rows = []
    for (season, rnd), group in samples.groupby(["Temporada", "Ronda"], sort=False):
        race = _prepare_race(laps.loc[(laps.Temporada == season) & (laps.Ronda == rnd)])
        gp = race.GranPremio.iloc[0]
        scheduled = next(v for k, v in total_laps.items() if k[:2] == (season, rnd))
        prior = _prior_loss(loss_table, int(season), int(rnd), gp)
        race_losses = loss_table[(int(season), int(rnd))][1]
        by_driver = {d: g for d, g in race.groupby("DriverNumber")}
        team_of = race.groupby("DriverNumber").Team.first().to_dict()
        pits = race.loc[race.t_pit_in.notna(), ["DriverNumber", "LapNumber", "t_pit_in"]]
        lap_end = race.set_index(["DriverNumber", "LapNumber"])[["t_end", "Position"]]
        timelines = {}
        for other, other_laps in by_driver.items():
            ordered = other_laps.loc[np.isfinite(other_laps.t_end)].sort_values("t_end")
            other_pits = pits.loc[pits.DriverNumber == other].sort_values("t_pit_in")
            timelines[other] = (ordered.t_end.to_numpy(), other_pits.t_pit_in.to_numpy(),
                                other_pits.LapNumber.to_numpy())
        for sample in group.itertuples(index=False):
            driver, lap, t = str(sample.DriverNumber), int(sample.prediction_lap), float(sample.decision_time_seconds)
            own = by_driver.get(driver, race.iloc[:0])
            seen = own.loc[own.t_end <= t]
            own_pits = pits.loc[(pits.DriverNumber == driver) & (pits.t_pit_in <= t)]
            last_lap_time = seen.lap_time.iloc[-1] if len(seen) and np.isfinite(seen.lap_time.iloc[-1]) else 100.0
            window_start = t - 2 * max(last_lap_time, 60.0)
            row = {"sample_id": sample.sample_id}
            # Can the current tyre reach the flag?
            compound = tyre = np.nan
            if len(seen):
                current = seen.loc[seen.Stint == seen.Stint.iloc[-1]]
                compound = current.Compound.dropna().iloc[-1] if current.Compound.notna().any() else np.nan
                tyre = current.TyreLife.dropna().iloc[-1] if current.TyreLife.notna().any() else np.nan
            typical = stint_median.get(compound, np.nan)
            laps_remaining = scheduled - (lap - 1)
            row["laps_short_to_finish"] = laps_remaining - (typical - tyre)
            # Teammate.
            mates = [d for d, team in team_of.items() if d != driver and team == team_of.get(driver)]
            mate_pits = pits.loc[pits.DriverNumber.isin(mates) & (pits.t_pit_in <= t)]
            row["teammate_pitted_last2laps"] = float((mate_pits.t_pit_in > window_start).any()) if mates else np.nan
            row["teammate_stops_minus_own"] = (len(mate_pits) / len(mates) - len(own_pits)) if mates else np.nan
            # Pit window: cars that would pass during a stop.
            known = [loss for available, loss in race_losses if available <= t]
            pit_loss = float(np.median(known)) if len(known) >= 2 else prior
            row["pit_loss_est_s"] = pit_loss
            ref = lap_end.loc[(driver, lap - 2)] if (driver, lap - 2) in lap_end.index else None
            margin, within = np.nan, np.nan
            behind_pitted, behind_faster = np.nan, np.nan
            if ref is not None and ref.t_end <= t and np.isfinite(pit_loss):
                crossings = race.loc[(race.LapNumber == lap - 2) & (race.t_end > ref.t_end)
                                     & (race.t_end <= t) & (race.DriverNumber != driver)]
                gaps = crossings.t_end - ref.t_end
                within = float((gaps <= pit_loss).sum())
                if len(gaps):
                    margin = float(gaps.min() - pit_loss)
                    nearest = crossings.loc[gaps.idxmin()]
                    other = nearest.DriverNumber
                    other_pits = pits.loc[(pits.DriverNumber == other) & (pits.t_pit_in <= t)]
                    behind_pitted = float((other_pits.t_pit_in > window_start).any())
                    other_seen = by_driver[other]
                    other_clean = other_seen.loc[(other_seen.t_end <= t) & other_seen.clean]
                    own_clean = seen.loc[seen.clean]
                    if len(other_clean) and len(own_clean):
                        behind_faster = float(own_clean.lap_time.iloc[-1] - other_clean.lap_time.iloc[-1])
            row["free_stop_margin_s"] = margin
            row["cars_within_pit_loss_behind"] = within
            row["nearest_behind_pitted_last2laps"] = behind_pitted
            row["behind_faster_by_s"] = behind_faster
            # Safety car on the last completed lap.
            sc = float(any(code in str(seen.TrackStatus.iloc[-1]) for code in SC_CODES)) if len(seen) else np.nan
            row["sc_vsc_last_lap"] = sc
            row["cheap_stop_index"] = sc * (tyre / typical) if np.isfinite(typical) and typical else np.nan
            # Rain versus tyre type.
            if len(seen):
                recent = seen.tail(2)
                raining = bool(recent.Weather_Rainfall.iloc[-1]) if pd.notna(recent.Weather_Rainfall.iloc[-1]) else False
                row["rain_on_slicks"] = float(raining and compound in DRY)
                row["dry_on_wets"] = float((~recent.Weather_Rainfall.fillna(False).astype(bool)).all() and compound in WET)
            else:
                row["rain_on_slicks"] = row["dry_on_wets"] = np.nan
            # Field tyre freshness.
            # A rival is on fresher tyres if its latest stop happened on a later lap than ours.
            own_last_stop = own_pits.LapNumber.max() if len(own_pits) else 0
            fresher, counted = 0, 0
            for other, (ends, pit_times, pit_laps) in timelines.items():
                if other == driver or not np.searchsorted(ends, t, side="right"):
                    continue
                pit_count = np.searchsorted(pit_times, t, side="right")
                counted += 1
                fresher += (pit_laps[pit_count - 1] if pit_count else 0) > own_last_stop
            row["share_field_fresher_tyres"] = fresher / counted if counted else np.nan
            rows.append(row)
    return pd.DataFrame(rows)

"""Boundary tests for paired pit-visit labels under SPEC-LABEL-002."""
import unittest

import pandas as pd

from scripts.etiquetar_paradas import (
    assert_legacy_invariance, extract_events, label_horizons, review_source_type,
)


def race_frame(pit_in=None, pit_out=None, laps=range(1, 7)):
    rows = []
    for lap in laps:
        rows.append({
            "Temporada": 2022, "Ronda": 1, "GranPremio": "Test GP",
            "DriverNumber": "1", "LapNumber": lap,
            "PitInTime": pit_in[1] if pit_in and lap == pit_in[0] else pd.NaT,
            "PitOutTime": pit_out[1] if pit_out and lap == pit_out[0] else pd.NaT,
            "FastF1Generated": False,
            "LapStartTime": pd.Timedelta((lap - 1) * 60, unit="s"),
            "Time": pd.Timedelta(lap * 60, unit="s"),
            "Stint": 1,
        })
    frame = pd.DataFrame(rows)
    frame["PitInTime"] = pd.Series(
        [row["PitInTime"] for row in rows], dtype="timedelta64[ns]"
    )
    frame["PitOutTime"] = pd.Series(
        [row["PitOutTime"] for row in rows], dtype="timedelta64[ns]"
    )
    frame["LapStartTime"] = pd.to_timedelta(frame["LapStartTime"])
    frame["Time"] = pd.to_timedelta(frame["Time"])
    return frame


class LabelContractTests(unittest.TestCase):
    def test_review_source_type_distinguishes_fia_from_formula1(self):
        self.assertEqual(review_source_type("https://www.fia.com/report"), "FIA")
        self.assertEqual(review_source_type("https://www.formula1.com/report"), "Formula1")

    def test_official_confirmation_does_not_replace_missing_pit_out(self):
        frame = race_frame(pit_in=(3, pd.Timedelta(130, unit="s")))
        events, _ = extract_events(frame, {"2022-01-1-IN-3": "pit_stop_confirmed_by_FIA"})
        event = events.loc[events["event_lap"] == 3].iloc[0]
        self.assertEqual(event["event_status"], "unknown")
        self.assertEqual(event["reason"], "pit_in_without_valid_pit_out")

    def test_orphan_pit_out_is_not_a_positive_event(self):
        frame = race_frame(pit_out=(3, pd.Timedelta(130, unit="s")))
        events, _ = extract_events(frame, {})
        event = events.loc[events["reason"] == "orphan_pit_out_has_no_entry_lap"].iloc[0]
        self.assertEqual(event["event_status"], "unknown")
        self.assertEqual(event["reason"], "orphan_pit_out_has_no_entry_lap")
        self.assertTrue(pd.isna(event["event_lap"]))

    def test_valid_pair_after_decision_is_positive(self):
        frame = race_frame(
            pit_in=(3, pd.Timedelta(130, unit="s")),
            pit_out=(3, pd.Timedelta(150, unit="s")),
        )
        events, _ = extract_events(frame, {})
        labels = label_horizons(frame, events)
        decision = labels.loc[labels["LapNumber"] == 1].iloc[0]
        self.assertTrue(bool(decision["PitWithin3Laps"]))

    def test_entry_before_current_decision_is_not_future_positive(self):
        frame = race_frame(
            pit_in=(1, pd.Timedelta(20, unit="s")),
            pit_out=(1, pd.Timedelta(50, unit="s")),
        )
        frame.loc[frame["LapNumber"] == 1, "LapStartTime"] = pd.Timedelta(30, unit="s")
        events, _ = extract_events(frame, {})
        labels = label_horizons(frame, events)
        decision = labels.loc[labels["LapNumber"] == 1].iloc[0]
        self.assertFalse(bool(decision["PitWithin3Laps"]))

    def test_later_lap_number_does_not_override_event_time_before_decision(self):
        frame = race_frame(
            pit_in=(3, pd.Timedelta(20, unit="s")),
            pit_out=(3, pd.Timedelta(50, unit="s")),
        )
        frame.loc[frame["LapNumber"] == 1, "LapStartTime"] = pd.Timedelta(30, unit="s")
        events, _ = extract_events(frame, {})
        labels = label_horizons(frame, events)
        decision = labels.loc[labels["LapNumber"] == 1].iloc[0]
        self.assertFalse(bool(decision["PitWithin3Laps"]))

    def test_event_at_n_plus_2_is_in_horizon_but_n_plus_3_is_not(self):
        inside = race_frame(
            pit_in=(3, pd.Timedelta(130, unit="s")),
            pit_out=(3, pd.Timedelta(150, unit="s")),
        )
        inside_events, _ = extract_events(inside, {})
        inside_labels = label_horizons(inside, inside_events)
        self.assertTrue(bool(inside_labels.loc[inside_labels.LapNumber.eq(1), "PitWithin3Laps"].iloc[0]))
        outside = race_frame(
            pit_in=(4, pd.Timedelta(190, unit="s")),
            pit_out=(4, pd.Timedelta(210, unit="s")),
        )
        outside_events, _ = extract_events(outside, {})
        outside_labels = label_horizons(outside, outside_events)
        self.assertFalse(bool(outside_labels.loc[outside_labels.LapNumber.eq(1), "PitWithin3Laps"].iloc[0]))

    def test_tail_horizon_without_all_three_laps_stays_unknown(self):
        frame = race_frame(laps=range(1, 4))
        events, _ = extract_events(frame, {})
        labels = label_horizons(frame, events)
        last = labels.loc[labels.LapNumber.eq(2)].iloc[0]
        self.assertTrue(pd.isna(last.PitWithin3Laps))
        self.assertEqual(last.LabelReason, "incomplete_horizon")

    def test_legacy_comparison_requires_exact_labels_and_reasons(self):
        base = pd.DataFrame([{
            "Temporada": 2022, "Ronda": 1, "DriverNumber": "1", "LapNumber": 2,
            "PitWithin3Laps": pd.NA, "LabelReason": "unresolved_pit_event",
            "LabelBasis": "unresolved", "RuleVersion": "pit_entry_v2",
            "EligibleForSupervisedLearning": False,
        }])
        base["PitWithin3Laps"] = pd.array(base.PitWithin3Laps, dtype="boolean")
        self.assertEqual(1, assert_legacy_invariance(base, base.copy())["rows"])
        changed = base.copy()
        changed.loc[0, "LabelReason"] = "incomplete_horizon"
        with self.assertRaisesRegex(ValueError, "labels differ"):
            assert_legacy_invariance(changed, base)


if __name__ == "__main__":
    unittest.main()

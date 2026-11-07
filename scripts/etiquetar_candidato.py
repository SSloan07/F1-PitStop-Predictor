"""Inventario y etiquetas candidatas para SPEC-LABEL-001."""
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "EDA/dataset/datos_carreras_completos_2022_2025"
SOURCE = DATA / "vueltas_2022_2025_limpieza_v1.pkl"
OUTPUT = DATA / "etiquetado_candidate_v1"
SEASONS = [2022, 2023]
KEY = ["Temporada", "Ronda", "DriverNumber", "LapNumber"]
GROUP = ["Temporada", "Ronda", "DriverNumber"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_or_none(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return None if value.upper() in {"", "NONE", "NAN", "UNKNOWN", "<NA>"} else value


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_pickle(SOURCE)
    frame = frame.loc[frame["Temporada"].isin(SEASONS)].copy()
    required = KEY + [
        "GranPremio", "PitInTime", "PitOutTime", "LapStartTime", "Time",
        "Stint", "Compound", "TyreLife", "FreshTyre", "FastF1Generated",
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError("Null or duplicated sample keys")
    frame["DriverNumber"] = frame["DriverNumber"].astype(str)
    frame = frame.sort_values(GROUP + ["LapNumber"], kind="stable").reset_index(drop=True)

    events = []
    orphan_exits = []
    for group_key, group in frame.groupby(GROUP, sort=True, dropna=False):
        entries = group.loc[group["PitInTime"].notna()].sort_values("PitInTime")
        exits = group.loc[group["PitOutTime"].notna()].sort_values("PitOutTime")
        used_exit = set()
        for _, entry in entries.iterrows():
            later = exits.loc[
                (exits["PitOutTime"] > entry["PitInTime"])
                & (~exits.index.isin(used_exit))
            ]
            next_entry = entries.loc[
                entries["PitInTime"] > entry["PitInTime"], "PitInTime"
            ].min()
            if pd.notna(next_entry):
                later = later.loc[later["PitOutTime"] < next_entry]
            exit_row = later.iloc[0] if len(later) else None
            if exit_row is not None:
                used_exit.add(exit_row.name)
            lap = int(entry["LapNumber"])
            preceding = group.loc[group["LapNumber"] == lap - 1]
            following = group.loc[group["LapNumber"] == lap + 1]
            before = preceding.iloc[0] if len(preceding) == 1 else None
            after = following.iloc[0] if len(following) == 1 else None
            stint_increase = bool(
                before is not None and after is not None
                and pd.notna(before["Stint"]) and pd.notna(after["Stint"])
                and after["Stint"] > before["Stint"]
            )
            event_id = f"{group_key[0]}-{int(group_key[1]):02d}-{group_key[2]}-IN-{lap}"
            events.append({
                "event_id": event_id, "Temporada": int(group_key[0]),
                "Ronda": int(group_key[1]), "GranPremio": str(entry["GranPremio"]),
                "DriverNumber": group_key[2], "entry_lap": lap,
                "entry_time": str(entry["PitInTime"]),
                "exit_lap": None if exit_row is None else int(exit_row["LapNumber"]),
                "exit_time": None if exit_row is None else str(exit_row["PitOutTime"]),
                "stint_before": None if before is None else before["Stint"],
                "stint_after": None if after is None else after["Stint"],
                "compound_before": None if before is None else text_or_none(before["Compound"]),
                "compound_after": None if after is None else text_or_none(after["Compound"]),
                "tyrelife_before": None if before is None else before["TyreLife"],
                "tyrelife_after": None if after is None else after["TyreLife"],
                "fresh_after": None if after is None else after["FreshTyre"],
                "immediate_stint_increase": stint_increase,
                "event_status": "unknown", "evidence_basis": "unresolved",
                "review_status": "not_reviewed", "reason": "external_review_required",
            })
        for _, exit_row in exits.iterrows():
            if exit_row.name not in used_exit:
                orphan_exits.append({
                    "Temporada": int(group_key[0]), "Ronda": int(group_key[1]),
                    "DriverNumber": group_key[2], "lap": exit_row["LapNumber"],
                    "time": str(exit_row["PitOutTime"]), "reason": "unpaired_exit",
                })

    event_frame = pd.DataFrame(events)
    event_frame.to_csv(OUTPUT / "events.csv", index=False)
    pd.DataFrame(orphan_exits).to_csv(OUTPUT / "orphan_exits.csv", index=False)

    rng = random.Random(42)
    random_events = rng.sample(events, min(24, len(events)))
    selected = {event["event_id"]: {"event_id": event["event_id"], "strata": ["random"]}
                for event in random_events}
    strata = {
        "same_compound": lambda e: e["compound_before"] is not None and e["compound_before"] == e["compound_after"],
        "different_compound": lambda e: e["compound_before"] is not None and e["compound_after"] is not None and e["compound_before"] != e["compound_after"],
        "used_tyre": lambda e: e["fresh_after"] is False,
        "no_immediate_stint_increase": lambda e: not e["immediate_stint_increase"],
        "incomplete_stint_evidence": lambda e: e["stint_before"] is None or e["stint_after"] is None,
    }
    stratum_counts = {}
    for name, predicate in strata.items():
        candidates = [e for e in events if predicate(e) and e["event_id"] not in selected]
        chosen = candidates[:2]
        stratum_counts[name] = len(chosen)
        for event in chosen:
            selected[event["event_id"]] = {"event_id": event["event_id"], "strata": [name]}
    review = {
        "seed": 42, "seasons": SEASONS, "random_target": 24,
        "random_event_ids": sorted(e["event_id"] for e in random_events),
        "stratum_counts": stratum_counts,
        "cases": [selected[key] for key in sorted(selected)],
        "sources_per_case_max": 3, "manual_review_required": True,
    }
    (OUTPUT / "review_cases.json").write_text(json.dumps(review, indent=2) + "\n")

    # Decisions require a documented human review; absent review never implies False.
    decisions_path = OUTPUT / "review_decisions.json"
    decisions = json.loads(decisions_path.read_text()) if decisions_path.exists() else []
    by_id = {item["event_id"]: item for item in decisions}
    for idx, event in event_frame.iterrows():
        decision = by_id.get(event["event_id"])
        if decision:
            for field in ("event_status", "evidence_basis", "review_status", "reason"):
                if field in decision:
                    event_frame.at[idx, field] = decision[field]
    event_frame.to_csv(OUTPUT / "events.csv", index=False)

    changes = event_frame.loc[event_frame["event_status"] == "change"]
    review_changes = sum(d.get("review_status") == "verified_change" for d in decisions)
    review_no_change = sum(d.get("review_status") == "verified_no_change" for d in decisions)
    r1_contradictions = sum(
        d.get("review_status") == "verified_no_change"
        and bool(event_frame.loc[event_frame["event_id"] == d["event_id"], "immediate_stint_increase"].any())
        for d in decisions
    )
    r1_enabled = review_changes >= 10 and review_no_change >= 1 and r1_contradictions == 0
    if r1_enabled:
        pattern_ids = set(event_frame.loc[event_frame["immediate_stint_increase"], "event_id"])
        event_frame.loc[event_frame["event_id"].isin(pattern_ids), ["event_status", "evidence_basis", "reason"]] = [
            "change", "candidate_rule", "entry_exit_immediate_stint_increase"
        ]

    labels = []
    for group_key, group in frame.groupby(GROUP, sort=False, dropna=False):
        indexed = group.set_index("LapNumber", drop=False)
        driver_events = event_frame.loc[
            (event_frame["Temporada"] == group_key[0])
            & (event_frame["Ronda"] == group_key[1])
            & (event_frame["DriverNumber"] == group_key[2])
        ]
        for _, sample in group.iterrows():
            current_ok = (
                not pd.isna(sample["FastF1Generated"])
                and not bool(sample["FastF1Generated"])
                and pd.notna(sample["LapStartTime"])
            )
            target, reason, basis, eligible = pd.NA, "invalid_current_sample", "unresolved", False
            lap = int(sample["LapNumber"])
            if current_ok:
                horizon = {lap, lap + 1, lap + 2}
                relevant = driver_events.loc[driver_events["entry_lap"].isin(horizon)]
                positive = relevant.loc[relevant["event_status"] == "change"]
                if len(positive):
                    target, reason, basis, eligible = True, "change_in_horizon", positive.iloc[0]["evidence_basis"], True
                else:
                    complete = True
                    for future_lap in sorted(horizon):
                        if future_lap not in indexed.index:
                            complete = False
                            break
                        row = indexed.loc[future_lap]
                        if isinstance(row, pd.DataFrame) or bool(row["FastF1Generated"]) or pd.isna(row["LapStartTime"]) or pd.isna(row["Time"]) or row["Time"] < row["LapStartTime"]:
                            complete = False
                            break
                    unresolved = (relevant["event_status"] == "unknown").any()
                    if complete and not unresolved:
                        target, reason, basis, eligible = False, "complete_horizon_no_change", "source_coverage", True
                    else:
                        reason = "ambiguous_event" if unresolved else "insufficient_coverage"
            labels.append({
                "Temporada": int(group_key[0]), "Ronda": int(group_key[1]),
                "DriverNumber": group_key[2], "LapNumber": lap,
                "PitWithin3Laps": target, "LabelReason": reason,
                "LabelBasis": basis, "RuleVersion": "candidate_v1",
                "EligibleForSupervisedLearning": eligible,
            })
    label_frame = pd.DataFrame(labels)
    label_frame["PitWithin3Laps"] = pd.array(label_frame["PitWithin3Laps"], dtype="boolean")
    label_frame.to_pickle(OUTPUT / "labels.pkl")
    coverage = label_frame.groupby(["Temporada", "Ronda"], as_index=False).agg(
        samples=("LapNumber", "size"),
        known=("PitWithin3Laps", "count"),
        positives=("PitWithin3Laps", lambda s: int((s == True).sum())),
        negatives=("PitWithin3Laps", lambda s: int((s == False).sum())),
        unknown=("PitWithin3Laps", lambda s: int(s.isna().sum())),
    )
    coverage.to_csv(OUTPUT / "coverage.csv", index=False)
    manifest = {
        "rule_version": "candidate_v1", "seasons": SEASONS,
        "horizon_laps": 3, "prediction_instant": "lap_start",
        "event_lap": "pit_entry", "seed": 42,
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256": sha256(SOURCE),
        "rule_status": "candidate_enabled" if r1_enabled else "requires_review",
        "R1_enabled": r1_enabled, "events": len(event_frame),
        "review_cases": len(selected), "external_decisions": len(decisions),
        "labels": len(label_frame), "unknown_labels": int(label_frame["PitWithin3Laps"].isna().sum()),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = (
        "# Etiquetado candidato v1\n\n"
        f"Estado de R1: **{manifest['rule_status']}**.\n\n"
        f"- Eventos de entrada: {len(event_frame)}; salidas huérfanas: {len(orphan_exits)}.\n"
        f"- Casos seleccionados: {len(selected)}; decisiones externas registradas: {len(decisions)}.\n"
        f"- Etiquetas positivas: {int((label_frame['PitWithin3Laps'] == True).sum())}; "
        f"negativas: {int((label_frame['PitWithin3Laps'] == False).sum())}; "
        f"desconocidas: {manifest['unknown_labels']}.\n"
        "- Las temporadas usadas para seleccionar casos son 2022–2023.\n"
        "- Las negativas representan cobertura de las fuentes observadas, no verificación física independiente.\n"
    )
    (OUTPUT / "report.md").write_text(report)
    print(report)
    print(f"Artefactos: {OUTPUT}")


if __name__ == "__main__":
    main()

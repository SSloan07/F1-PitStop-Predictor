"""End-to-end synthetic smoke tests; these are not F1 experiment results."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from f1_pitstop.rnn_data import load_split, read_json, sha256, verify_preprocessed, write_json
from f1_pitstop.rnn_experiment import attach_decision_times, event_coverage, evaluate_bundle, train_experiment

STAGING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGING / "scripts"))
from empaquetar_experimento_rnn import CODE, package_experiment


def fixture(root):
    data = root / "data"
    data.mkdir()
    vocab = lambda values: [{"id": i + 4, "category": v} for i, v in enumerate(values)]
    state = {"fit_split": "train", "schema": {"window_length": 2,
        "numeric_feature_names": ["LapNumber", "signal"],
        "categorical_feature_names": ["Compound_asof", "track_phase_asof"],
        "static_feature_names": ["Driver", "Team", "GranPremio"]},
        "vocabularies": {"dynamic": {"Compound_asof": vocab(["SOFT"]),
            "track_phase_asof": vocab(["GREEN"])}, "static": {
            "Driver": vocab(["VER"]), "Team": vocab(["Team"]), "GranPremio": vocab(["Race"])}}}
    write_json(data / "preprocessor.json", state)
    write_json(data / "config.json", {"synthetic_fixture": True})
    for split, year, offset in [("train", 2022, 0), ("validation", 2024, 8)]:
        n, w = 8, 2
        rng = np.random.default_rng(3)
        numeric = rng.normal(size=(n, w, 2)).astype(np.float32)
        ids = np.full((n, w, 2), 4, dtype=np.int32)
        static = np.full((n, 3), 4, dtype=np.int32)
        reason = np.zeros((n, w, 4), dtype=np.uint8)
        padding = np.zeros((n, w), bool)
        padding[0, 0] = True
        reason[0, 0, :] = 1
        numeric[0, 0, :] = 0
        reason[:, -1, 1] = 3
        numeric[:, -1, 1] = 0
        sample_ids = np.arange(offset, offset+n)
        np.savez_compressed(data / f"{split}_inputs.npz", X_numeric=numeric,
            X_dynamic_categorical_ids=ids, X_static_categorical_ids=static,
            mask_reason_codes=reason, padding_mask=padding, slot_exists_mask=~padding,
            slot_row_missing_mask=np.zeros((n,w),bool), temporal_unavailable_mask=reason==3,
            feature_missing_mask=np.isin(reason,[2,4,5]), sample_id=sample_ids)
        eligible = [True] * n
        if split == "train":
            eligible[-1] = False
        meta = pd.DataFrame({"sample_id": sample_ids, "Temporada": year, "Ronda": 1,
            "DriverNumber": "1", "prediction_lap": np.arange(1,9),
            "decision_time_seconds": np.arange(1,9)*10, "split": split,
            "supervised_eligible": eligible})
        targets = [False,True,True,True,False,True,True,True]
        if split == "train":
            targets[-1] = ""
        labels = meta.copy()
        labels["PitWithin3Laps"] = targets
        if split == "validation":
            meta.to_csv(root/"sample_index.csv.gz",index=False,compression="gzip")
        meta.drop(columns="decision_time_seconds").to_csv(data/f"{split}_metadata.csv.gz",index=False,compression="gzip")
        labels.drop(columns="decision_time_seconds").to_csv(data/f"{split}_labels.csv.gz",index=False,compression="gzip")
    digest = hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()
    write_json(data/"manifest.json", {"artifact_version":"preprocessed_causal_inputs_v1",
        "fit_state_sha256":digest,"source_hashes":{"sample_index":sha256(root/"sample_index.csv.gz")},
        "output_files_sha256":{p.name:sha256(p) for p in data.iterdir()}})
    events = root / "events.csv"
    pd.DataFrame([{"event_id":f"e{lap}","Temporada":2024,"Ronda":1,"DriverNumber":"1",
                  "entry_lap":lap,"entry_time":str(pd.Timedelta(seconds=lap*10+5)),
                  "event_status":"pit_stop"} for lap in [4,8]]).to_csv(events,index=False)
    config = read_json(STAGING/"configs/experimento_rnn_v1.json")
    config.update({"preprocessed_package":"data","preprocessed_manifest_sha256":sha256(data/"manifest.json"),
                   "run_purpose":"synthetic_smoke"})
    config["model"]["window_length"] = 2
    config["training"].update({"max_epochs":2,"patience":1,"batch_size":4,"device":"cpu"})
    config["evaluation"].update({"events_source":"events.csv","events_sha256":sha256(events),
        "decision_index_source":"sample_index.csv.gz","decision_index_sha256":sha256(root/"sample_index.csv.gz")})
    config_path = root/"configs/experimento_rnn_v1.json"
    config_path.parent.mkdir()
    write_json(config_path,config)
    return data, config_path, state


class ExperimentIntegrationTests(unittest.TestCase):
    def test_synthetic_train_export_cpu_reload_and_evaluate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); data, config, state=fixture(root)
            run=root/"run"
            train_experiment(config,run,root)
            manifest=read_json(run/"manifest.json")
            self.assertEqual(manifest["train_eligible_samples"],7)
            self.assertFalse(manifest["test_used"])
            self.assertTrue(run.with_suffix('.zip').exists())
            original_state=sha256(run/"preprocessor.json")
            output=root/"evaluation"
            evaluate_bundle(run,data,output,"cpu",root)
            metrics=read_json(output/"metrics.json")
            self.assertTrue(metrics["parity_passed"])
            self.assertLess(metrics["max_probability_difference"],1e-6)
            self.assertEqual(metrics["event_metrics"]["confirmed_events"],2)
            self.assertEqual(original_state,sha256(run/"preprocessor.json"))
            with self.assertRaises(FileExistsError):
                train_experiment(config,run,root)
            with self.assertRaises(FileExistsError):
                evaluate_bundle(run,data,output,"cpu",root)

    def test_hash_rejection_before_run_publication_and_test_split_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); data, config, state=fixture(root)
            with self.assertRaisesRegex(ValueError,"final test remains reserved"):
                load_split(data,"test_final_reserved",state)
            (data/"train_labels.csv.gz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                train_experiment(config,root/"run",root)
            self.assertFalse((root/"run").exists())

    def test_colab_packaging_whitelist_and_input_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); data, config, state=fixture(root)
            for name in CODE:
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(STAGING/name,target)
            (root/".env").write_text("should not be packaged")
            (data/"test_inputs.npz").write_bytes(b"reserved")
            result=package_experiment(config,root/"colab.zip",root)
            with zipfile.ZipFile(result["path"]) as z:
                self.assertNotIn('.env',z.namelist())
                self.assertNotIn('data/test_inputs.npz',z.namelist())
                manifest=json.loads(z.read('colab_package_manifest.json'))
                for name,digest in manifest['files_sha256'].items():
                    self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),digest)
            with self.assertRaises(FileExistsError):
                package_experiment(config,root/"colab.zip",root)

    def test_event_coverage_keeps_events_without_evaluable_opportunities(self):
        events=pd.DataFrame([{"event_id":"a","event_status":"pit_stop","Temporada":2024,
            "Ronda":1,"DriverNumber":"1","entry_lap":4,"entry_time":"0 days 00:00:45"},
            {"event_id":"b","event_status":"pit_stop","Temporada":2024,
            "Ronda":1,"DriverNumber":"2","entry_lap":4,"entry_time":"0 days 00:00:45"}])
        preds=pd.DataFrame([{"Temporada":2024,"Ronda":1,"DriverNumber":"1","prediction_lap":3,
                             "decision_time_seconds":30,"target":1,"alert":True}])
        rows,metrics=event_coverage(events,preds)
        self.assertEqual(metrics['confirmed_events'],2)
        self.assertEqual(metrics['events_without_evaluable_opportunity'],1)
        self.assertEqual(metrics['recall_per_evaluable_event'],1)

    def test_decision_anchor_join_rejects_wrong_key_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); data, config, state=fixture(root)
            predictions=pd.read_csv(data/'validation_metadata.csv.gz',dtype={'DriverNumber':str})
            anchors=pd.read_csv(root/'sample_index.csv.gz',dtype={'DriverNumber':str})
            joined=attach_decision_times(predictions,anchors.iloc[::-1])
            self.assertEqual(joined.decision_time_seconds.tolist(),list(range(10,81,10)))
            wrong=anchors.copy();wrong.loc[0,'prediction_lap']=99
            with self.assertRaisesRegex(ValueError,'Missing or invalid causal decision anchor'):
                attach_decision_times(predictions,wrong)
            with self.assertRaisesRegex(ValueError,'unique complete sample_id'):
                attach_decision_times(predictions,pd.concat([anchors,anchors.iloc[:1]]))


if __name__=='__main__':
    unittest.main()

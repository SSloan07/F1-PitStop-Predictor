"""Companion config packaging checks using only synthetic train/validation data."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

from test_rnn_experiment import CODE, STAGING, fixture, package_experiment
from f1_pitstop.rnn_data import read_json, write_json


class CompanionConfigPackagingTests(unittest.TestCase):
    def _fixture(self, root):
        _, original_config, _ = fixture(root)
        for name in CODE:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = STAGING / name
            if name == "scripts/entrenar_comparacion_capacidad.py" and not source.is_file():
                # The runner may be under development in parallel; these tests
                # exercise packaging bytes, not execution of that script.
                target.write_text("# Synthetic paired runner fixture\n", encoding="utf-8")
            else:
                shutil.copyfile(source, target)
        main = root / "configs/experimento_rnn16_v1.json"
        companion = root / "configs/experimento_rnn32_pareado_v1.json"
        config = read_json(original_config)
        config["companion_configs"] = [companion.relative_to(root).as_posix()]
        write_json(main, config)
        reference = read_json(original_config)
        reference["model"]["hidden_size"] = 32
        write_json(companion, reference)
        (root / ".env").write_text("TOKEN=must_not_be_packaged\n", encoding="utf-8")
        return main, companion

    def test_companion_is_present_hashed_and_main_identity_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main, companion = self._fixture(root)
            result = package_experiment(main, root / "output/paired.zip", root)
            with zipfile.ZipFile(result["path"]) as archive:
                manifest = json.loads(archive.read("colab_package_manifest.json"))
                relative = companion.relative_to(root).as_posix()
                digest = hashlib.sha256(companion.read_bytes()).hexdigest()
                self.assertEqual(manifest["experiment_config"], main.relative_to(root).as_posix())
                self.assertEqual(manifest["companion_configs"], [relative])
                self.assertEqual(manifest["companion_configs_sha256"], {relative: digest})
                self.assertEqual(manifest["files_sha256"][relative], digest)
                self.assertEqual(archive.read(relative), companion.read_bytes())
                self.assertIn("scripts/entrenar_comparacion_capacidad.py", archive.namelist())
                self.assertNotIn(".env", archive.namelist())
                self.assertFalse(manifest["test_arrays_included"])
                self.assertFalse(result["test_arrays_included"])
                for name, expected in manifest["files_sha256"].items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected)

    def test_existing_f2_and_f1_configs_without_companions_still_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main, _ = self._fixture(root)
            for name, experiment_id in (("experimento_rnn_v1.json", "rnn_reference_v1"),
                                        ("experimento_rnn_f1_v1.json", "rnn_reference_f1_v1")):
                with self.subTest(experiment_id=experiment_id):
                    config = read_json(main)
                    config.pop("companion_configs")
                    config["experiment_id"] = experiment_id
                    config_path = root / "configs" / name
                    write_json(config_path, config)
                    output = root / f"{experiment_id}.zip"
                    package_experiment(config_path, output, root)
                    with zipfile.ZipFile(output) as archive:
                        manifest = json.loads(archive.read("colab_package_manifest.json"))
                        self.assertEqual(manifest["experiment_config"], f"configs/{name}")
                        self.assertEqual(manifest["companion_configs"], [])
                        self.assertEqual(manifest["companion_configs_sha256"], {})
                        self.assertNotIn("configs/experimento_rnn32_pareado_v1.json", archive.namelist())
                        self.assertNotIn(".env", archive.namelist())
                        self.assertFalse(manifest["test_arrays_included"])

    def test_invalid_declarations_fail_before_creating_output_or_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main, companion = self._fixture(root)
            relative = companion.relative_to(root).as_posix()
            invalid = [
                None, relative, {}, [None], [1], [True], [[relative]],
                [str(companion)], ["../configs/reference.json"], ["configs/../.env"],
                [".env"], ["configs/.env"], ["configs/reference.txt"],
                ["configs/nested/reference.json"], ["configs/missing.json"],
                ["configs/./experimento_rnn32_pareado_v1.json"],
                [relative, relative], [main.relative_to(root).as_posix()],
            ]
            for declaration in invalid:
                with self.subTest(companion_configs=declaration):
                    config = read_json(main)
                    config["companion_configs"] = declaration
                    write_json(main, config)
                    output = root / "uncreated/result.zip"
                    with self.assertRaises(ValueError):
                        package_experiment(main, output, root)
                    self.assertFalse(output.exists())
                    self.assertFalse(output.parent.exists())

    def test_symlink_escape_duplicate_alias_and_main_alias_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
            root = Path(temporary)
            main, companion = self._fixture(root)
            external = Path(outside) / "external.json"
            write_json(external, {"external": True})
            targets = {
                "escape.json": external,
                "env_alias.json": root / ".env",
                "alias.json": companion,
                "main_alias.json": main,
            }
            for name, target in targets.items():
                (root / "configs" / name).symlink_to(target)
            for declaration in (["configs/escape.json"], ["configs/env_alias.json"],
                                ["configs/experimento_rnn32_pareado_v1.json", "configs/alias.json"],
                                ["configs/main_alias.json"]):
                with self.subTest(companion_configs=declaration):
                    config = read_json(main)
                    config["companion_configs"] = declaration
                    write_json(main, config)
                    output = root / "uncreated/result.zip"
                    with self.assertRaises(ValueError):
                        package_experiment(main, output, root)
                    self.assertFalse(output.parent.exists())

    def test_companion_must_be_json_object_and_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main, companion = self._fixture(root)
            for contents in ("not json", "[]", "null"):
                with self.subTest(contents=contents):
                    companion.write_text(contents, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        package_experiment(main, root / "uncreated/result.zip", root)
                    self.assertFalse((root / "uncreated").exists())
            companion.unlink()
            companion.mkdir()
            with self.assertRaises(ValueError):
                package_experiment(main, root / "uncreated/result.zip", root)
            self.assertFalse((root / "uncreated").exists())


if __name__ == "__main__":
    unittest.main()

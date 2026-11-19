"""Guards of the single-run final evaluation (no data is loaded in these tests)."""
from pathlib import Path
import sys
import tempfile
import unittest

STAGING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGING / "scripts"))
import evaluar_modelo_final as final


class FinalEvaluationGuardTest(unittest.TestCase):
    def test_test_split_requires_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit):
            final.run("test", Path(tmp) / "out", confirmed=False)

    def test_existing_output_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileExistsError):
            final.run("test", Path(tmp), confirmed=True)

    def test_frozen_model_and_sources_are_unchanged(self):
        model, features = final.verify_frozen()
        self.assertEqual(model["threshold"], 0.14)
        self.assertEqual(features["count"], 66)


if __name__ == "__main__":
    unittest.main()

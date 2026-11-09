"""Compatibility entrypoint for the conservative audit and Compound normalizer.

Default behavior remains the read-only quality audit. Pass ``--normalize-compound``
to create a new, audited historical matrix under the approved exact-literal rule.
"""
import sys
from pathlib import Path
import runpy


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
AUDITOR = SCRIPTS / "auditar_calidad_matriz.py"
NORMALIZER = SCRIPTS / "normalizar_compound.py"


def main():
    if "--normalize-compound" in sys.argv[1:]:
        sys.argv.remove("--normalize-compound")
        runpy.run_path(str(NORMALIZER), run_name="__main__")
    else:
        runpy.run_path(str(AUDITOR), run_name="__main__")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CLI to materialize causal windows from the published historical sources."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1_pitstop.causal_samples import ContractError, build_samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/muestras_causales_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "EDA/dataset/muestras_causales_v1_3")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    try:
        print(build_samples(args.config, output))
    except (ContractError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

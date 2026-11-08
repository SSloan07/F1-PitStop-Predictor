"""Load an exported model and reproduce validation locally without retuning."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_pitstop.rnn_experiment import evaluate_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Unzipped exported run directory")
    parser.add_argument("--data", type=Path, default=ROOT / "EDA/dataset/muestras_preprocesadas_v1_3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = parser.parse_args()
    print(evaluate_bundle(args.bundle, args.data, args.output, args.device, root=ROOT))


if __name__ == "__main__":
    main()

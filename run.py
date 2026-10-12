"""Usage: python run.py train | predict --input examples/sample_companies.csv"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

def main():
    parser = argparse.ArgumentParser(description="Corporate Bankruptcy Prediction")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="reproduce the complete fixed experiment")
    pr = sub.add_parser("predict", help="predict from a CSV with X1-X64 columns")
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    if args.command == "train":
        from bankruptcy.train import train
        train(ROOT)
    else:
        import pandas as pd
        from bankruptcy.predict import load_bundle, predict_frame
        result = predict_frame(load_bundle(ROOT), pd.read_csv(args.input, na_values=["?"]))
        result.to_csv(args.output, index=False)
        print(result.to_string(index=False))
        print(f"Saved {args.output}")

if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse, json
from ..eval.runner import evaluate_dataset

def main():
    ap = argparse.ArgumentParser(description="Evaluate SPLIT-RAG: run full pipeline & metrics.")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--split", default="test", choices=["dev","test"])
    args = ap.parse_args()
    res = evaluate_dataset(args.cfg, split=args.split)
    print(json.dumps(res.macro, indent=2))

if __name__ == "__main__":
    main()

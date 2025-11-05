from __future__ import annotations
import argparse, yaml, json
from pathlib import Path
from ..matching.registry import build_registry, save_registry, RegistryConfig

def _load_yaml(p: str | Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser(description="Build agent registry 𝓡 (Sec. 3.2).")
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()

    cfg = _load_yaml(args.cfg)
    P = cfg["paths"]
    m = cfg["matching"]

    reg = build_registry(RegistryConfig(
        index_dir=Path(P["index_dir"]),
        q_train_jsonl=Path(P["train_q"]),
        N_max=m["N_max"],
        theta_coh=m["theta_coh"]
    ))
    save_registry(reg, P["registry_json"])
    print(f"Saved registry to {P['registry_json']}")

if __name__ == "__main__":
    main()

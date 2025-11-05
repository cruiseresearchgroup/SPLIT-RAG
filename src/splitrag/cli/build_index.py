from __future__ import annotations
import argparse, yaml
from pathlib import Path
from ..kg.graph import load_kg
from ..indexer.build_index import build_partitioned_kg, IndexBuildConfig

def _load_yaml(p: str | Path):
    import yaml
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser(description="Build partitioned KG index (Sec. 3.1).")
    ap.add_argument("--cfg", required=True, help="Path to dataset YAML (e.g., configs/webqsp.yaml)")
    args = ap.parse_args()

    cfg = _load_yaml(args.cfg)
    P = cfg["paths"]

    kg = load_kg(P["kg_tsv"], P["ent_json"], P["rel_json"])
    ib = cfg["indexer"]
    ibc = IndexBuildConfig(
        lambda_size=ib["lambda_size"],
        theta_merge=ib["theta_merge"],
        eta_max_nodes=ib["eta_max_nodes"],
        tau_min_nodes=ib["tau_min_nodes"],
        topk_per_seed=ib["topk_per_seed"],
        max_paths_per_question=ib["max_paths_per_question"],
        seed=cfg.get("seed", 42),
        out_dir=Path(P["index_dir"])
    )
    build_partitioned_kg(
        config=ibc,
        kg=kg,
        q_train_jsonl=P["train_q"],
        ent_map_json=P["ent_json"]
    )

if __name__ == "__main__":
    main()

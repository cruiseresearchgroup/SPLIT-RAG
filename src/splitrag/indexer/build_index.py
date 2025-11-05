from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path
from ..utils.io import ensure_dir, write_jsonl, write_json, read_json, read_jsonl
from ..utils.logging import get_logger, timer
from ..utils.rng import set_seed
from ..kg.graph import KG, load_kg, Triple
from .slicing import slice_paths
from .partition import (
    initialize_seeds_from_slices,
    greedy_merge,
    merge_small_subgraphs,
    Subgraph
)

logger = get_logger(__name__)

@dataclass
class IndexBuildConfig:
    lambda_size: float
    theta_merge: float
    eta_max_nodes: int
    tau_min_nodes: int
    topk_per_seed: int
    max_paths_per_question: int
    seed: int
    out_dir: Path

def _collect_training_slices(kg: KG,
                             q_train: List[Dict],
                             topk_per_seed: int,
                             max_paths_per_question: int) -> List[Triple]:
    """
    Gather path slices Ṕ from training questions. If gold paths exist, use them;
    otherwise enumerate ≤2-hop paths from linked entities (bounded).
    """
    all_slices: List[Triple] = []
    for r in q_train:
        entities = [e["id"] for e in r.get("entities", []) if "id" in e]
        gold = r.get("gold_paths")
        if gold:
            paths: List[List[Triple]] = []
            for gp in gold:
                # gp is [["e1","r","e2"],["e2","r2","e3"],...]
                triples = [(t[0], t[1], t[2]) for t in gp]
                paths.append(triples)
        else:
            # bounded path enumeration
            paths = kg.enumerate_paths_le2(entities, topk_per_seed=topk_per_seed)
            paths = paths[:max_paths_per_question]
        all_slices.extend(slice_paths(paths))
    return all_slices

def build_partitioned_kg(config: IndexBuildConfig,
                         kg: KG,
                         q_train_jsonl: str | Path,
                         ent_map_json: str | Path) -> List[Subgraph]:
    """
    Orchestrate:  Q_train, D  →  Ḱ={Ď_1,...,Ď_M}.
    Produces JSONL with subgraphs (triples) and metadata in config.out_dir.
    """
    set_seed(config.seed)
    out_dir = ensure_dir(config.out_dir)

    q_train = read_jsonl(q_train_jsonl)

    with timer("Collect training path slices", logger):
        slices = _collect_training_slices(
            kg=kg,
            q_train=q_train,
            topk_per_seed=config.topk_per_seed,
            max_paths_per_question=config.max_paths_per_question
        )
        logger.info(f"Collected {len(slices)} atomic slices (triples) from training")

    with timer("Initialize seed subgraphs", logger):
        seeds = initialize_seeds_from_slices(slices)
        total_nodes = len(kg.ent_meta)
        logger.info(f"Seed subgraphs: {len(seeds)}")

    with timer("Greedy merge by information gain", logger):
        merged = greedy_merge(
            subs=seeds,
            lambda_size=config.lambda_size,
            theta_merge=config.theta_merge,
            eta_max_nodes=config.eta_max_nodes,
            total_nodes=total_nodes
        )
        logger.info(f"After greedy merge: {len(merged)} subgraphs")

    with timer("Merge tiny residuals", logger):
        final = merge_small_subgraphs(merged, tau_min_nodes=config.tau_min_nodes)
        logger.info(f"Final subgraphs: {len(final)}")

    # Persist index
    sg_jsonl = out_dir / "subgraphs.jsonl"
    rows = []
    for sg in final:
        rows.append({
            "id": sg.id,
            "num_nodes": len(sg.nodes),
            "num_triples": len(sg.triples),
            "triples": sg.triples
        })
    write_jsonl(sg_jsonl, rows)
    write_json(out_dir / "meta.json", {
        "lambda_size": config.lambda_size,
        "theta_merge": config.theta_merge,
        "eta_max_nodes": config.eta_max_nodes,
        "tau_min_nodes": config.tau_min_nodes,
        "topk_per_seed": config.topk_per_seed,
        "max_paths_per_question": config.max_paths_per_question,
        "seed": config.seed,
        "num_subgraphs": len(final)
    })
    logger.info(f"Wrote partitioned KG to {out_dir}")
    return final

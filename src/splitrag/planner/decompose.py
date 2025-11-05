from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from ..matching.registry import Registry
from .similarity import SimilarityScorer, SimilarityConfig
from .path_seed import PathSeedResult
from .select import SelectConfig, greedy_cover_groups, AgentSelection

Triple = Tuple[str, str, str]

@dataclass
class Task:
    """
    A decomposed unit assigned to a single agent group.
    For simplicity, a task holds a set of atomic triples that the agent should retrieve/verify.
    """
    agent_group: int
    triples: List[Triple]
    target_subgraphs: List[int]

@dataclass
class Plan:
    """
    Planner output:
      - selected agent groups
      - list of tasks (one or more per group)
      - provenance (similarity/routing info)
    """
    groups: List[int]
    tasks: List[Task]
    provenance: Dict[str, float]  # e.g., {"sim_top": 0.82, "num_seed_sg": 12}

def build_agent_plan(
    registry: Registry,
    q_new: Dict,                    # {"qid":..., "Qs":..., "Qe":..., "paths":[[tri..],...], "entities":[...]}
    train_questions: List[Dict],
    sim_cfg: SimilarityConfig,
    sel_cfg: SelectConfig,
    seed: PathSeedResult
) -> Plan:
    """
    Implements \S3.3: similarity transfer vs. path-driven seeding, and greedy selection of agent groups.
    - If top-1 similarity >= theta_direct, prefer similarity-transfer (reusing groups covering q_sim)
    - Otherwise use path-driven seeds (subgraphs hit by φ(q_new)).
    - In both cases, perform greedy cover to choose up to B groups.
    - Decompose tasks by assigning atomic triples to the nearest selected group.
    """
    scorer = SimilarityScorer(sim_cfg, registry)

    # 1) Rank similar training questions
    ranked = scorer.rank_similar(q_new.get("Qe",""), seed.sg_hits, train_questions)
    top_sim = ranked[0][1] if ranked else 0.0
    top_cov = set(ranked[0][3]) if ranked else set()
    source = "paths"

    # 2) Candidate groups from similarity or paths
    cand_groups: List[int] = []
    required_sg: Set[int] = set()

    if ranked and top_sim >= sim_cfg.theta_direct:
        # Similarity transfer
        # Candidate groups are those touching the coverage subgraphs of top-sim question
        # We map each sg_id to its group via registry.A
        sg2group: Dict[int, int] = {}
        for g in registry.A:
            for sid in g.subgraph_ids:
                sg2group[sid] = g.id
        cand_groups = list(dict.fromkeys([sg2group[s] for s in top_cov if s in sg2group]))
        required_sg = top_cov
        source = "similarity"
    else:
        # Path-driven
        sg2group: Dict[int, int] = {}
        for g in registry.A:
            for sid in g.subgraph_ids:
                sg2group[sid] = g.id
        cand_groups = list(dict.fromkeys([sg2group[s] for s in seed.sg_hits if s in sg2group]))
        required_sg = set(seed.sg_hits)
        source = "paths"

    # 3) Greedy selection up to B
    sel = greedy_cover_groups(registry, cand_groups, required_sg, B=sel_cfg.B)
    sel.source = source

    # 4) Decompose tasks Ψ(q): cluster atomic triples by nearest selected group (majority vote via subgraph owners)
    # Build triple -> owning group(s) via any sg that contains the triple; choose the most frequent group
    # For efficiency, build inverted index (triple -> [sg]) on-the-fly from index_dir
    # Here we read subgraphs.jsonl once.
    import json, collections
    from pathlib import Path

    inv_triple_to_groups: Dict[Triple, List[int]] = {}
    p = Path(registry.index_dir) / "subgraphs.jsonl"
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            sid = row["id"]
            # Find group id for this sid if selected
            gid = None
            for g in registry.A:
                if g.id in sel.chosen_groups and sid in g.subgraph_ids:
                    gid = g.id
                    break
            if gid is None:
                continue
            for h, r, t in row["triples"]:
                tri: Triple = (h, r, t)
                inv_triple_to_groups.setdefault(tri, []).append(gid)

    # Assign each atomic triple to the most frequent selected group that contains it
    group_to_triples: Dict[int, List[Triple]] = {gid: [] for gid in sel.chosen_groups}
    for tri in seed.atomic:
        gids = inv_triple_to_groups.get(tri, [])
        # If multiple groups, majority vote; if none, assign to the first selected group
        if gids:
            # count occurrences
            counts = collections.Counter(gids)
            gid_star = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            group_to_triples.setdefault(gid_star, []).append(tri)
        else:
            if sel.chosen_groups:
                group_to_triples[sel.chosen_groups[0]].append(tri)

    tasks: List[Task] = []
    for gid, tris in group_to_triples.items():
        if not tris:
            continue
        # target subgraphs = intersection of the group's subgraphs and required set (for bookkeeping)
        group_sg = set()
        for g in registry.A:
            if g.id == gid:
                group_sg = set(g.subgraph_ids)
                break
        target = list(sorted(group_sg & required_sg))
        tasks.append(Task(agent_group=gid, triples=tris, target_subgraphs=target))

    prov = {
        "sim_top": float(top_sim),
        "num_seed_sg": float(len(seed.sg_hits)),
        "source": sel.source
    }
    return Plan(groups=sel.chosen_groups, tasks=tasks, provenance=prov)

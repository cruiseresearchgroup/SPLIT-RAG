from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Iterable
from pathlib import Path
import json
import numpy as np

Triple = Tuple[str, str, str]

@dataclass
class Subgraph:
    id: int
    triples: List[Triple]
    nodes: Set[str]

def load_subgraphs(index_dir: str | Path) -> List[Subgraph]:
    """
    Load partitioned subgraphs produced by indexer/build_index.py.
    Expects index_dir/subgraphs.jsonl with fields:
      { "id": int, "num_nodes": int, "num_triples": int, "triples": [[h,r,t], ...] }
    """
    subgraphs: List[Subgraph] = []
    p = Path(index_dir) / "subgraphs.jsonl"
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            triples = [tuple(t) for t in row["triples"]]
            nodes = set([t[0] for t in triples] + [t[2] for t in triples])
            subgraphs.append(Subgraph(id=row["id"], triples=triples, nodes=nodes))
    return subgraphs

def _triples_set(sg: Subgraph) -> Set[Triple]:
    return set(sg.triples)

def compute_association_matrix(
    questions: List[Dict],
    subgraphs: List[Subgraph],
) -> Tuple[np.ndarray, List[Set[int]], Dict[int, Set[Triple]]]:
    """
    Build the association matrix A_ij (Eq. 4) and coverage sets C_i (Eq. 5).
    - questions: each dict may contain "qid" and either "gold_paths" or "paths" (list of list of triples).
      Path format: [["e1","r","e2"], ...] per step; will be converted to tuples.
    - subgraphs: loaded from partition.
    Returns:
      A: (m x n) dense float32 matrix in [0,1]
      coverage_sets: list of sets C_i = { j | A_ij > 0 }
      sg_triple_sets: cache mapping sg.id -> set(triples) for faster membership tests
    """
    m = len(questions)
    n = len(subgraphs)

    sg_triple_sets: Dict[int, Set[Triple]] = {sg.id: _triples_set(sg) for sg in subgraphs}

    A = np.zeros((m, n), dtype=np.float32)
    coverage_sets: List[Set[int]] = [set() for _ in range(m)]

    for i, q in enumerate(questions):
        raw_paths = q.get("gold_paths") or q.get("paths") or []
        # Normalize to list[List[Triple]]
        paths: List[List[Triple]] = []
        for p in raw_paths:
            if not p:
                continue
            triples = [tuple(t) for t in p]  # type: ignore
            paths.append(triples)
        if not paths:
            continue  # no coverage if no paths are recorded

        denom = float(len(paths))
        for j, sg in enumerate(subgraphs):
            S = sg_triple_sets[sg.id]
            # A_ij = fraction of q_i paths fully contained in Ď_j
            count = 0
            for p in paths:
                if all(t in S for t in p):
                    count += 1
            if count > 0:
                A[i, j] = count / denom
                coverage_sets[i].add(j)

    return A, coverage_sets, sg_triple_sets

def frequency_per_subgraph(coverage_sets: List[Set[int]], n_subgraphs: int) -> List[int]:
    """
    used for trimming by frequency.
    """
    freq = [0] * n_subgraphs
    for Ci in coverage_sets:
        for j in Ci:
            freq[j] += 1
    return freq

def coverage_density(coverage_sets: List[Set[int]]) -> List[float]:
    """
    (density) used to prioritize sets.
    """
    m = len(coverage_sets)
    densities = [0.0] * m
    for i in range(m):
        Ci = coverage_sets[i]
        if not Ci:
            densities[i] = 0.0
            continue
        acc = 0
        for j in range(m):
            if i == j:
                continue
            Cj = coverage_sets[j]
            if not Cj:
                continue
            acc += len(Ci & Cj)
        densities[i] = acc / max(1.0, len(Ci)) ** 0.5
    return densities

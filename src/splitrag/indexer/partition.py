from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Set
import math, itertools, random

Triple = Tuple[str, str, str]

@dataclass
class Subgraph:
    id: int
    triples: List[Triple]
    nodes: Set[str]    # vertex set V_s

def _cond_entropy(slices: List[Triple]) -> float:
    """
    H(Ṕ | s): entropy over triple frequency within subgraph s.
    """
    if not slices:
        return 0.0
    freq: Dict[Triple, int] = {}
    for t in slices: freq[t] = freq.get(t, 0) + 1
    total = float(sum(freq.values()))
    H = 0.0
    for c in freq.values():
        p = c / total
        H -= p * math.log(max(p, 1e-12))
    return H

def _size_penalty(num_nodes: int, total_nodes: int) -> float:
    """
    H_size(s) = |V_s|/|V| * log(|V|/|V_s|)
    """
    if num_nodes <= 0 or total_nodes <= 0:
        return 0.0
    ratio = num_nodes / total_nodes
    return ratio * math.log(max(total_nodes / num_nodes, 1.0))

def ig_of_partition(subs: List[Subgraph],
                    lambda_size: float,
                    total_nodes: int) -> float:
    return sum(_cond_entropy(sg.triples) - lambda_size * _size_penalty(len(sg.nodes), total_nodes)
               for sg in subs)

def greedy_merge(subs: List[Subgraph],
                 lambda_size: float,
                 theta_merge: float,
                 eta_max_nodes: int,
                 total_nodes: int,
                 max_iter: int = 10_000) -> List[Subgraph]:
    """
    Greedy merge pairs that increase IG, respecting |V_s| ≤ η_max.
    """
    # Precompute pair iterator; in practice use a heap keyed by ΔIG if needed.
    iters = 0
    improved = True
    while improved and iters < max_iter:
        iters += 1
        improved = False
        best_gain = theta_merge
        best_pair: Tuple[int,int] | None = None
        # Try all unordered pairs (can be optimized with blocking)
        for i in range(len(subs)):
            for j in range(i+1, len(subs)):
                si, sj = subs[i], subs[j]
                merged_nodes = si.nodes | sj.nodes
                if len(merged_nodes) > eta_max_nodes:
                    continue
                # ΔIG = IG(si∪sj) - IG(si) - IG(sj)
                merged_triples = si.triples + sj.triples
                gain = (_cond_entropy(merged_triples) - lambda_size * _size_penalty(len(merged_nodes), total_nodes)) \
                     - (_cond_entropy(si.triples) - lambda_size * _size_penalty(len(si.nodes), total_nodes)) \
                     - (_cond_entropy(sj.triples) - lambda_size * _size_penalty(len(sj.nodes), total_nodes))
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (i, j)
        if best_pair is not None:
            i, j = best_pair
            si, sj = subs[i], subs[j]
            merged = Subgraph(
                id=min(si.id, sj.id),
                triples=si.triples + sj.triples,
                nodes=si.nodes | sj.nodes,
            )
            # Replace i with merged; remove j
            subs[i] = merged
            subs.pop(j)
            improved = True
    return subs

def initialize_seeds_from_slices(slices: List[Triple]) -> List[Subgraph]:
    """
    Initialize candidate subgraphs from atomic slices (triples): group identical triples.
    """
    buckets: Dict[Triple, List[Triple]] = {}
    for t in slices:
        buckets.setdefault(t, []).append(t)
    subs: List[Subgraph] = []
    for k, grp in buckets.items():
        h, r, o = k
        subs.append(Subgraph(
            id=len(subs),
            triples=grp.copy(),
            nodes={h, o}
        ))
    return subs

def merge_small_subgraphs(subs: List[Subgraph],
                          tau_min_nodes: int) -> List[Subgraph]:
    """
    Merge very small subgraphs into nearest neighbors (by triple overlap).
    """
    large: List[Subgraph] = [s for s in subs if len(s.nodes) >= tau_min_nodes]
    small: List[Subgraph] = [s for s in subs if len(s.nodes) < tau_min_nodes]
    if not small or not large:
        return subs
    # Build simple overlap map
    def overlap(a: Subgraph, b: Subgraph) -> int:
        aset = set(a.triples)
        return sum(1 for t in b.triples if t in aset)
    for s in small:
        best = max(large, key=lambda L: overlap(s, L))
        best.triples.extend(s.triples)
        best.nodes |= s.nodes
    # Re-number
    final = []
    for s in large:
        s.id = len(final)
        final.append(s)
    return final

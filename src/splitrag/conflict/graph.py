from __future__ import annotations
from typing import List, Tuple, Dict, Set
from .rules import RuleConfig, is_conflict

Triple = Tuple[str, str, str]

def build_compatibility_graph(
    triples: List[Triple],
    weights: List[float],
    rules: RuleConfig
) -> Tuple[List[Set[int]], List[float]]:
    """
    Build an undirected graph where an edge (i,j) indicates compatibility (no conflict).
    This allows selecting a (near) max-weight clique as the consistent subset.
    Returns:
      adj: list of neighbor sets (compatibility edges)
      w:   node weights aligned to triples
    """
    n = len(triples)
    adj: List[Set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if not is_conflict(triples[i], triples[j], rules):
                adj[i].add(j)
                adj[j].add(i)
    return adj, weights

def max_weight_clique_greedy(
    adj: List[Set[int]],
    weights: List[float],
    beam: int = 4
) -> List[int]:
    """
    Lightweight beam-search for approximately maximal-weight clique.
    Works well in practice for dozens to low hundreds of nodes (post-dedup).
    """
    n = len(adj)
    # Order nodes by weight descending as initial candidates
    order = sorted(range(n), key=lambda i: weights[i], reverse=True)
    beams: List[List[int]] = [[]]  # each beam holds a current clique (node indices)

    best_clique: List[int] = []
    best_weight = 0.0

    for seed in order:
        new_beams: List[Tuple[float, List[int]]] = []
        for C in beams:
            # Seed or extend by seed if compatible with all in C
            if all(seed in adj[v] or seed == v for v in C):
                C2 = C + [seed]
                w2 = sum(weights[i] for i in C2)
                new_beams.append((w2, C2))
                if w2 > best_weight:
                    best_weight = w2
                    best_clique = C2
            # also keep original C
            new_beams.append((sum(weights[i] for i in C), C))
        # Keep top-k beams
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = [c for _, c in new_beams[:beam]]
    return best_clique

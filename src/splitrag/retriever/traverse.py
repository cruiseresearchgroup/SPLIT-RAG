from __future__ import annotations
from typing import Dict, List, Tuple, Set, Iterable
from collections import defaultdict, deque

Triple = Tuple[str, str, str]

class SubgraphAdj:
    """
    Adjacency structure restricted to a subgraph (triple list).
    """
    def __init__(self, triples: List[Triple]):
        self.out = defaultdict(list)  # e -> [(r, e2)]
        for h, r, t in triples:
            self.out[h].append((r, t))

    def neighbors(self, e: str) -> List[Tuple[str, str]]:
        return self.out.get(e, [])

def enumerate_paths_in_subgraph(
    subgraph_triples: List[Triple],
    seed_entities: Iterable[str],
    topk_per_seed: int = 5,
    max_depth: int = 2
) -> List[List[Triple]]:
    """
    Enumerate simple 1/2-hop paths inside a given subgraph starting from seed entities.
    - Bounded by topk_per_seed per frontier expansion.
    - Non-exhaustive by design; good enough as retrieval candidate generator.
    """
    adj = SubgraphAdj(subgraph_triples)
    paths: List[List[Triple]] = []

    for s in seed_entities:
        # 1-hop
        for r1, e2 in adj.neighbors(s)[:topk_per_seed]:
            p1 = [(s, r1, e2)]
            paths.append(p1)
            if max_depth >= 2:
                # 2-hop
                cnt = 0
                for r2, e3 in adj.neighbors(e2):
                    if e3 == s: 
                        continue
                    paths.append([(s, r1, e2), (e2, r2, e3)])
                    cnt += 1
                    if cnt >= topk_per_seed:
                        break
    return paths

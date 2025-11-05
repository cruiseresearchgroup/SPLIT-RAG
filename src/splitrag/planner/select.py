from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import math
from ..matching.registry import Registry

@dataclass
class SelectConfig:
    B: int = 3                     # max number of agent groups to activate
    prefer_sim_transfer: bool = True

@dataclass
class AgentSelection:
    chosen_groups: List[int]       # agent group ids
    covered_subgraphs: Set[int]    # union of subgraphs covered
    source: str                    # "similarity", "paths", or "hybrid"

def greedy_cover_groups(
    registry: Registry,
    candidate_groups: List[int],
    required_sg: Set[int],
    B: int
) -> AgentSelection:
    """
    Greedy max-coverage over agent groups to cover as many required subgraphs as possible.
    """
    # Build group -> subgraph set
    gid2sg: Dict[int, Set[int]] = {}
    for g in registry.A:
        gid2sg[g.id] = set(g.subgraph_ids)

    chosen: List[int] = []
    covered: Set[int] = set()
    req = set(required_sg)

    avail = list(dict.fromkeys(candidate_groups))  # unique, keep order
    for _ in range(B):
        best_gid, best_gain = None, -1
        for gid in avail:
            gain = len((gid2sg[gid] & req) - covered)
            if gain > best_gain:
                best_gid, best_gain = gid, gain
        if best_gid is None or best_gain <= 0:
            # No gain; if we still have slots, pick the densest group intersecting req
            remain = [gid for gid in avail if gid not in chosen]
            if not remain: break
            densest = max(remain, key=lambda g: len(gid2sg[g] & req))
            chosen.append(densest)
            covered |= gid2sg[densest]
            continue
        chosen.append(best_gid)
        covered |= gid2sg[best_gid]
        avail.remove(best_gid)
        if len(chosen) >= B:
            break
    return AgentSelection(chosen_groups=chosen, covered_subgraphs=covered, source="hybrid")

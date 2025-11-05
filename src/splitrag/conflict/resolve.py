from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from .graph import build_compatibility_graph, max_weight_clique_greedy
from .rules import RuleConfig, default_rules

Triple = Tuple[str, str, str]

@dataclass
class ResolveConfig:
    beam: int = 4

@dataclass
class ResolveResult:
    keep_indices: List[int]
    kept_triples: List[Triple]
    dropped_indices: List[int]

def score_triples_by_support(
    triples: List[Triple],
    who_supports: Dict[Triple, List[int]],
    agent_confidence: Dict[int, float] | None = None
) -> List[float]:
    """
    Weight of a triple = sum of confidences of supporting agents/groups.
    If confidence map is None, use 1.0 per supporter.
    """
    agent_confidence = agent_confidence or {}
    w: List[float] = []
    for tri in triples:
        supporters = who_supports.get(tri, [])
        s = 0.0
        for gid in supporters:
            s += agent_confidence.get(gid, 1.0)
        w.append(s if s > 0 else 1.0)  # default min weight 1 to avoid zeros
    return w

def resolve_conflicts(
    triples: List[Triple],
    who_supports: Dict[Triple, List[int]],
    agent_confidence: Dict[int, float] | None = None,
    rules: RuleConfig | None = None,
    cfg: ResolveConfig = ResolveConfig()
) -> ResolveResult:
    """
    Build a compatibility graph and select an approximate max-weight clique.
    Returns indices of kept triples and dropped ones.
    """
    rules = rules or default_rules()
    weights = score_triples_by_support(triples, who_supports, agent_confidence)
    adj, w = build_compatibility_graph(triples, weights, rules)
    keep = max_weight_clique_greedy(adj, w, beam=cfg.beam)
    drop = [i for i in range(len(triples)) if i not in keep]
    return ResolveResult(
        keep_indices=keep,
        kept_triples=[triples[i] for i in keep],
        dropped_indices=drop
    )

from __future__ import annotations
from typing import List, Set, Tuple

Triple = Tuple[str, str, str]

def entities_from_triples(path: List[Triple]) -> Set[str]:
    """
    Collect unique entity ids from a path (list of triples).
    """
    ents: Set[str] = set()
    for h, _, t in path:
        ents.add(h); ents.add(t)
    return ents

def match_score(path: List[Triple], task_entities: Set[str]) -> float:
    """
    Eq. (11): Match(p, t) = |Entities(p) ∩ Entities(t)| / |Entities(t)|
    where t is represented by its entity set (from atomic triples of the task).
    """
    if not task_entities:
        return 0.0
    ents_p = entities_from_triples(path)
    inter = len(ents_p & task_entities)
    return inter / max(1, len(task_entities))

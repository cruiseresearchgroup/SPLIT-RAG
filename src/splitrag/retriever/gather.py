from __future__ import annotations
from typing import List, Set, Tuple
from .match_score import match_score

Triple = Tuple[str, str, str]

def gather_triplets_from_paths(paths: List[List[Triple]]) -> List[Triple]:
    """
    Union of triples across candidate paths, preserve order of first appearance.
    """
    seen: Set[Triple] = set()
    out: List[Triple] = []
    for p in paths:
        for tri in p:
            if tri not in seen:
                seen.add(tri)
                out.append(tri)
    return out

def filter_paths_by_match(paths: List[List[Triple]],
                          task_entities: Set[str],
                          theta_match: float) -> List[List[Triple]]:
    """
    Keep only paths whose match score >= theta_match.
    """
    kept: List[List[Triple]] = []
    for p in paths:
        s = match_score(p, task_entities)
        if s >= theta_match:
            kept.append(p)
    return kept

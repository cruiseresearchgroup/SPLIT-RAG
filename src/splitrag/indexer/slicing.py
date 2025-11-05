from __future__ import annotations
from typing import List, Tuple

Triple = Tuple[str, str, str]

def slice_path_at_most2(path: List[Triple]) -> List[Triple]:
    """
    φ(p): break a path into consecutive at-most-2-hop fragments.
    For IG objective we use 1-hop slices (triples) as atomic units.
    """
    # Atomic 1-hop slices suffice for our IG accounting (Eq. 2–3 simplified).
    return path  # each triple is already 1 hop

def slice_paths(paths: List[List[Triple]]) -> List[Triple]:
    """
    Combine slices from multiple paths
    """
    slices: List[Triple] = []
    for p in paths:
        slices.extend(slice_path_at_most2(p))
    return slices

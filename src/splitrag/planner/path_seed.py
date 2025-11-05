from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Iterable, Optional
import json
from pathlib import Path

Triple = Tuple[str, str, str]

@dataclass
class PathSeedConfig:
    theta_match: float = 0.5   # min entity overlap proportion to accept a path (Eq. match)
    max_atomic: int = 128      # cap number of atomic slices kept
    # In this module, "match" is simplified to full triple membership in subgraphs.

def load_subgraphs(index_dir: str | Path) -> List[Dict]:
    rows = []
    p = Path(index_dir) / "subgraphs.jsonl"
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def build_triple_to_sg(index_dir: str | Path) -> Dict[Triple, List[int]]:
    """
    Inverted index: (h,r,t) → [sg_ids...]
    """
    inv: Dict[Triple, List[int]] = {}
    for row in load_subgraphs(index_dir):
        sid = row["id"]
        for h, r, t in row["triples"]:
            key: Triple = (h, r, t)
            inv.setdefault(key, []).append(sid)
    return inv

def atomic_slices_from_paths(paths: List[List[Triple]]) -> List[Triple]:
    """
    φ_atomic: flatten into 1-hop triples; deduplicate preserving order.
    """
    seen: Set[Triple] = set()
    out: List[Triple] = []
    for p in paths:
        for tr in p:
            tr = (tr[0], tr[1], tr[2])
            if tr not in seen:
                seen.add(tr)
                out.append(tr)
    return out

@dataclass
class PathSeedResult:
    atomic: List[Triple]      # retained atomic triples (≤ max_atomic)
    sg_hits: Set[int]         # subgraph ids touched by any retained triple

def seed_by_paths(paths: List[List[Triple]],
                  index_dir: str | Path,
                  cfg: PathSeedConfig) -> PathSeedResult:
    """
    Given candidate paths p(q_new), produce:
      - atomic slices (triples)
      - the set of subgraphs that contain at least one retained triple
    """
    atomic = atomic_slices_from_paths(paths)[: cfg.max_atomic]
    inv = build_triple_to_sg(index_dir)
    sg_hits: Set[int] = set()
    for tri in atomic:
        for sid in inv.get(tri, []):
            sg_hits.add(sid)
    return PathSeedResult(atomic=atomic, sg_hits=sg_hits)

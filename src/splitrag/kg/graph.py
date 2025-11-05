from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Set
from pathlib import Path
from ..utils.io import read_json, ensure_dir

Triple = Tuple[str, str, str]  # (head, relation, tail)

@dataclass
class KG:
    """
    Lightweight KG abstraction with adjacency lists for ≤2-hop traversal.
    """
    # Entity/Relation metadata (optional but useful)
    ent_meta: Dict[str, Dict]   # id -> {"name":..., "type":...}
    rel_meta: Dict[str, Dict]   # id -> {"name":...}

    # Adjacency
    out_adj: Dict[str, List[Tuple[str, str]]]   # e -> list of (r, e2)
    in_adj: Dict[str, List[Tuple[str, str]]]    # e -> list of (r, e1)

    def neighbors(self, e: str) -> List[Tuple[str, str]]:
        return self.out_adj.get(e, [])

    def enumerate_paths_le2(self,
                            seeds: Iterable[str],
                            topk_per_seed: int = 50) -> List[List[Triple]]:
        """
        Enumerate simple paths of length 1 or 2 starting from seed entities.
        Returns a list of paths (each path is a list of triples).
        This is bounded and non-exhaustive (top-K fanout per seed).
        """
        paths: List[List[Triple]] = []
        for s in seeds:
            # 1-hop
            for r1, e2 in self.neighbors(s)[:topk_per_seed]:
                paths.append([(s, r1, e2)])
                # 2-hop
                for r2, e3 in self.neighbors(e2)[:topk_per_seed]:
                    if e3 == s:  # avoid tiny cycle
                        continue
                    paths.append([(s, r1, e2), (e2, r2, e3)])
        return paths

def load_kg(triple_tsv: str | Path,
            ent_map_json: str | Path,
            rel_map_json: str | Path) -> KG:
    ent_meta = read_json(ent_map_json)
    rel_meta = read_json(rel_map_json)
    out_adj: Dict[str, List[Tuple[str, str]]] = {}
    in_adj:  Dict[str, List[Tuple[str, str]]] = {}
    with open(triple_tsv, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            h, r, t = line.rstrip("\n").split("\t")
            out_adj.setdefault(h, []).append((r, t))
            in_adj.setdefault(t, []).append((r, h))
    return KG(ent_meta=ent_meta, rel_meta=rel_meta, out_adj=out_adj, in_adj=in_adj)

from __future__ import annotations
from typing import Dict, Tuple, List

Triple = Tuple[str, str, str]

class Textualizer:
    """
    Convert KG triples into short evidence sentences.
    """
    def __init__(self, ent_meta: Dict[str, Dict], rel_meta: Dict[str, Dict]):
        self.ent = ent_meta
        self.rel = rel_meta

    def _ename(self, e: str) -> str:
        meta = self.ent.get(e, {})
        return meta.get("name", e)

    def _rname(self, r: str) -> str:
        meta = self.rel.get(r, {})
        return meta.get("name", r)

    def triple_to_text(self, tri: Triple) -> str:
        h, r, t = tri
        return f"{self._ename(h)} — {self._rname(r)} — {self._ename(t)}."

    def batch_textualize(self, triples: List[Triple]) -> List[str]:
        return [self.triple_to_text(tri) for tri in triples]

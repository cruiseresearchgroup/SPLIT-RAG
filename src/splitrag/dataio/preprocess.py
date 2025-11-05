from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple, Optional

_WORD = re.compile(r"\w+")

def _default_stopwords() -> set[str]:
    # Minimal list; replace with a full stopword list if desired.
    return {"the","a","an","of","and","to","in","on","for","with","by","at","from","is","are","was","were"}

def mark_entities(text: str, entities: List[Dict[str, Any]]) -> str:
    """
    Insert [E] tags around entity mentions in text (best-effort, surface string match).
    """
    s = text
    # Sort by length desc to avoid partial overlaps.
    ents = sorted(entities, key=lambda e: len(e.get("surface","")), reverse=True)
    for e in ents:
        surf = re.escape(e.get("surface",""))
        s = re.sub(rf"\b{surf}\b", "[E] " + e.get("surface","") + " [/E]", s, flags=re.IGNORECASE)
    return s

def replace_entities_with_types(text: str, entities: List[Dict[str, Any]]) -> str:
    """
    Replace entity mention spans with their KB types, e.g., "Tom Hanks" -> "<Person>".
    """
    s = text
    ents = sorted(entities, key=lambda e: len(e.get("surface","")), reverse=True)
    for e in ents:
        surf = re.escape(e.get("surface",""))
        typ  = e.get("type","Entity")
        s = re.sub(rf"\b{surf}\b", f"<{typ}>", s, flags=re.IGNORECASE)
    return s

def strip_stopwords(text: str, stopwords: Optional[set[str]] = None) -> str:
    stopwords = stopwords or _default_stopwords()
    tokens = _WORD.findall(text)
    kept = [t for t in tokens if t.lower() not in stopwords]
    return " ".join(kept)

def build_q_views(text: str,
                  entities: List[Dict[str, Any]],
                  stopwords: Optional[set[str]] = None) -> Tuple[str, str]:
    """
    Return (Q_s, Q_e) per Sec. 3.1.
    Q_s: stopwords removed + [E] marks
    Q_e: entity mentions replaced by types
    """
    with_marks = mark_entities(text, entities)
    Qs = strip_stopwords(with_marks, stopwords=stopwords)
    Qe = replace_entities_with_types(text, entities)
    return Qs, Qe

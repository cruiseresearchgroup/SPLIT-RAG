from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from .preprocess import build_q_views
from ..utils.io import read_jsonl, read_json
from ..utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class Question:
    qid: str
    text: str
    entities: List[Dict[str, Any]]  # [{"surface": "...", "id": "e123", "type": "Person"}, ...]
    gold_paths: List[List[Tuple[str, str, str]]]  # [[(e_s, r, e_o), ...], ...] optional
    Qs: str = ""   # semantic view: stopwords removed, [E] markers inserted
    Qe: str = ""   # entity-type view: entity mentions replaced by types
    p:  List[List[Tuple[str, str, str]]] | None = None  # stored for convenience

def load_questions(jsonl_path: str | Path,
                   ent_map_path: str | Path,
                   stopwords: Optional[set[str]] = None) -> List[Question]:
    """
    Expected JSONL fields per line:
      { "qid": "...", "text": "...",
        "entities": [{"surface": "...", "id": "e123", "type": "Person"}, ...],
        "gold_paths": [[["e1","r2","e3"], ["e3","r4","e5"]], ...]  # optional
      }
    """
    rows = read_jsonl(jsonl_path)
    ent_map = read_json(ent_map_path)
    questions: List[Question] = []
    for r in rows:
        q = Question(
            qid=r["qid"],
            text=r["text"],
            entities=r.get("entities", []),
            gold_paths=r.get("gold_paths", []),
        )
        Qs, Qe = build_q_views(q.text, q.entities, stopwords=stopwords)
        q.Qs, q.Qe = Qs, Qe
        q.p = q.gold_paths if q.gold_paths else None
        questions.append(q)
    logger.info(f"Loaded {len(questions)} questions from {jsonl_path}")
    return questions

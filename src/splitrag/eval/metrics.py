from __future__ import annotations
from typing import List, Tuple, Dict
import re

_WS = re.compile(r"\s+")
_PUNC = re.compile(r"[^\w\s]")

def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = _PUNC.sub(" ", s)
    s = _WS.sub(" ", s)
    return s.strip()

def exact_match(pred: str, golds: List[str], normalize: bool = True) -> int:
    """
    Returns 1 if pred matches any gold exactly, else 0.
    """
    if pred is None:
        return 0
    if normalize:
        pred = _normalize(pred)
        golds = [_normalize(g) for g in golds]
    return 1 if pred in set(golds) else 0

def token_f1(pred: str, golds: List[str], normalize: bool = True) -> float:
    """
    SQuAD-style token-level F1 against the best gold.
    """
    if pred is None or not golds:
        return 0.0
    if normalize:
        pred = _normalize(pred)
        golds = [_normalize(g) for g in golds]
    p_tokens = pred.split()
    if not p_tokens:
        return 0.0
    def f1_single(g: str) -> float:
        g_tokens = g.split()
        common = {}
        for t in p_tokens:
            if t in g_tokens:
                common[t] = min(p_tokens.count(t), g_tokens.count(t))
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(p_tokens)
        recall = num_same / len(g_tokens)
        return 2 * precision * recall / (precision + recall)
    return max(f1_single(g) for g in golds)

def hit(pred: str, golds: List[str], normalize: bool = True) -> int:
    """
    Hit metric: consider partial match if any gold is a substring of pred or vice versa after normalization.
    """
    if pred is None or not golds:
        return 0
    if normalize:
        pred_n = _normalize(pred)
        golds = [_normalize(g) for g in golds]
    else:
        pred_n = pred
    for g in golds:
        if g in pred_n or pred_n in g:
            return 1
    return 0

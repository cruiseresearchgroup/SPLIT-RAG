from __future__ import annotations
from typing import List

def rough_token_count(s: str) -> int:
    """
    Very light-weight token estimator (~1.3x words).
    This avoids external tokenizers while giving a stable budget guardrail.
    """
    if not s: return 0
    words = s.strip().split()
    return int(len(words) * 1.3)

def truncate_to_tokens(s: str, max_tokens: int) -> str:
    """
    Truncate by words until the rough token estimator fits.
    """
    if rough_token_count(s) <= max_tokens: 
        return s
    words = s.strip().split()
    lo, hi = 0, len(words)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        piece = " ".join(words[:mid])
        if rough_token_count(piece) <= max_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return " ".join(words[:best])

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional

Triple = Tuple[str, str, str]  # (head, relation, tail)

@dataclass
class RuleConfig:
    """
    Relation-level priors used for logical incompatibility checks.
    All sets/dicts are optional; empty means "no constraint."
    """
    functional: Set[str]                    # r(h,·) is functional: same (h,r) cannot yield two different tails
    inverse_functional: Set[str]            # r(·,t) is inverse-functional: same (r,t) cannot come from two different heads
    symmetric: Set[str]                      # symmetric(r): r(h,t) implies r(t,h)  (non-conflict, but useful for dedup)
    antisymmetric: Set[str]                  # antisymmetric(r): r(h,t) & r(t,h) with h≠t is a conflict (e.g., parentOf)
    mutually_exclusive: Dict[str, Set[str]] # r is incompatible with any r' in the set (at same head or same head-tail)
    negates: Dict[str, str]                 # r_neg is negation of r_pos (either direction)

def default_rules() -> RuleConfig:
    # A conservative default: treat common KG edges as non-functional unless
    # dataset-specific priors are supplied later from configs.
    return RuleConfig(
        functional=set(),
        inverse_functional=set(),
        symmetric=set(),
        antisymmetric=set(),
        mutually_exclusive={},   # e.g., {"spouseOf": {"divorcedFrom"}}
        negates={}               # e.g., {"isAlive": "isDead"}
    )

def is_conflict(t1: Triple, t2: Triple, rules: RuleConfig) -> bool:
    """
    Returns True iff t1 and t2 cannot both be true under the rule set.
    Implements Eq. (conflict) proxy: τ1 ⊢ ¬τ2  or  τ2 ⊢ ¬τ1.
    """
    if t1 == t2:
        return False  # identical facts are not in conflict

    h1, r1, o1 = t1
    h2, r2, o2 = t2

    # Functional: (h,r) -> unique tail
    if r1 == r2 and h1 == h2 and o1 != o2 and r1 in rules.functional:
        return True

    # Inverse functional: (r,t) -> unique head
    if r1 == r2 and o1 == o2 and h1 != h2 and r1 in rules.inverse_functional:
        return True

    # Antisymmetric: r(h,t) & r(t,h) with h!=t is a conflict
    if r1 == r2 and r1 in rules.antisymmetric and h1 == o2 and o1 == h2 and h1 != o1:
        return True

    # Mutual exclusion at the same head (broad but practical)
    if r1 in rules.mutually_exclusive and r2 in rules.mutually_exclusive[r1] and h1 == h2:
        return True
    if r2 in rules.mutually_exclusive and r1 in rules.mutually_exclusive[r2] and h1 == h2:
        return True

    # Negations (direction-agnostic variant)
    if r1 in rules.negates and rules.negates[r1] == r2 and h1 == h2 and o1 == o2:
        return True
    if r2 in rules.negates and rules.negates[r2] == r1 and h1 == h2 and o1 == o2:
        return True

    # Otherwise assume compatible
    return False

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Iterable, Optional
import heapq
import math
import numpy as np

# ---- Subgraph vectorization (relation bag-of-words + cosine) ----

def _build_relation_vocab(subgraphs: List["Subgraph"]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for sg in subgraphs:
        for _, r, _ in sg.triples:
            if r not in vocab:
                vocab[r] = len(vocab)
    return vocab

def _sg_vec(sg: "Subgraph", rel_vocab: Dict[str, int]) -> np.ndarray:
    v = np.zeros(len(rel_vocab), dtype=np.float32)
    for _, r, _ in sg.triples:
        idx = rel_vocab[r]
        v[idx] += 1.0
    # l2 normalize
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = (np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0:
        return 0.0
    return float(np.dot(a, b) / den)

def coherence(group_vecs: List[np.ndarray], min_size: int = 1) -> float:
    """
    Coherence(G) = mean cosine(sim(vec, centroid)).
    """
    if len(group_vecs) < min_size:
        return 0.0
    C = np.mean(group_vecs, axis=0)
    sims = [ _cosine(v, C) for v in group_vecs ]
    return float(np.mean(sims)) if sims else 0.0

# ---- Data structures ----

class Subgraph:
    def __init__(self, id: int, triples: List[Tuple[str,str,str]]):
        self.id = id
        self.triples = triples

@dataclass
class GroupingConfig:
    N_max: int = 32                 # capacity per agent
    theta_coh: float = 0.35         # minimum coherence to accept a candidate group

@dataclass
class GroupingResult:
    groups: List[List[int]]         # list of groups, each is list of subgraph ids
    centroid_vecs: List[np.ndarray] # centroid vectors for groups
    sg2group: Dict[int, int]        # subgraph id -> group index

# ---- Greedy allocation algorithm (Alg. 2) ----

def group_subgraphs_question_centric(
    subgraphs: List[Subgraph],
    coverage_sets: List[Set[int]],
    freq_per_sg: List[int],
    cfg: GroupingConfig
) -> GroupingResult:
    """
    Implements Alg. 2 in Sec. 3.2:
      - Priority by coverage density ρ(C_i)
      - Candidate group = C* ∩ Unassigned
      - Trim to N_max by frequency f(Ď_j)
      - Accept if Coherence >= θ_coh
      - Remove covered queries and reheapify
      - Assign residual subgraphs to nearest group (cosine to centroid)
    """
    # Build relation vocabulary & vectors
    rel_vocab = _build_relation_vocab(subgraphs)
    sg_vecs: Dict[int, np.ndarray] = {sg.id: _sg_vec(sg, rel_vocab) for sg in subgraphs}
    n = len(subgraphs)

    # Unassigned pool
    unassigned: Set[int] = set(sg.id for sg in subgraphs)

    # Precompute density per coverage set; use a max-heap (-ρ, idx)
    def _density(Ci: Set[int]) -> float:
        # proxy: sum of pairwise overlaps approximated by |Ci| * mean freq
        if not Ci:
            return 0.0
        mean_freq = np.mean([freq_per_sg[j] for j in Ci])
        return float(len(Ci) * math.sqrt(max(mean_freq, 1e-6)) / math.sqrt(len(Ci)))

    heap: List[Tuple[float, int]] = []
    for i, Ci in enumerate(coverage_sets):
        rho = _density(Ci)
        heapq.heappush(heap, (-rho, i))

    groups: List[List[int]] = []
    centroid_vecs: List[np.ndarray] = []

    while heap and unassigned:
        _neg, i = heapq.heappop(heap)
        Ci = coverage_sets[i]
        if not Ci:
            continue
        # Candidate = intersection with unassigned
        cand = [j for j in Ci if j in unassigned]
        if not cand:
            continue
        # Trim by frequency if exceeds capacity
        if len(cand) > cfg.N_max:
            cand.sort(key=lambda j: freq_per_sg[j], reverse=True)
            cand = cand[:cfg.N_max]
        # Check coherence
        vecs = [sg_vecs[j] for j in cand]
        coh = coherence(vecs, min_size=1)
        if coh >= cfg.theta_coh:
            # Commit group
            groups.append(cand)
            centroid_vecs.append(np.mean(vecs, axis=0))
            for j in cand:
                unassigned.discard(j)
            # Remove overlapping coverage sets: cheap lazy strategy—skip; heap will end
        # else: skip, fallback assignment will handle

    # Fallback: assign residual subgraphs to nearest group by cosine similarity
    # If no group exists yet, create singleton groups greedily by high-frequency items.
    if not groups:
        # seed by highest-frequency subgraphs
        order = sorted(list(unassigned), key=lambda j: freq_per_sg[j], reverse=True)
        while order:
            seed = order.pop(0)
            vec = sg_vecs[seed]
            groups.append([seed])
            centroid_vecs.append(vec)
            unassigned.discard(seed)

    # Fill groups respecting capacity
    while unassigned:
        j = unassigned.pop()
        v = sg_vecs[j]
        # choose best group with available capacity
        best_idx, best_sim = -1, -1.0
        for g_idx, (G, Cvec) in enumerate(zip(groups, centroid_vecs)):
            if len(G) >= cfg.N_max:
                continue
            sim = _cosine(v, Cvec)
            if sim > best_sim:
                best_sim = sim
                best_idx = g_idx
        if best_idx >= 0:
            groups[best_idx].append(j)
            # update centroid
            Gvecs = [sg_vecs[x] for x in groups[best_idx]]
            centroid_vecs[best_idx] = np.mean(Gvecs, axis=0)
        else:
            # all groups full; start a new singleton group
            groups.append([j])
            centroid_vecs.append(v)

    # Build reverse map
    sg2group: Dict[int, int] = {}
    for gid, G in enumerate(groups):
        for j in G:
            sg2group[j] = gid

    return GroupingResult(groups=groups, centroid_vecs=centroid_vecs, sg2group=sg2group)

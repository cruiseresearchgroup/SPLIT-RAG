from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional, Set
import numpy as np
from ..matching.registry import Registry

# ---------- Embedding interface ----------

EmbedFn = Callable[[List[str]], np.ndarray]
# Expected behavior: returns (n_docs x dim) float32/64 matrix, row-normalized is fine.

def _bow_embed(texts: List[str]) -> np.ndarray:
    """
    Lightweight fallback embedder (no external deps):
    character 3-grams hashing → bag-of-ngrams. For reproducibility & speed only.
    """
    dim = 1024
    vecs = np.zeros((len(texts), dim), dtype=np.float32)
    for i, s in enumerate(texts):
        s = s.lower()
        grams = [s[j:j+3] for j in range(max(0, len(s)-2))]
        for g in grams:
            h = hash(g) % dim
            vecs[i, h] += 1.0
    # l2 normalize
    n = np.linalg.norm(vecs, axis=1, keepdims=True)
    n[n == 0] = 1.0
    vecs = vecs / n
    return vecs

def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between single vector a (dim,) and matrix b (n,dim) → (n,)
    """
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (b_norm @ a_norm)

# ---------- Config & API ----------

@dataclass
class SimilarityConfig:
    alpha: float = 0.7     # weight for entity-type cosine
    beta: float = 0.3      # weight for path-overlap
    k_sim: int = 10        # top-k similar training questions to consider
    theta_direct: float = 0.75  # if Sim >= theta_direct, use direct transfer path

class SimilarityScorer:
    """
    Computes Sim(q_new, q_i) = alpha * cos(Emb(Qe_new), Emb(Qe_i)) + beta * Overlap
    Overlap(q_new, q_i) computed via subgraph overlap between seeds of q_new and C_i in registry.
    """
    def __init__(self,
                 cfg: SimilarityConfig,
                 registry: Registry,
                 embed_fn: Optional[EmbedFn] = None):
        self.cfg = cfg
        self.registry = registry
        self.embed_fn = embed_fn or _bow_embed

    def _embed_corpus(self, texts: List[str]) -> np.ndarray:
        return self.embed_fn(texts)

    def rank_similar(
        self,
        Qe_new: str,
        new_seed_sg_ids: Set[int],  # subgraphs touched by path seeds φ(q_new)
        train_questions: List[Dict]
    ) -> List[Tuple[int, float, str, Set[int]]]:
        """
        Returns list of (idx, Sim, qid, C_i) sorted desc by similarity.
        """
        # Prepare corpus of training Q_e
        Qe_list: List[str] = []
        qids: List[str] = []
        cover_sets: List[Set[int]] = []
        for i, q in enumerate(train_questions):
            qid = q.get("qid", f"q{i}")
            meta = self.registry.M.get(qid, {})
            Ci = set(meta.get("coverage", []))
            Qe = q.get("Qe") or q.get("entity_type_context") or q.get("text", "")
            Qe_list.append(Qe)
            qids.append(qid)
            cover_sets.append(Ci)

        # Embeddings
        emb_all = self._embed_corpus(Qe_list)  # (m, d)
        emb_new = self._embed_corpus([Qe_new])[0]  # (d,)

        # Cosine similarities
        cos = _cosine(emb_new, emb_all)  # (m,)

        # Path-overlap via subgraph overlap Jaccard
        overlap = np.zeros_like(cos)
        if new_seed_sg_ids:
            A = set(new_seed_sg_ids)
            for i, Ci in enumerate(cover_sets):
                B = Ci
                if not A and not B:
                    ov = 0.0
                else:
                    inter = len(A & B)
                    union = len(A | B) if (A or B) else 1
                    ov = inter / max(1, union)
                overlap[i] = ov

        sim = self.cfg.alpha * cos + self.cfg.beta * overlap
        order = np.argsort(-sim)[: self.cfg.k_sim]
        ranked = [(int(i), float(sim[i]), qids[i], cover_sets[i]) for i in order]
        return ranked

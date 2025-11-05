from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Any
import json, time
from pathlib import Path

from .base import BaseAgent, AgentResult, RetrievalBudget
from ..planner.decompose import Task
from ..retriever.traverse import enumerate_paths_in_subgraph
from ..retriever.gather import gather_triplets_from_paths, filter_paths_by_match
from ..kg.textualize import Textualizer, Triple
from ..utils.text import truncate_to_tokens

class SubgraphAgent(BaseAgent):
    """
    Processes a Task over the subgraphs assigned to this agent group:
      1) Build subgraph-bounded candidate paths seeded by task entities
      2) Filter candidates by Match(p, t) threshold
      3) Harvest unique triples
      4) Textualize triples to short evidence and optionally summarize
    """
    def __init__(self,
                 group_id: int,
                 budget: RetrievalBudget,
                 index_dir: str | Path,
                 ent_meta: Dict[str, Dict],
                 rel_meta: Dict[str, Dict],
                 summarizer = None  # callable(prompt:str, max_new_tokens:int, temperature:float) -> str
                 ):
        super().__init__(group_id, budget)
        self.index_dir = Path(index_dir)
        self.textualizer = Textualizer(ent_meta, rel_meta)
        self.summarizer = summarizer
        # cache: subgraph_id -> triples
        self._sg_cache: Dict[int, List[Triple]] = {}
        # map: group_id -> [subgraph_ids] loaded from registry.json? Users should instantiate with that info externally.

    # --- helpers ---

    def _load_subgraph_triples(self, sid: int) -> List[Triple]:
        if sid in self._sg_cache:
            return self._sg_cache[sid]
        p = self.index_dir / "subgraphs.jsonl"
        triples: List[Triple] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["id"] == sid:
                    triples = [tuple(t) for t in row["triples"]]
                    break
        self._sg_cache[sid] = triples
        return triples

    def _task_entities(self, task: Task) -> Set[str]:
        ents: Set[str] = set()
        for h, _, t in task.triples:
            ents.add(h); ents.add(t)
        return ents

    def _candidate_paths(self, task: Task) -> List[List[Triple]]:
        """
        Enumerate candidate paths inside each target subgraph, seeded by task entities.
        """
        seeds = list(self._task_entities(task))
        cand: List[List[Triple]] = []
        for sid in task.target_subgraphs:
            tris = self._load_subgraph_triples(sid)
            # ≤2-hop, bounded fanout
            paths = enumerate_paths_in_subgraph(
                subgraph_triples=tris,
                seed_entities=seeds,
                topk_per_seed=self.budget.topk_per_seed,
                max_depth=2
            )
            cand.extend(paths)
        return cand

    def _textualize_evidence(self, triples: List[Triple]) -> str:
        """
        Turn harvested triples into a compact evidence paragraph.
        Uses optional summarizer; otherwise returns concatenated sentences capped by tokens.
        """
        sents = self.textualizer.batch_textualize(triples)
        if not sents:
            return ""
        raw = " ".join(sents)
        if self.summarizer is None:
            # Simple cap by token budget
            return truncate_to_tokens(raw, self.budget.evidence_cap_tokens)
        # Otherwise, ask the summarizer to compress to salient evidence.
        prompt = (
            "Summarize the following KG facts into a concise, faithful evidence paragraph. "
            "Do not introduce external facts; keep entity/relation names intact.\n\n"
            + raw
        )
        summary = self.summarizer(prompt, max_new_tokens=256, temperature=0.3)
        return truncate_to_tokens(summary, self.budget.evidence_cap_tokens)

    # --- main ---

    def run(self, task: Task) -> AgentResult:
        t0 = time.time()

        # 1) generate candidates
        candidates = self._candidate_paths(task)

        # 2) filter by Match(p, t)
        task_ents = self._task_entities(task)
        kept = filter_paths_by_match(
            candidates, 
            task_entities=task_ents, 
            theta_match=self.budget.theta_match
        )

        # 3) harvest unique triples
        triples = gather_triplets_from_paths(kept)

        # 4) textualize → evidence
        evidence = self._textualize_evidence(triples)

        dt = time.time() - t0
        return AgentResult(
            group_id=self.group_id,
            triplets=triples,
            evidence_text=evidence,
            latency_s=dt
        )

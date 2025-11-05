from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time
from pathlib import Path

from .base import AgentResult
from ..conflict.resolve import resolve_conflicts, ResolveConfig
from ..conflict.rules import RuleConfig, default_rules

Triple = Tuple[str, str, str]

@dataclass
class HeadBudget:
    max_input_tokens: int = 3200
    max_new_tokens: int = 256
    temperature: float = 0.3
    top_p: float = 0.95
    stop: Optional[List[str]] = None

@dataclass
class HeadOutput:
    answer: str
    T_clean: List[Triple]
    evidence_text: str
    latency_s: float

class HeadAgent:
    """
    Head agent: aggregates multi-agent outputs, resolves conflicts, and synthesizes the final answer.
    """
    def __init__(self,
                 llm_generate,              # callable(prompt,max_new_tokens,temperature,top_p,stop) -> str
                 budget: HeadBudget = HeadBudget(),
                 rules: RuleConfig | None = None):
        self.llm_generate = llm_generate
        self.budget = budget
        self.rules = rules or default_rules()

    # ---------- aggregation ----------

    @staticmethod
    def aggregate(results: List[AgentResult]) -> Tuple[List[Triple], str, Dict[Triple, List[int]]]:
        """
        Combine per-agent (TRI_i, ET_i). Returns:
          - union of triples (dedup, keep first-seen order)
          - concatenated evidence text (simple concat; caller can re-truncate)
          - support map: triple -> [group_ids] that reported it
        """
        seen: set[Triple] = set()
        tri_all: List[Triple] = []
        support: Dict[Triple, List[int]] = {}
        evidence_chunks: List[str] = []

        for r in results:
            evidence_chunks.append(r.evidence_text)
            for t in r.triplets:
                if t not in seen:
                    seen.add(t)
                    tri_all.append(t)
                support.setdefault(t, []).append(r.group_id)

        evidence_text = " ".join([e for e in evidence_chunks if e])
        return tri_all, evidence_text, support

    # ---------- conflict resolution + synthesis ----------

    def answer(self,
               q_text: str,
               results: List[AgentResult],
               agent_confidence: Dict[int, float] | None = None,
               resolve_cfg: ResolveConfig = ResolveConfig()) -> HeadOutput:
        """
        Full pipeline:
          1) Aggregate per-agent outputs
          2) Resolve conflicts via compatibility-graph clique
          3) Synthesize final answer using verified triples and evidence
        """
        t0 = time.time()

        triples_all, evidence_all, support = self.aggregate(results)
        res = resolve_conflicts(
            triples_all,
            who_supports=support,
            agent_confidence=agent_confidence,
            rules=self.rules,
            cfg=resolve_cfg
        )
        T_clean = res.T_clean

        # Build compact, grounded prompt
        facts = "\n".join([f"- ({h}, {r}, {t})" for (h, r, t) in T_clean])
        prompt = (
            "You are given a question, a set of verified KG facts (triples), and concise evidence. "
            "Answer the question strictly based on the verified facts. "
            "If insufficient information exists, say 'Insufficient evidence'. "
            "Return a short, direct answer followed by one brief justification.\n\n"
            f"Question:\n{q_text}\n\n"
            f"Verified KG facts:\n{facts}\n\n"
            "Evidence (optional, may contain paraphrases of the facts):\n"
            f"{evidence_all}\n\n"
            "Answer:"
        )

        # Token-budgeting is handled by the upstream generator; here just call it.
        ans = self.llm_generate(
            prompt=prompt,
            max_new_tokens=self.budget.max_new_tokens,
            temperature=self.budget.temperature,
            top_p=self.budget.top_p,
            stop=self.budget.stop or ["\n\n"]
        )

        dt = time.time() - t0
        return HeadOutput(answer=ans, T_clean=T_clean, evidence_text=evidence_all, latency_s=dt)

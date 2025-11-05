from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Iterable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

Triple = Tuple[str, str, str]

@dataclass
class RetrievalBudget:
    theta_match: float = 0.5
    topk_per_seed: int = 5
    evidence_cap_tokens: int = 2600  # per dataset; MetaQA=2600, WebQSP/CWQ=3200

class AgentResult:
    """
    Container for per-agent outputs.
    """
    def __init__(self, group_id: int, triplets: List[Triple], evidence_text: str, latency_s: float):
        self.group_id = group_id
        self.triplets = triplets
        self.evidence_text = evidence_text
        self.latency_s = latency_s

class BaseAgent:
    """
    Base interface for agents that process a decomposed task and return (TRI, E).
    """
    def __init__(self, group_id: int, budget: RetrievalBudget):
        self.group_id = group_id
        self.budget = budget

    def run(self, task: Any) -> AgentResult:
        raise NotImplementedError

class MultiAgentExecutor:
    """
    Parallel runner for a set of agents. One agent per selected group id.
    """
    def __init__(self, agents: Dict[int, BaseAgent], max_workers: int = 8):
        self.agents = agents
        self.max_workers = max_workers

    def run_tasks(self, tasks: List[Any]) -> List[AgentResult]:
        """
        Dispatch each task to its responsible agent by group id.
        """
        futs = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for t in tasks:
                ag = self.agents.get(t.agent_group)
                if not ag:
                    continue
                futs.append(ex.submit(ag.run, t))
            results: List[AgentResult] = []
            for f in as_completed(futs):
                results.append(f.result())
        return results

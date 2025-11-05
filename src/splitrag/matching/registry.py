from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Tuple, Any
from pathlib import Path
import json
import numpy as np

from .association import load_subgraphs, compute_association_matrix, frequency_per_subgraph
from .grouping import group_subgraphs_question_centric, GroupingConfig, Subgraph as SG2

@dataclass
class RegistryConfig:
    index_dir: Path
    q_train_jsonl: Path
    N_max: int = 32
    theta_coh: float = 0.35

@dataclass
class AgentGroup:
    id: int
    subgraph_ids: List[int]

@dataclass
class Registry:
    """
    Persistent registry 𝓡 = {𝓐, Ď, 𝓜} used by the planner (Sec. 3.3).
    - A: list of agent groups, each is a set of subgraph ids
    - D: subgraphs (their ids; detailed triples live in index_dir/subgraphs.jsonl)
    - M: training-time mappings (here we store coverage sets per question as patterns)
    """
    index_dir: str
    A: List[AgentGroup]
    D_ids: List[int]
    M: Dict[str, Dict[str, Any]]  # qid -> {"coverage": [sg_ids], "touch_count": int}

def _load_questions(jsonl_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def build_registry(cfg: RegistryConfig) -> Registry:
    """
    End-to-end:
      1) Load Ď from index
      2) Compute A_ij and coverage sets C_i on Q_train
      3) Group Ď into agent groups with capacity & coherence constraints
      4) Persist 𝓡 = {𝓐, Ď_ids, 𝓜}
    """
    # 1. Load subgraphs
    sgs1 = load_subgraphs(cfg.index_dir)
    subgraphs = [SG2(id=sg.id, triples=sg.triples) for sg in sgs1]
    D_ids = [sg.id for sg in sgs1]

    # 2. Association & coverage
    Q = _load_questions(cfg.q_train_jsonl)
    A, coverage_sets, _ = compute_association_matrix(Q, sgs1)
    freq = frequency_per_subgraph(coverage_sets, n_subgraphs=len(subgraphs))

    # 3. Grouping
    gcfg = GroupingConfig(N_max=cfg.N_max, theta_coh=cfg.theta_coh)
    res = group_subgraphs_question_centric(subgraphs, coverage_sets, freq_per_sg=freq, cfg=gcfg)

    # 4. Build 𝓜 patterns (record minimal info now; planner will extend)
    M: Dict[str, Dict[str, Any]] = {}
    for i, q in enumerate(Q):
        qid = q.get("qid", f"q{i}")
        cov = sorted(list(coverage_sets[i]))
        touches = len(set(res.sg2group[j] for j in cov)) if cov else 0
        M[qid] = {"coverage": cov, "touch_count": touches}

    # 5. Serialize registry (JSON)
    A_groups = [AgentGroup(id=i, subgraph_ids=sorted(G)) for i, G in enumerate(res.groups)]
    reg = Registry(
        index_dir=str(cfg.index_dir),
        A=A_groups,
        D_ids=D_ids,
        M=M
    )
    return reg

def save_registry(reg: Registry, out_path: str | Path) -> None:
    """
    Persist registry as JSON (lists for dataclasses).
    """
    obj = {
        "index_dir": reg.index_dir,
        "A": [{"id": g.id, "subgraph_ids": g.subgraph_ids} for g in reg.A],
        "D_ids": reg.D_ids,
        "M": reg.M,
    }
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_registry(path: str | Path) -> Registry:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    A_groups = [AgentGroup(**g) for g in obj["A"]]
    return Registry(
        index_dir=obj["index_dir"],
        A=A_groups,
        D_ids=obj["D_ids"],
        M=obj["M"]
    )

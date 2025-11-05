from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json, time, yaml, os
from ..utils.io import read_jsonl, write_jsonl, ensure_dir
from ..utils.logging import get_logger, timer
from ..kg.graph import load_kg
from ..dataio.preprocess import build_q_views
from ..planner.path_seed import seed_by_paths, PathSeedConfig
from ..planner.decompose import build_agent_plan
from ..planner.similarity import SimilarityConfig
from ..planner.select import SelectConfig
from ..agents.base import RetrievalBudget, MultiAgentExecutor
from ..agents.subgraph_agent import SubgraphAgent
from ..agents.head_agent import HeadAgent, HeadBudget
from ..eval.metrics import exact_match, token_f1, hit
from ..matching.registry import load_registry
from splitrag.llm.vllm_client import VLLMProvider, VLLMConfig
from splitrag.llm.hf_local import HFLocalProvider, HFLocalConfig


logger = get_logger(__name__)

def _load_yaml(p: str | Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@dataclass
class EvalResult:
    rows: List[Dict[str, Any]]
    macro: Dict[str, float]

def _enumerate_candidate_paths_global(kg,
                                      entities: List[str],
                                      topk_per_seed: int,
                                      max_paths_per_question: int):
    """
    Fallback when question JSON lacks 'paths'/'gold_paths': enumerate ≤2-hop paths globally from seed entities.
    """
    paths = kg.enumerate_paths_le2(entities, topk_per_seed=topk_per_seed)
    return paths[: max_paths_per_question]

def _dummy_summarizer(prompt: str, max_new_tokens: int, temperature: float) -> str:
    # Cheap compress: truncate prompt tail (evidence is already templated).
    return prompt.split("\n\n")[-1][: 8000]

def _dummy_generate(prompt: str, max_new_tokens: int, temperature: float, top_p: float, stop=None) -> str:
    # Deterministic baseline for offline testing; replace with real provider in production.
    # Heuristic: pick entity after the last '(' as an "answer".
    lines = [ln for ln in prompt.splitlines() if ln.strip().startswith("- (")]
    if lines:
        # pick tail of first fact
        first = lines[0]
        try:
            obj = first.split(",")[-1].strip(" )")
            return obj
        except Exception:
            return "Insufficient evidence"
    return "Insufficient evidence"

def evaluate_dataset(cfg_path: str | Path,
                     split: str = "test",
                     llm_generate=_dummy_generate) -> EvalResult:
    """
    End-to-end evaluation for a dataset described by cfg_path (YAML).
    Produces per-question JSONL rows and returns macro metrics.
    """
    cfg = _load_yaml(cfg_path)
    P = cfg["paths"]
    kg = load_kg(P["kg_tsv"], P["ent_json"], P["rel_json"])
    registry = load_registry(P["registry_json"])

    # Planner configs
    sim_cfg = SimilarityConfig(**cfg["planner"]["similarity"])
    sel_cfg = SelectConfig(**cfg["planner"]["select"])
    pseed_cfg = PathSeedConfig(**cfg["planner"]["path_seed"])

    # Agents
    sga_cfg = cfg["subgraph_agent"]
    sga_budget = RetrievalBudget(
        theta_match=sga_cfg["theta_match"],
        topk_per_seed=sga_cfg["topk_per_seed"],
        evidence_cap_tokens=sga_cfg["evidence_cap_tokens"]
    )
    max_workers = sga_cfg.get("max_workers", 8)

    # Head
    head_cfg = cfg["head"]
    head_budget = HeadBudget(
        max_input_tokens=head_cfg["max_input_tokens"],
        max_new_tokens=head_cfg["max_new_tokens"],
        temperature=head_cfg["temperature"],
        top_p=head_cfg["top_p"],
        stop=head_cfg.get("stop")
    )
    # provider = VLLMProvider(VLLMConfig(base_url="http://127.0.0.1:8000/v1",
    #                                api_key="xxx",
    #                                model="meta-llama/Meta-Llama-3.1-8B-Instruct"))
    # === NEW: local Falcon provider ===
    provider = HFLocalProvider(HFLocalConfig(
        model_id="tiiuae/Falcon-7B-Instruct",
        device="cuda",
        dtype="bfloat16",
        max_input_tokens=2048,
        max_concurrency=1
    ))


    head = HeadAgent(llm_generate=provider.generate, budget=head_budget)

    # Load questions
    q_path = P[f"{split}_q"]
    rows = read_jsonl(q_path)
    out_rows: List[Dict[str, Any]] = []

    run_dir = ensure_dir(Path(P["runs_dir"]) / time.strftime("run_%Y%m%d_%H%M%S"))
    pred_out = run_dir / f"pred_{split}.jsonl"

    # Eval accumulators
    H, H1, F1, N = 0.0, 0.0, 0.0, 0

    for r in rows:
        N += 1
        qid = r.get("qid", f"{split}-{N}")
        text = r["text"]
        entities = [e["id"] for e in r.get("entities", []) if "id" in e]

        # Ensure Q_s/Q_e views
        Qs, Qe = build_q_views(text, r.get("entities", []))
        r["Qs"], r["Qe"] = Qs, Qe

        # Candidate paths (either provided or enumerated globally)
        raw_paths = r.get("paths") or r.get("gold_paths")
        if not raw_paths:
            raw_paths = _enumerate_candidate_paths_global(
                kg, entities,
                topk_per_seed=cfg["indexer"]["topk_per_seed"],
                max_paths_per_question=cfg["indexer"]["max_paths_per_question"]
            )
        else:
            raw_paths = [[tuple(t) for t in p] for p in raw_paths]

        # Seed by paths
        seed = seed_by_paths(raw_paths, index_dir=P["index_dir"], cfg=pseed_cfg)

        # Build plan
        # Training corpus for similarity ranking = training split rows
        train_rows = read_jsonl(P["train_q"])
        plan = build_agent_plan(
            registry=registry,
            q_new={"Qe": Qe},
            train_questions=train_rows,
            sim_cfg=sim_cfg,
            sel_cfg=sel_cfg,
            seed=seed
        )

        # Instantiate agents for selected groups
        agents = {}
        for gid in plan.groups:
            agents[gid] = SubgraphAgent(
                group_id=gid,
                budget=sga_budget,
                index_dir=P["index_dir"],
                ent_meta=kg.ent_meta,
                rel_meta=kg.rel_meta,
                summarizer=provider.summarize
            )

        # Execute tasks in parallel
        exec = MultiAgentExecutor(agents, max_workers=max_workers)
        t0 = time.time()
        agent_results = exec.run_tasks(plan.tasks)
        t_agents = time.time() - t0

        # Head synthesis (includes conflict resolution)
        t0 = time.time()
        head_out = head.answer(q_text=text, results=agent_results)
        t_head = time.time() - t0

        # Metrics
        golds = r.get("answers", []) or r.get("answer_aliases", [])
        pred = head_out.answer
        em = exact_match(pred, golds, normalize=cfg["eval"]["normalize_answer"])
        f1 = token_f1(pred, golds, normalize=cfg["eval"]["normalize_answer"])
        hh = hit(pred, golds, normalize=cfg["eval"]["normalize_answer"])
        H1 += em; F1 += f1; H += hh

        out = {
            "qid": qid,
            "text": text,
            "pred": pred,
            "gold": golds,
            "hit": hh,
            "h@1": em,
            "f1": f1,
            "latency_agents_s": t_agents,
            "latency_head_s": t_head,
            "latency_e2e_s": t_agents + t_head,
            "provenance": plan.provenance,
            "groups": plan.groups,
            "num_tasks": len(plan.tasks),
            "num_triples_clean": len(head_out.T_clean)
        }
        out_rows.append(out)

    # Macro
    macro = {
        "Hit": round(H / max(1,N), 4),
        "H@1": round(H1 / max(1,N), 4),
        "F1": round(F1 / max(1,N), 4),
        "N": N
    }
    # Save predictions
    write_jsonl(pred_out, out_rows)
    with open(run_dir / "macro.json", "w", encoding="utf-8") as f:
        json.dump(macro, f, indent=2)
    logger.info(f"Saved run to {run_dir}")
    logger.info(f"Macro: {macro}")
    return EvalResult(rows=out_rows, macro=macro)

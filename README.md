# SPLIT-RAG: Divide by Question, Conquer by Agent

**Paper:** *SPLIT-RAG: Divide by Question, Conquer by Agent*  
**ArXiv (PDF):** <https://arxiv.org/pdf/2505.13994?>  
**Benchmarks:** WebQSP, CWQ, MetaQA-2Hop, MetaQA-3Hop  
**Summary:** SPLIT-RAG partitions a knowledge graph (KG) with question-type signals, assigns specialized agent groups to subgraphs, retrieves facts in parallel, resolves conflicts with a compatibility graph + max-weight clique heuristic, and synthesizes a final answer with a head agent.

---

## ✨ Highlights

- **Question-centric planning (§3):** similar-question transfer or path-driven seeding decides which agent groups to use under budget \(B\).  
- **Subgraph-bounded retrieval (§3.4):** ≤2-hop traversal inside selected subgraphs; triplet harvesting and compact evidence generation.  
- **Conflict resolution (§3.5):** rule-aware compatibility graph with a max-weight clique heuristic returns a consistent triple set.  
- **Head agent synthesis:** answers strictly from verified triples + evidence.  
- **Latency & flexibility (§4.4):** lighter models can serve subgraph agents, while a stronger model serves the head agent.

---

## 🗂️ Repository Structure

```
SPLIT-RAG/
├─ configs/
│ ├─ metaqa.yaml
│ ├─ webqsp.yaml
│ └─ cwq.yaml
├─ scripts/
│ ├─ run_metaqa.sh
│ ├─ run_webqsp.sh
│ └─ run_cwq.sh
├─ src/splitrag/
│ ├─ cli/ # build_index, match_agents, eval entry points
│ ├─ dataio/ # loading & preprocessing
│ ├─ eval/ # metrics + runner (Hit, H@1, F1, latency)
│ ├─ indexer/ # KG partitioning (§3.1)
│ ├─ kg/ # KG IO + textualization
│ ├─ llm/
│ │ ├─ hf_local.py # local HuggingFace provider (Falcon/Llama)
│ │ ├─ vllm_client.py # optional: vLLM OpenAI-compatible client
│ │ └─ openai_compat.py# optional: OpenAI-compatible hosted providers
│ ├─ matching/ # build agent registry 𝓡 (§3.2)
│ ├─ planner/ # similarity, seeding, selection, decomposition (§3.3)
│ ├─ retriever/ # traversal, match, gather (§3.4)
│ ├─ agents/ # SubgraphAgent, HeadAgent, executor (§3.4–3.5)
│ ├─ conflict/ # rules + graph + clique (§3.5)
│ └─ utils/ # IO, logging, token helpers
└─ README.md

---
```

## 📚 Datasets & File Layout

Expected files under `data/processed/`:

```
data/processed/
├─ kg.tsv # head<TAB>relation<TAB>tail
├─ entities.json # {entity_id: {"name": "...", ...}, ...}
├─ relations.json # {relation_id: {"name": "...", ...}, ...}
├─ index/<dataset>/ # subgraphs.jsonl + artifacts (built)
├─ registry/<dataset>.json# agent registry (built)
├─ runs/<dataset>/... # predictions + macro metrics (produced)
├─ <dataset>_train.jsonl # question JSONL (format below)
├─ <dataset>_dev.jsonl
└─ <dataset>_test.jsonl
```

**Question JSONL format** (one JSON per line):
```json
{
  "qid": "webqsp-0001",
  "text": "Who directed the film that starred Tom Hanks?",
  "entities": [{"id": "m.Tom_Hanks"}, {"id": "m.Forrest_Gump"}],
  "answers": ["Robert Zemeckis", "Zemeckis"],
  "paths": [
    [["m.Tom_Hanks","acted_in","m.Forrest_Gump"],
     ["m.Forrest_Gump","directed_by","m.Robert_Zemeckis"]]
  ]
}
```

## 🚀 Quickstart

```bash
# 1) Build partitioned KG index (§3.1)
python -m splitrag.cli.build_index --cfg configs/metaqa.yaml

# 2) Build agent registry 𝓡 (§3.2)
python -m splitrag.cli.match_agents --cfg configs/metaqa.yaml

# 3) Evaluate (planning → multi-agent retrieval → conflict resolution → head synthesis)
python -m splitrag.cli.eval_all --cfg configs/metaqa.yaml --split test

```


## 🔐 License

This project is licensed under the terms of the MIT license.


## Citation

```bibtex
@misc{yang2025splitrag,
  title         = {SPLIT-RAG: Divide by Question, Conquer by Agent},
  author        = {Yang, Ruiyi and others},
  year          = {2025},
  eprint        = {2505.13994},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/pdf/2505.13994?}
}
```
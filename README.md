# Textonic — Climate Hallucination Analysis + RAG/KG/Hybrid Implementations

Textonic is a research codebase for studying **hallucination behavior in LLMs** on **climate-related prompts**, and experimenting with multiple grounding strategies:

- **INLP baselines + instructed generation** (Qwen, LLaMA) + evaluation
- **RAG implementation** (PDF ingestion → retrieval → reranking → LLM)
- **KG implementation** (entity/relation extraction + Neo4j/Cypher utilities)
- **Hybrid implementation** (KG + RAG + reranking, multiple retrieval modes)

---

## Repository Structure (high-level)

```text
.
├── INLP_Project/
│   ├── LLM_baseline.py
│   ├── finetune_llm.py
│   ├── Metric_Evaluation.ipynb
│   └── DATASET/
│       ├── climate_hallucination_dataset.json
│       ├── Llama_baseline_dataset.json
│       ├── Llama_instructed_dataset.json
│       ├── Qwen_baseline_dataset.json
│       └── Qwen_instructed_dataset.json
├── METRICS/
│   └── V2_INLP_Project_metrics_all.ipynb
├── RAG_IMPLEMENTATION/
│   ├── main.py
│   ├── requirements.txt
│   └── src/
│       ├── llm.py
│       ├── pdf_processing.py
│       ├── retrieval.py
│       └── rerank.py
├── KG_IMPLEMENTATION/
│   ├── GRAPH/
│   └── KG_Implementation/
│       ├── llm_label.py
│       ├── Data_preparation/
│       └── Entity_relation/
├── HYBRID_IMPLEMENTATION/
│   ├── requirements.txt
│   ├── run_llama_modes_first.sh
│   ├── eval/run_full_dataset.py
│   └── src/
│       ├── configs/config.py
│       └── hybrid/pipeline.py
└── FINETUNE_MODEL_IMPLEMENTATION/
    └── Fine_Tuning.ipynb
```

---

## INLP_Project: Baselines, Instruction Tuning, and Evaluation

- Prompt dataset: `INLP_Project/DATASET/climate_hallucination_dataset.json`
- Generate baseline outputs:

```bash
python INLP_Project/LLM_baseline.py
```
- Generate instructed outputs:

```bash
python INLP_Project/finetune_llm.py
```
- Evaluate:
  - `INLP_Project/Metric_Evaluation.ipynb`
  - `METRICS/V2_INLP_Project_metrics_all.ipynb`

---

## RAG_IMPLEMENTATION (RAG)

```bash
cd RAG_IMPLEMENTATION
pip install -r requirements.txt
python main.py
```

---

## KG_IMPLEMENTATION (Knowledge Graph)

- Labeling: `KG_IMPLEMENTATION/KG_Implementation/llm_label.py`
- Entity/relation extraction: `KG_IMPLEMENTATION/KG_Implementation/Entity_relation/`
- Graph pipeline utilities (Neo4j/Cypher): `KG_IMPLEMENTATION/GRAPH/`

---

## HYBRID_IMPLEMENTATION (KG + RAG Hybrid)

Hybrid combines KG + text retrieval (RAG) + reranking, and supports retrieval modes:
- `seq_kg_first`
- `seq_rag_first`
- `parallel`

### Run (minimal)

```bash
cd HYBRID_IMPLEMENTATION

# setup
conda create -n hybrid_qa python=3.10 -y
conda activate hybrid_qa
pip install -r requirements.txt

# required env
export HF_TOKEN="your_huggingface_token"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
export HYBRID_CACHE_ROOT=/tmp/hybrid_cache

# full dataset
python eval/run_full_dataset.py \
  --input climate_hallucination_dataset_test_230_v2.json \
  --model qwen \
  --retrieval-mode parallel \
  --use-gpu \
  --output eval/hybrid_qwen_parallel_full_dataset_answers.json

# batch script (multiple modes)
bash run_llama_modes_first.sh
```

Notes:
- First run may take longer due to model downloads and index/cache creation.
- If disk quota is tight, add `--allow-tmp-fallback`.

---

## Fine-tuning

- Notebook: `FINETUNE_MODEL_IMPLEMENTATION/Fine_Tuning.ipynb`
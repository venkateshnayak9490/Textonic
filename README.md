# Textonic — Climate Hallucination Analysis + RAG/KG Experiments

Textonic is a research project focused on **hallucination behavior in Large Language Models (LLMs)** for **climate-related prompts**.  
It includes:

- Baseline vs instruction-tuned response generation (**Qwen** and **LLaMA**) — in `INLP_Project/`
- Metrics and evaluation notebooks — in `INLP_Project/` and `METRICS/`
- Additional implementations/experiments:
  - **RAG pipeline** — `RAG_IMPLEMENTATION/`
  - **Knowledge Graph (KG) labeling / processing** — `KG_IMPLEMENTATION/`
  - **Fine-tuning notebook** — `FINETUNE_MODEL_IMPLEMENTATION/`

---

## Repository Structure

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
└── FINETUNE_MODEL_IMPLEMENTATION/
    └── Fine_Tuning.ipynb
```

---

## INLP_Project: Baselines, Instruction Tuning, and Evaluation

### Dataset (ground truth prompts)
Main prompt dataset:

- `INLP_Project/DATASET/climate_hallucination_dataset.json`

### Generate baseline responses
Runs baseline generation for **Qwen** and **LLaMA**:

```bash
python INLP_Project/LLM_baseline.py
```

Outputs (stored in `INLP_Project/DATASET/`):

- `Llama_baseline_dataset.json`
- `Qwen_baseline_dataset.json`

### Generate instruction-tuned / prompted responses
```bash
python INLP_Project/finetune_llm.py
```

Outputs:

- `Llama_instructed_dataset.json`
- `Qwen_instructed_dataset.json`

### Evaluate results
Open and run:

- `INLP_Project/Metric_Evaluation.ipynb`

Additional metrics notebook:

- `METRICS/V2_INLP_Project_metrics_all.ipynb`

---

## RAG_IMPLEMENTATION: Retrieval-Augmented Generation (RAG)

This folder contains a RAG pipeline with:

- `main.py` — entry point
- `src/pdf_processing.py` — PDF ingestion / text extraction
- `src/retrieval.py` — retrieval logic
- `src/rerank.py` — reranking step
- `src/llm.py` — LLM calling wrapper

### Setup & run (RAG)
```bash
cd RAG_IMPLEMENTATION
pip install -r requirements.txt
python main.py
```

---

## KG_IMPLEMENTATION: Knowledge Graph work

This module includes KG-related processing and labeling utilities.

Key script:

- `KG_IMPLEMENTATION/KG_Implementation/llm_label.py` — labeling logic (LLM-assisted)

Supporting folders:

- `KG_IMPLEMENTATION/KG_Implementation/Data_preparation/`
- `KG_IMPLEMENTATION/KG_Implementation/Entity_relation/`
- `KG_IMPLEMENTATION/GRAPH/` — graph artifacts

---

## Fine-tuning Notebook

- `FINETUNE_MODEL_IMPLEMENTATION/Fine_Tuning.ipynb` — notebook for fine-tuning experiments

---

## Tech Stack

- Python
- Jupyter Notebooks
- JSON datasets
- LLMs (Qwen, LLaMA)
- RAG components (retrieval + reranking + LLM wrapper)
- KG utilities (labeling + preparation)

---

## Notes

- Some scripts/notebooks may require API keys (for labeling / LLM calls).  
  If you want, tell me which provider you used (Gemini/OpenAI/HF/etc.) and I’ll add an exact `.env` section + environment variable names.

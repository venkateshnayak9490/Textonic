# Repository Structure

```text
.
├── .gitignore
├── README.md
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
│   │   ├── cypher_generator.py
│   │   ├── evaluation.py
│   │   ├── llm_handler.py
│   │   ├── neo4j_client.py
│   │   ├── nim_client.py
│   │   ├── pipeline.py
│   │   ├── qwen_client.py
│   │   ├── schema_discovery.py
│   │   └── test.py
│   └── KG_Implementation/
│       ├── llm_label.py
│       ├── Data_preparation/
│       │   ├── chunker.py
│       │   ├── config.py
│       │   ├── content_scorer.py
│       │   ├── pdf_extractor.py
│       │   ├── pipeline.py
│       │   ├── text_filter.py
│       │   └── utils.py
│       └── Entity_relation/
│           ├── entity_extractor.py
│           └── relation_extractor.py
├── HYBRID_IMPLEMENTATION/
│   ├── AUDIT_REPORT.md
│   ├── DEPENDENCIES_INSTALLED.md
│   ├── MULTI_GPU_SETUP.md
│   ├── requirements.txt
│   ├── run_llama_modes_first.sh
│   ├── KG/
│   ├── data/
│   │   └── text_chunks/
│   │       └── chunks.json
│   ├── eval/
│   │   └── run_full_dataset.py
│   └── src/
│       ├── configs/
│       │   ├── __init__.py
│       │   └── config.py
│       └── hybrid/
│           ├── __init__.py
│           ├── kg_retriever.py
│           ├── pipeline.py
│           └── text_retriever.py
└── FINETUNE_MODEL_IMPLEMENTATION/
    └── Fine_Tuning.ipynb
```


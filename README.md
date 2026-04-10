# Project Overview

This project explores hallucination behavior in Large Language Models (LLMs) for climate-related queries. The goal is to analyze how LLMs generate responses and evaluate the difference between baseline model outputs and instruction-based (fine-tuned) outputs.

We use two open-source LLMs as baseline models:

1) Qwen

2) LLaMA

Responses from these models are generated using a climate-related dataset and evaluated using multiple metrics to study hallucination patterns in LLM outputs.

These models were used to generate answers for the climate-related prompts in the dataset.

For **hallucination label generation**, we used the **Gemini API** to automatically label whether the generated responses contained hallucinated information.

## Dataset Description
climate_hallucination_dataset.json

This is the main dataset used in the project.

It contains climate-related prompts designed to analyze hallucination behavior in language model responses.

## Methodology
1. Clone the Repository

git clone https://github.com/Jy0810/TexTonic_INLP.git

cd TexTonic_INLP/INLP_Project

4. Baseline Model Generation
   
We first generate responses using baseline LLMs (Qwen and LLaMA) without additional instruction tuning.

Run the script:

python LLM_baseline.py

This script generates the following datasets:

Llama_baseline_dataset.json

Qwen_baseline_dataset.json

7. Instruction-Based / Fine-tuned Generation
   
Next, we generate responses using instruction-based prompting to observe improvements in response quality and hallucination reduction.

Run the script:

python finetune_llm.py

This generates:

Llama_instructed_dataset.json

Qwen_instructed_dataset.json

9. Evaluation

All generated responses are evaluated using multiple metrics to compare:

Baseline outputs

Instructed outputs

Model performance differences

Evaluation is performed using the notebook:

Metric_Evaluation.ipynb

This notebook computes evaluation metrics and analyzes hallucination behavior across the different model outputs.

## Technologies Used

Python

Large Language Models (LLMs)

Qwen

LLaMA

JSON datasets

Jupyter Notebook

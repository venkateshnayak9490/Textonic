import json
import os
from tqdm import tqdm
from src.pdf_processing import load_and_chunk_pdfs
from src.retrieval import build_faiss_index, retrieve_top_k
from src.rerank import rerank
from src.llm import load_model, generate_answer

import argparse

# ================= CONFIG =================
PDF_FOLDER = "filtered"
INPUT_JSON = "climate_hallucination_dataset_test_230_v2.json"
USE_RERANK = True 
HF_TOKEN = "hf_xxx"  # [PLACEHOLDER] Paste your Hugging Face token here

# Default models
MODELS = {
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen": "Qwen/Qwen2.5-3B-Instruct"
}

# ==========================================

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama", choices=["llama", "qwen"], help="Model to use")
    args = parser.parse_args()

    model_id = MODELS[args.model]
    output_json = f"outputs/{args.model}_results.json"
    
    print(f"Using model: {model_id}")
    print("Loading dataset...")
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    print("Processing PDFs...")
    chunks = load_and_chunk_pdfs(PDF_FOLDER)

    print("Building FAISS index...")
    index, embedding_model = build_faiss_index(chunks)

    print("Loading LLM...")
    tokenizer, model = load_model(model_id, token=HF_TOKEN)

    results = []

    for item in tqdm(data):
        query = item["question"]

        retrieved = retrieve_top_k(query, index, chunks, embedding_model, k=20)

        if USE_RERANK:
            top_chunks = rerank(query, retrieved, top_k=3)
        else:
            top_chunks = retrieved[:3]

        answer = generate_answer(query, top_chunks, tokenizer, model)

        item["llm_answer"] = answer
        item["retrieved_chunks"] = [c["text"] for c in top_chunks]

        results.append(item)

    os.makedirs("outputs", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print("Done. Output saved to", output_json)

if __name__ == "__main__":
    run()

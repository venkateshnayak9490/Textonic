import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# change to:
# "meta-llama/Meta-Llama-3-8B-Instruct"
# "google/gemma-7b-it"
# "Qwen/Qwen2.5-7B-Instruct"

INPUT_FILE = "climate_hallucination_dataset_FV.json"
OUTPUT_FILE = "llama_3.2_1b_finetune.json"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return tokenizer, model


def build_prompt(question, topic, qtype):

    if qtype == "factual":
        limit = 10
    else:
        limit = 20

    prompt = f"""You are an expert in {topic}. Answer the question concisely.

In less than {limit} words, answer this question.

Question: {question}

Answer:
"""
    return prompt


def generate_answer(tokenizer, model, question, topic, qtype):

    prompt = build_prompt(question, topic, qtype)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        temperature=0.2,
        top_p=0.9
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    answer = text.split("Answer:")[-1].strip()

    return answer


def main():

    with open(INPUT_FILE) as f:
        data = json.load(f)

    tokenizer, model = load_model()

    for item in data:

        question = item["question"]
        topic = item["topic"]
        qtype = item["type"]

        print("Question:", question)

        answer = generate_answer(
            tokenizer,
            model,
            question,
            topic,
            qtype
        )

        item["llm_answer"] = answer

    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print("Saved results to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
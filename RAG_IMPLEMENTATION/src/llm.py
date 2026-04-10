import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT_TEMPLATE = """Use the following information to answer the question. Limit the answers to 25-30 words. 
Context:
{context}

Question:
{question}

Answer:"""

def load_model(model_name, token=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token=token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        token=token
    )
    return tokenizer, model


def generate_answer(query, context_chunks, tokenizer, model):
    context = "\n\n".join([c["text"] for c in context_chunks])

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=query
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.2,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt
    answer = decoded[len(prompt):].strip()

    return answer
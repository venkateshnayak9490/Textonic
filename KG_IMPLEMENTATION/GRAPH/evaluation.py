import json
from tqdm import tqdm
from pipeline import kg_rag_pipeline


UNWANTED_PHRASES = [
    "Valid Cypher generated on attempt",
    "Cypher validation failed on attempt",
    "All retries failed",
    "Using fallback query",
    "Fallback query generated:",
    "Loading schema",
    "Schema loaded",
]

NO_INFO_PHRASES = [
    "no information found",
    "not available in the knowledge graph",
    "no relevant information",
    "this information is not available"
]


def clean_answer(raw_answer: str) -> str:
    """Remove system logs and keep only the meaningful answer."""
    
    lines = [
        line.strip()
        for line in raw_answer.split("\n")
        if line.strip() and not any(p in line for p in UNWANTED_PHRASES)
    ]

    result = " ".join(lines)

    if any(p in result.lower() for p in NO_INFO_PHRASES):
        return "No information found in the knowledge graph."

    return result


def run_batch_evaluation(input_path: str, output_path: str = None):
    """Run KG-RAG pipeline on a batch of questions and save results."""
    
    output_path = output_path or input_path

    print(f"\n Loading questions from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f" Loaded {total} questions\n")

    print(" Starting batch evaluation...\n")

    for i, item in enumerate(tqdm(data, desc="Processing"), 1):
        question = item.get("question", "")

        print("\n" + "=" * 70)
        print(f"Question {i}/{total} (ID: {item.get('id')})")
        print(f"Topic: {item.get('topic')}")
        print(f"Q: {question}")
        print("=" * 70)

        try:
            raw_answer, kg_triples = kg_rag_pipeline(question)  # unpack tuple
            item["llm_answer"] = clean_answer(raw_answer)
            item["KG"] = kg_triples  # [] if nothing found in KG

            print(f"\n Answer: {item['llm_answer'][:200]}...")
            if kg_triples:
                print(f" KG triples used: {len(kg_triples)}")
                for t in kg_triples[:3]:
                    print(f"   ({t['subject']}) --[{t['relation']}]--> ({t['object']})")
            else:
                print(" KG: no triples retrieved")

        except Exception as e:
            item["llm_answer"] = f"Error: {str(e)}"
            item["KG"] = []
            print(f"\n {item['llm_answer']}")

        # Save progress after each iteration
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(" BATCH EVALUATION COMPLETE!")
    print(f" Results saved to: {output_path}")
    print("=" * 70 + "\n")

    # Summary
    def is_valid_answer(x):
        return x and "No information found" not in x and "Error:" not in x

    answered = sum(is_valid_answer(i.get("llm_answer")) for i in data)
    no_info = sum("No information found" in i.get("llm_answer", "") for i in data)
    errors = sum("Error:" in i.get("llm_answer", "") for i in data)

    print(" Summary:")
    print(f"   - Total questions: {total}")
    print(f"   - Answered with data: {answered}")
    print(f"   - No information found: {no_info}")
    print(f"   - Errors: {errors}")


if __name__ == "__main__":
    INPUT_JSON = "C:\\Users\\jyoth\\Downloads\\climate_hallucination_dataset_test_230_v2.json"
    OUTPUT_JSON = "C:\\Users\\jyoth\\Downloads\\evaluation_results_qwen.json"

    run_batch_evaluation(INPUT_JSON, OUTPUT_JSON)
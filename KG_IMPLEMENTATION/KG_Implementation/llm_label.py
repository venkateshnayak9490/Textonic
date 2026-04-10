"""
Two-pass open-ended label discovery for climate KG.

Pass 1: Free-form label extraction per chunk  →  raw_labels.json
Pass 2: LLM consolidation of all raw labels   →  canonical_schema.json
"""

import re
import json
import time
import random
import requests
from typing import List, Dict
from collections import Counter

PASS1_SYSTEM = """\
You are building a knowledge graph from climate and urban sustainability documents.
Your job is to read a passage and assign it 1-3 node-type labels.

A label describes the CATEGORY OF CONCEPT present — not the topic itself.
  - Labels should be abstract and reusable across many different passages
  - Think about what TYPE of thing is being discussed:
      is it a physical process? a policy instrument? a measurable quantity?
      a geographic entity? a causal mechanism? a social concept?
  - Do NOT copy examples from these instructions — invent labels that fit the text
  - snake_case only, no spaces

Return valid JSON only, no explanation, no markdown.\
"""

PASS1_USER = """\
Passage:
\"\"\"
{chunk_text}
\"\"\"

What node-type labels best describe the CONCEPTS in this passage?

Return ONLY this JSON:
{{
  "labels": ["snake_case_label", "snake_case_label"],
  "reasoning": "One sentence explaining the choice."
}}\
"""

# Pass 2 — consolidation
PASS2_SYSTEM = """\
You are a knowledge graph schema designer.
You have collected raw node-type labels from a large corpus of climate and urban 
sustainability documents. Many labels are duplicates, near-synonyms, or too vague.

Your job: produce a clean canonical label schema from this raw list.\
"""

PASS2_SYSTEM = """\
You are a knowledge graph schema designer.
Your job is to clean and consolidate a list of raw labels into canonical ones.
Return valid JSON only. No explanation, no markdown.\
"""

PASS2_USER = """\
These are raw labels collected from {n_chunks} document chunks:

{label_freq_block}

Return ONLY a JSON object mapping every raw label to its canonical form.
Rules:
- Merge synonyms to one canonical name (e.g. "policy_framework" → "climate_policy")
- Drop noise by mapping to null (e.g. "person", "normal", "reference" → null)
- Keep snake_case

Return ONLY this format, no other text:
{{
  "raw_label_1": "canonical_label",
  "raw_label_2": "canonical_label",
  "noise_label": null
}}\
"""

class NvidiaLLMClient:
    def __init__(self, api_key: str, model: str = "meta/llama-3.1-8b-instruct"):
        self.api_key  = api_key
        self.model    = model
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.headers  = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.0,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        raise Exception(f"API {r.status_code}: {r.text}")


def parse_json_response(raw: str) -> dict | None:
    """Strip markdown fences if present, then parse."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$",       "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def normalise_label(lbl: str) -> str:
    lbl = str(lbl).strip().lower()
    lbl = re.sub(r"[\s\-]+", "_", lbl)
    lbl = re.sub(r"[^a-z0-9_]", "", lbl)
    return lbl


def fix_chunk_text(text: str) -> str:
    match = re.search(r'(?<!\w)[A-Z]', text)
    if match and match.start() > 0 and match.start() < 60:
        text = text[match.start():]
    return text.strip()


def pass1_extract_labels(
    chunks: List[Dict],
    llm: NvidiaLLMClient,
    sample_size: int = 200,
    delay: float = 0.3,
    verbose: bool = True,
) -> Dict:
    """Extract free-form labels from each chunk independently."""

    sample = random.sample(chunks, min(sample_size, len(chunks)))

    if verbose:
        print(f"\n{'='*60}")
        print(f"PASS 1 — Free-form label extraction")
        print(f"{'='*60}")
        print(f"Chunks to process: {len(sample)}\n")

    all_labels:   List[str]  = []
    chunk_results: List[Dict] = []
    errors = 0

    for i, chunk in enumerate(sample, 1):
        text = fix_chunk_text(chunk["text"])

        try:
            raw = llm.generate(
                user_prompt   = PASS1_USER.format(chunk_text=text),
                system_prompt = PASS1_SYSTEM,
                max_tokens    = 150,
            )
            print(f"\n--- RAW RESPONSE (chunk {i}) ---")
            print(repr(raw[:300]))   # repr shows hidden characters

            parsed = parse_json_response(raw)

            if parsed and "labels" in parsed:
                labels    = [normalise_label(l) for l in parsed["labels"][:3] if l]
                reasoning = parsed.get("reasoning", "")
            else:
                # Fallback: grab snake_case tokens from raw text
                labels    = re.findall(r'\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b', raw)[:3]
                reasoning = ""

            all_labels.extend(labels)
            chunk_results.append({
                "chunk_id":  chunk["chunk_id"],
                "labels":    labels,
                "reasoning": reasoning,
            })

        except Exception as exc:
            errors += 1
            print(f"  [ERROR] chunk {i}: {exc}")
            chunk_results.append({
                "chunk_id": chunk["chunk_id"],
                "labels": [],
                "reasoning": f"ERROR: {exc}",
            })

        if verbose and i % 25 == 0:
            print(f"  {i}/{len(sample)} chunks processed  "
                  f"({len(set(all_labels))} unique labels so far)")
        time.sleep(delay)

    label_counts = Counter(all_labels)

    result = {
        "chunk_results": chunk_results,
        "label_frequency": dict(label_counts.most_common()),
        "statistics": {
            "chunks_processed":    len(sample),
            "errors":              errors,
            "unique_labels":       len(label_counts),
            "total_assignments":   len(all_labels),
            "avg_labels_per_chunk": round(len(all_labels) / max(len(sample), 1), 2),
        },
    }

    if verbose:
        print(f"\n  Done. {len(label_counts)} unique labels found.\n")
        print("  Top 25 raw labels:")
        for label, count in label_counts.most_common(25):
            bar = "" * min(int(count / max(label_counts.values()) * 30), 30)
            print(f"  {label:<40s} {count:4d}  {bar}")

    return result

def pass2_consolidate(
    pass1_result: Dict,
    llm: NvidiaLLMClient,
    min_frequency: int = 2,
    verbose: bool = True,
) -> Dict:

    from collections import defaultdict

    label_freq = pass1_result["label_frequency"]
    n_chunks   = pass1_result["statistics"]["chunks_processed"]

    filtered = {k: v for k, v in label_freq.items() if v >= min_frequency}

    if verbose:
        print(f"\n{'='*60}")
        print(f"PASS 2 — Schema consolidation")
        print(f"{'='*60}")
        print(f"Raw labels       : {len(label_freq)}")
        print(f"After freq filter: {len(filtered)}  (min_frequency={min_frequency})\n")

    # Format label list for prompt
    label_block = "\n".join(
        f"  {label} ({count})"
        for label, count in sorted(filtered.items(), key=lambda x: -x[1])
    )

    raw = llm.generate(
        user_prompt   = PASS2_USER.format(
            n_chunks         = n_chunks,
            label_freq_block = label_block,
        ),
        system_prompt = PASS2_SYSTEM,
        max_tokens    = 4000,
        temperature   = 0.1,
    )

    if verbose:
        print("\n--- PASS 2 RAW ---")
        print(raw[:2000])
        print("---\n")

    parsed = parse_json_response(raw)

    if parsed is None:
        print("  [WARN] Could not parse Pass 2 response. Saving raw text.")
        return {"raw_response": raw, "error": "json_parse_failed"}

    canonical_map = {k: v for k, v in parsed.items() if v is not None}
    dropped       = [k for k, v in parsed.items() if v is None]

    groups: Dict = defaultdict(list)
    for raw_label, canonical in canonical_map.items():
        groups[canonical].append(raw_label)

    if verbose:
        print(f"  Canonical labels : {len(groups)}")
        print(f"  Dropped (noise)  : {len(dropped)}")
        print(f"\n  Final schema:")
        for canonical, sources in sorted(groups.items()):
            merged = ", ".join(sources)
            print(f"  {canonical:<35s} ← {merged}")
        if dropped:
            print(f"\n  Dropped: {', '.join(dropped)}")

    return {
        "canonical_map":    canonical_map,       
        "canonical_labels": list(groups.keys()), 
        "merged_groups":    dict(groups),         
        "dropped_labels":   dropped,
    }


def run_label_discovery(
    chunks_path: str       = "chunks.json",
    api_key:     str       = "None",
    model:       str       = "meta/llama-3.2-1b-instruct",
    sample_size: int       = 200,
    min_frequency: int     = 2,
    out_dir:     str       = ".",
    delay:       float     = 0.3,

):
    import os
    os.makedirs(out_dir, exist_ok=True)

    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)
        chunks = data["chunks"]
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    llm = NvidiaLLMClient(api_key=api_key, model=model)

    pass1 = pass1_extract_labels(
        chunks, llm,
        sample_size=sample_size,
        delay=delay,
    )
    with open(f"{out_dir}/raw_labels.json", "w", encoding="utf-8") as f:
        json.dump(pass1, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_dir}/raw_labels.json")

    pass2 = pass2_consolidate(pass1, llm, min_frequency=min_frequency)
    with open(f"{out_dir}/canonical_schema.json", "w", encoding="utf-8") as f:
        json.dump(pass2, f, indent=2, ensure_ascii=False)
    print(f"Saved → {out_dir}/canonical_schema.json")

    return pass1, pass2


if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("--chunks",       default="chunks.json")
    p.add_argument("--api_key",      default=os.getenv("NIM_API_KEY", ""))
    p.add_argument("--model",        default="meta/llama-3.1-8b-instruct")
    p.add_argument("--sample_size",  type=int,   default=500)
    p.add_argument("--min_frequency",type=int,   default=2)
    p.add_argument("--out_dir",      default="kg_output")
    p.add_argument("--delay",        type=float, default=0.3)
    args = p.parse_args()

    run_label_discovery(
        chunks_path   = args.chunks,
        api_key       = args.api_key,
        model         = args.model,
        sample_size   = args.sample_size,
        min_frequency = args.min_frequency,
        out_dir       = args.out_dir,
        delay         = args.delay,
    )
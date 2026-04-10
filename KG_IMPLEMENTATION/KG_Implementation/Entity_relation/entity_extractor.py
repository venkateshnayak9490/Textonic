"""
entity_extractor.py
Hybrid entity extraction pipeline:
  Step 1 — spaCy NER (fast, free) — used as hints only
  Step 2 — LLM classifies spaCy entities into canonical labels
            + extracts domain-specific entities spaCy missed
  Step 3 — Combine, deduplicate, save

Optimisations over v1:
  - Batches BATCH_SIZE chunks per API call (15k → ~3k calls)
  - Parallel threads via ThreadPoolExecutor
  - Flat JSON output — more reliable for small models (Qwen3 1.7B)
  - Hard label validation — discards any label not in canonical schema
"""

import re
import json
import time
import requests
from typing import List, Dict, Optional
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import spacy


SPACY_MODEL   = "en_core_web_sm"
SCHEMA_PATH   = "/mnt/sd1/jyothika/jyo/jyothika/INLP_Pro/INLP_Pro/kg_output/canonical_schema.json"
CHUNKS_PATH   = "/mnt/sd1/jyothika/jyo/jyothika/INLP_Pro/INLP_Pro/chunks.json"
OUTPUT_PATH   = "/mnt/sd1/jyothika/jyo/jyothika/INLP_Pro/INLP_Pro/kg_output/entities.json"
NIM_MODEL     = "meta/llama-3.1-8b-instruct"

BATCH_SIZE    = 5    # chunks per API call — reduce to 3 if model struggles
MAX_WORKERS   = 4    # parallel threads — keep ≤ 5 to avoid NIM rate limits
REQUEST_DELAY = 0.1  # seconds between API calls per thread
MAX_TOKENS    = 800  # enough for 5 chunks of entities

# spaCy labels useful for climate KG — used as hints to the LLM only
USEFUL_SPACY_LABELS = {
    "ORG",   # IPCC, UNEP, EPA, World Bank
    "GPE",   # Arctic, Amazon, South Asia, countries
    "LOC",   # geographic locations
    "LAW",   # Paris Agreement, Kyoto Protocol
    "DATE",  # 2050, pre-industrial era, 20-year horizon
}



def load_canonical_labels(schema_path: str = SCHEMA_PATH) -> List[str]:
    """Load canonical labels from Pass 2 output."""
    path = Path(schema_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Schema not found at {schema_path}. Run label_discovery.py first."
        )
    with open(path, encoding="utf-8") as f:
        schema = json.load(f)

    labels = schema.get("canonical_labels", [])
    if not labels:
        raise ValueError(
            "canonical_labels is empty in schema. "
            "Check that Pass 2 completed successfully."
        )
    print(f"Loaded {len(labels)} canonical labels from {schema_path}")
    return labels


class NvidiaLLMClient:
    def __init__(self, api_key: str, model: str = NIM_MODEL):
        if not api_key or not api_key.strip():
            raise ValueError("API key is empty. Pass a valid NIM API key.")
        self.api_key  = api_key
        self.model    = model
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.headers  = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }

    def generate(
        self,
        user_prompt:   str,
        system_prompt: str,
        max_tokens:    int   = MAX_TOKENS,
        temperature:   float = 0.1,
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
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        raise Exception(f"API {r.status_code}: {r.text}")


def parse_json_response(raw: str) -> Optional[list]:
    """
    Strip markdown fences if present, then parse JSON.
    Accepts a flat list OR a dict with a known list key.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$",       "", raw)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("entities", "classified_entities", "results"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
    except json.JSONDecodeError:
        pass
    return None


def normalise_text(text: str) -> str:
    """Lowercase + collapse whitespace — used for deduplication."""
    return re.sub(r"\s+", " ", text.strip().lower())


def deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    """
    Deduplicate by (normalised_text, canonical_label, chunk_id).
    Prevents cross-chunk deduplication — same entity in different
    chunks is a different record.
    """
    seen   = set()
    result = []
    for ent in entities:
        key = (
            normalise_text(ent.get("text", "")),
            ent.get("canonical_label", ""),
            ent.get("chunk_id", ""),       
        )
        if key not in seen and key[0]:
            seen.add(key)
            result.append(ent)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — spaCy NER (hints only)
# ─────────────────────────────────────────────────────────────────────────────
def extract_entities_spacy(text: str, nlp) -> List[Dict]:
    """
    Extract named entities with spaCy.
    Results are passed to LLM as hints — not used directly as KG entities.
    Only keeps entity types that are meaningful for a climate KG.
    """
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in USEFUL_SPACY_LABELS:
            entities.append({
                "text":        ent.text,
                "spacy_label": ent.label_,
            })
    return entities


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LLM classification + domain enrichment (batched)
# ─────────────────────────────────────────────────────────────────────────────
ENTITY_SYSTEM = """\
You are a climate science knowledge graph builder.
Your job is to extract named entities from text and assign each one a label
from a provided canonical list.
Return valid JSON only. No explanation, no markdown.\
"""

ENTITY_USER = """\
Extract climate-relevant entities from the passages below.

SpaCy hints (may be incomplete or noisy — use as guidance only):
{spacy_hints}

Your ONLY allowed labels are:
{canonical_labels}

Do NOT invent labels outside this list. Pick the closest fit.
Use labels EXACTLY as they appear above.

Passages:
{passages}

Tasks:
1. Classify the spaCy hints that are relevant to climate or sustainability.
   Skip irrelevant ones (generic names, unrelated orgs).

2. Extract additional entities that spaCy missed — focus on:
   - Gases and substances
   - Emission sources and processes
   - Climate effects and impacts
   - Policies, targets, agreements
   - Measurements and quantities
   Only extract entities that fit one of your allowed labels.

Return ONLY a flat JSON array — one entry per entity across ALL passages:
[
  {{"text": "<entity text>", "canonical_label": "<label>", "chunk_id": "<chunk_id>"}},
  {{"text": "<entity text>", "canonical_label": "<label>", "chunk_id": "<chunk_id>"}}
]\
"""


def classify_entities_llm_batch(
    batch:            List[Dict],
    spacy_hints:      Dict[str, List[Dict]],
    canonical_labels: List[str],
    llm:              NvidiaLLMClient,
) -> List[Dict]:
    """
    Send a batch of chunks to the LLM in one API call.
    Returns a flat list of validated entities.
    """
    passages_block = "\n\n".join(
        f'[chunk_id: {c["chunk_id"]}]\n{c["text"].strip()}'
        for c in batch
    )

    all_hints = []
    for c in batch:
        hints = spacy_hints.get(c["chunk_id"], [])
        for h in hints:
            all_hints.append(
                f'{h["text"]} ({h["spacy_label"]}) — from {c["chunk_id"]}'
            )
    hints_str = "\n".join(all_hints) if all_hints else "None found."

    prompt = ENTITY_USER.format(
        spacy_hints      = hints_str,
        canonical_labels = ", ".join(canonical_labels),
        passages         = passages_block,
    )

    raw    = llm.generate(prompt, system_prompt=ENTITY_SYSTEM, max_tokens=MAX_TOKENS)
    parsed = parse_json_response(raw)

    if parsed is None:
        return []

    label_set       = set(canonical_labels)
    valid_chunk_ids = {c["chunk_id"]: c for c in batch}
    validated       = []

    for ent in parsed:
        if not isinstance(ent, dict):
            continue

        text   = ent.get("text", "").strip()
        label  = ent.get("canonical_label", "").strip()
        cid    = ent.get("chunk_id", batch[0]["chunk_id"])

        if not text or not label:
            continue
        if label not in label_set:
            continue  
        if cid not in valid_chunk_ids:
            cid = batch[0]["chunk_id"]  

        source_file = valid_chunk_ids[cid].get("source_file", "unknown")

        validated.append({
            "text":            text,
            "canonical_label": label,
            "chunk_id":        cid,
            "source_file":     source_file,
        })

    return validated

def extract_all_entities(
    chunks:           List[Dict],
    nlp,
    llm:              NvidiaLLMClient,
    canonical_labels: List[str],
    max_chunks:       Optional[int] = None,
    batch_size:       int           = BATCH_SIZE,
    max_workers:      int           = MAX_WORKERS,
    delay:            float         = REQUEST_DELAY,
    verbose:          bool          = True,
) -> Dict:
    """
    Run hybrid extraction over all (or a subset of) chunks.
    spaCy runs first on all chunks (fast), then LLM processes batches in parallel.
    """
    if max_chunks:
        chunks = chunks[:max_chunks]

    # ── Step 1: Run spaCy on all chunks upfront (fast, no API cost) ──────────
    if verbose:
        print(f"\n{'='*60}")
        print(f"ENTITY EXTRACTION")
        print(f"{'='*60}")
        print(f"  Chunks          : {len(chunks)}")
        print(f"  Batch size      : {batch_size}")
        print(f"  Workers         : {max_workers}")
        print(f"  Canonical labels: {len(canonical_labels)}")
        print(f"\n  Running spaCy on all chunks...")

    spacy_hints: Dict[str, List[Dict]] = {}
    for chunk in chunks:
        found = extract_entities_spacy(chunk["text"], nlp)
        if found:
            spacy_hints[chunk["chunk_id"]] = found

    if verbose:
        total_hints = sum(len(v) for v in spacy_hints.values())
        print(f"  spaCy found {total_hints} hints across "
              f"{len(spacy_hints)} chunks\n")

    # ── Step 2: Split into batches ────────────────────────────────────────────
    batches = [
        chunks[i: i + batch_size]
        for i in range(0, len(chunks), batch_size)
    ]

    if verbose:
        print(f"  Sending {len(batches)} batches to LLM "
              f"({batch_size} chunks each)...\n")

    # ── Step 3: Parallel LLM calls ────────────────────────────────────────────
    all_entities: List[Dict] = []
    errors = 0
    done   = 0

    def process_batch(batch: List[Dict]) -> List[Dict]:
        result = classify_entities_llm_batch(
            batch, spacy_hints, canonical_labels, llm
        )
        time.sleep(delay)
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_batch, batch): batch
            for batch in batches
        }

        for future in as_completed(futures):
            try:
                entities = future.result()
                all_entities.extend(entities)
            except Exception as exc:
                errors += 1
                if verbose:
                    batch = futures[future]
                    print(f"  [ERROR] batch starting at "
                          f"{batch[0].get('chunk_id', '?')}: {exc}")

            done += 1
            if verbose and done % 20 == 0:
                print(f"  {done}/{len(batches)} batches done | "
                      f"{len(all_entities)} entities | "
                      f"{errors} errors")

    # ── Step 4: Global deduplication ─────────────────────────────────────────
    all_entities = deduplicate_entities(all_entities)

    label_dist = Counter(e.get("canonical_label", "unknown") for e in all_entities)

    result = {
        "entities": all_entities,
        "statistics": {
            "chunks_processed":   len(chunks),
            "batches_processed":  len(batches),
            "total_entities":     len(all_entities),
            "errors":             errors,
            "label_distribution": dict(label_dist.most_common()),
        },
    }

    if verbose:
        print(f"\n  Done.")
        print(f"  Total entities (deduplicated) : {len(all_entities)}")
        print(f"  Errors                        : {errors}")
        print(f"\n  Label distribution:")
        for label, count in label_dist.most_common():
            bar = "" * min(int(count / max(label_dist.values()) * 30), 30)
            print(f"  {label:<35s} {count:4d}  {bar}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse, os

    p = argparse.ArgumentParser(description="Hybrid entity extraction for climate KG")
    p.add_argument("--chunks",      default=CHUNKS_PATH)
    p.add_argument("--schema",      default=SCHEMA_PATH)
    p.add_argument("--output",      default=OUTPUT_PATH)
    p.add_argument("--api_key",     default=os.getenv("NIM_API_KEY", ""))
    p.add_argument("--model",       default=NIM_MODEL)
    p.add_argument("--max_chunks",  type=int,   default=None,
                   help="Process only first N chunks (for testing)")
    p.add_argument("--batch_size",  type=int,   default=BATCH_SIZE,
                   help="Chunks per API call")
    p.add_argument("--max_workers", type=int,   default=MAX_WORKERS,
                   help="Parallel threads")
    p.add_argument("--delay",       type=float, default=REQUEST_DELAY)
    args = p.parse_args()

    # ── Load resources ────────────────────────────────────────────────────────
    canonical_labels = load_canonical_labels(args.schema)

    print(f"Loading spaCy model: {SPACY_MODEL}")
    nlp = spacy.load(SPACY_MODEL)

    llm = NvidiaLLMClient(api_key=args.api_key, model=args.model)

    with open(args.chunks, encoding="utf-8") as f:
        data   = json.load(f)
        chunks = data["chunks"] if isinstance(data, dict) else data
    print(f"Loaded {len(chunks)} chunks from {args.chunks}")

    # ── Run extraction ────────────────────────────────────────────────────────
    result = extract_all_entities(
        chunks           = chunks,
        nlp              = nlp,
        llm              = llm,
        canonical_labels = canonical_labels,
        max_chunks       = args.max_chunks,
        batch_size       = args.batch_size,
        max_workers      = args.max_workers,
        delay            = args.delay,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
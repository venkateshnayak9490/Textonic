"""
matrix_relation_extractor.py

Schema-driven relation extraction using a label×label matrix.

Phase 1: Build 66×66 label relation matrix (LLM, one-time)
Phase 2: For each chunk, find co-occurring entity pairs
Phase 3: Look up (label_A, label_B) in matrix → assign relation
Phase 4: Save triples

This approach is:
  - Consistent  : same label pair always gets same relation
  - Fast        : no LLM call per chunk, matrix built once
  - Climate-specific : matrix defined from your canonical labels
"""

import json
import re
import time
import requests
import itertools
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA_PATH   = "/mnt/sd1/jyothika/jyo/jyothika/INLP_Pro/INLP_Pro/kg_output/canonical_schema.json"
ENTITIES_PATH = "/mnt/sd1/jyothika/jyo/jyothika/INLP_Pro/INLP_Pro/kg_output/entities.json"
CHUNKS_PATH   = "/mnt/sd1/jyothika/jyo/jyothika/INLP_Pro/INLP_Pro/chunks.json"
MATRIX_PATH   = "/mnt/sd1/jyothika/jyo/jyothika/INLP_Pro/INLP_Pro/kg_output/label_relation_matrix.json"
TRIPLES_PATH  = "/mnt/sd1/jyothika/jyo/jyothika/INLP_Pro/INLP_Pro/kg_output/triples_2.json"

NIM_MODEL     = "meta/llama-3.1-8b-instruct"
NIM_API_KEY   = ""   # set via --api_key or NIM_API_KEY env var

# ─────────────────────────────────────────────────────────────────────────────
# HYBRID MATRIX RULES
# ─────────────────────────────────────────────────────────────────────────────

# Labels whose entities should NEVER appear in triples
# (quantities, dates, citations are not meaningful KG nodes)
SKIP_LABELS = {
    "quantity",
    "date",
    "publication",
    "reference",
    "academic_reference",
    "citation",
}

# Manual overrides for the most important climate label pairs
# These are applied BEFORE the LLM — guaranteed correct relations
MANUAL_MATRIX = {
    # Core emission chain
    ("emission_source",   "greenhouse_gas"):    "EMITS",
    ("greenhouse_gas",    "climate_process"):   "CAUSES",
    ("greenhouse_gas",    "climate_impact"):    "CAUSES",
    ("greenhouse_gas",    "climate_effect"):    "CAUSES",
    ("emission_source",   "climate_process"):   "CONTRIBUTES_TO",
    ("emission_source",   "climate_impact"):    "CONTRIBUTES_TO",
    ("emission_source",   "climate_effect"):    "CONTRIBUTES_TO",

    # Reverse emission chain
    ("greenhouse_gas",    "emission_source"):   "EMITTED_BY",
    ("climate_process",   "emission_source"):   "CAUSED_BY",
    ("climate_impact",    "greenhouse_gas"):    "CAUSED_BY",
    ("climate_impact",    "emission_source"):   "CAUSED_BY",

    # Policy and regulation
    ("climate_policy",    "emission_source"):   "REGULATES",
    ("climate_policy",    "greenhouse_gas"):    "REDUCES",
    ("climate_policy",    "climate_impact"):    "MITIGATES",
    ("climate_policy",    "climate_effect"):    "MITIGATES",
    ("organization",      "climate_policy"):    "IMPLEMENTS",
    ("organization",      "emission_source"):   "MONITORS",
    ("organization",      "greenhouse_gas"):    "MONITORS",

    # Technology
    ("technology",        "emission_source"):   "REPLACES",
    ("technology",        "greenhouse_gas"):    "REDUCES",
    ("technology",        "climate_policy"):    "ENABLES",
    ("technology",        "climate_impact"):    "MITIGATES",

    # Ecosystem
    ("ecosystem",         "greenhouse_gas"):    "ABSORBS",
    ("ecosystem",         "climate_impact"):    "AFFECTED_BY",
    ("ecosystem",         "climate_effect"):    "AFFECTED_BY",

    # Measurement
    ("greenhouse_gas",    "emission_metric"):   "MEASURED_BY",
    ("emission_source",   "emission_metric"):   "MEASURED_BY",
    ("climate_effect",    "emission_metric"):   "MEASURED_BY",
}

# Label pairs that are always meaningless — block completely
ALWAYS_NONE = {
    ("quantity",          "quantity"),
    ("quantity",          "social_impact"),
    ("social_impact",     "quantity"),
    ("quantity",          "climate_process"),
    ("climate_process",   "quantity"),
    ("quantity",          "emission_source"),
    ("emission_source",   "quantity"),
    ("publication",       "quantity"),
    ("reference",         "quantity"),
    ("date",              "date"),
}


# ─────────────────────────────────────────────────────────────────────────────
# NIM CLIENT
# ─────────────────────────────────────────────────────────────────────────────
class NvidiaLLMClient:
    def __init__(self, api_key: str, model: str = NIM_MODEL):
        if not api_key or not api_key.strip():
            raise ValueError("API key is empty.")
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
        system_prompt: str = "You are a climate science knowledge graph expert. Return valid JSON only.",
        max_tokens:    int = 200,
        temperature:   float = 0.0,
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


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def parse_json(raw: str) -> Optional[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$",       "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def is_meaningful_entity(text: str) -> bool:
    """
    Returns False for entities that are purely numeric or too short
    to be meaningful KG nodes — years, percentages, counts, etc.
    """
    t = text.strip()
    if len(t) < 3:
        return False
    # Pure number, percentage, or numeric phrase
    if re.match(r'^[\d\s,\.%\-]+$', t):
        return False
    # Standalone year
    if re.match(r'^\d{4}$', t):
        return False
    # Looks like "X per cent" or "X million" etc.
    if re.match(r'^\d[\d\s\.,]*(per\s+cent|million|billion|trillion|%)', t, re.IGNORECASE):
        return False
    return True


def load_canonical_labels(schema_path: str) -> List[str]:
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    labels = schema.get("canonical_labels", [])
    if not labels:
        raise ValueError("canonical_labels is empty in schema.")
    return labels


def load_entities(entities_path: str) -> List[Dict]:
    with open(entities_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("entities", data) if isinstance(data, dict) else data


def load_chunks(chunks_path: str) -> List[Dict]:
    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"] if isinstance(data, dict) else data


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — BUILD LABEL RELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
MATRIX_SYSTEM = """\
You are a climate science ontology expert building a knowledge graph.
Your job is to define the most meaningful directed relationship between
two concept types in the climate science domain.
Return valid JSON only. No explanation, no markdown.\
"""

# Open-ended — LLM freely discovers the relation from label meanings
MATRIX_BATCH_PROMPT = """\
For each pair below, define the best directed relationship FROM the first
concept type TO the second, in a climate science knowledge graph.

Rules:
- Return a short verb phrase in UPPER_SNAKE_CASE
  Good examples: EMITS, CAUSES, ABSORBS, REGULATES, FUNDS, MONITORS,
                 INCREASES, STORED_IN, MEASURED_BY, CONTRIBUTES_TO
- Be specific — think about how these two concept types actually interact
  in climate science
- Consider directionality carefully:
    emission_source → greenhouse_gas = EMITS  (not EMITTED_BY)
- If there is genuinely NO meaningful directed relationship, return "NONE"
- Do NOT use vague catch-alls like RELATED_TO or ASSOCIATED_WITH
  unless nothing more specific applies

Pairs:
{pair_lines}

Return ONLY a JSON object mapping pair number to relation:
{{
  "1": "EMITS",
  "2": "NONE",
  "3": "REGULATES"
}}\
"""


def build_matrix(
    labels:  List[str],
    llm:     NvidiaLLMClient,
    delay:   float = 0.2,
    verbose: bool  = True,
) -> Dict[str, Dict[str, str]]:
    """
    Build a label×label matrix where matrix[A][B] = relation from A to B.
    Relations are discovered freely by the LLM — not constrained by a list.
    For 66 labels: 66×65 = 4290 pairs (skip A→A diagonal).
    """
    matrix: Dict[str, Dict[str, str]] = {l: {} for l in labels}

    pairs = [
        (a, b) for a in labels for b in labels
        if a != b
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"PHASE 1 — Building {len(labels)}×{len(labels)} label matrix")
        print(f"{'='*60}")
        print(f"  Total pairs to define : {len(pairs)}")
        print(f"  Batching 10 pairs per API call")
        print(f"  Est. API calls        : {len(pairs)//10 + 1}\n")

    BATCH = 10
    done  = 0

    # Separate pairs into: manual, always_none, and LLM-needed
    llm_pairs = []
    for (a, b) in pairs:
        if a in SKIP_LABELS or b in SKIP_LABELS:
            matrix[a][b] = "NONE"
        elif (a, b) in MANUAL_MATRIX:
            matrix[a][b] = MANUAL_MATRIX[(a, b)]
        elif (a, b) in ALWAYS_NONE:
            matrix[a][b] = "NONE"
        else:
            llm_pairs.append((a, b))

    manual_count = len(pairs) - len(llm_pairs)
    if verbose:
        print(f"  Manual overrides applied : {manual_count}")
        print(f"  Pairs sent to LLM        : {len(llm_pairs)}")
        print(f"  Est. API calls           : {len(llm_pairs)//BATCH + 1}\n")

    for i in range(0, len(llm_pairs), BATCH):
        batch = llm_pairs[i: i + BATCH]

        pair_lines = "\n".join(
            f'  {j+1}. FROM "{a}" TO "{b}"'
            for j, (a, b) in enumerate(batch)
        )

        prompt = MATRIX_BATCH_PROMPT.format(pair_lines=pair_lines)

        try:
            raw    = llm.generate(prompt, system_prompt=MATRIX_SYSTEM, max_tokens=200)
            parsed = parse_json(raw)

            if parsed:
                for j, (a, b) in enumerate(batch):
                    rel = parsed.get(str(j + 1), "NONE")
                    rel = rel.strip().upper().replace(" ", "_").replace("-", "_")
                    if not rel:
                        rel = "NONE"
                    matrix[a][b] = rel
            else:
                for (a, b) in batch:
                    matrix[a][b] = "NONE"

        except Exception as exc:
            print(f"  [ERROR] batch {i//BATCH}: {exc}")
            for (a, b) in batch:
                matrix[a][b] = "NONE"

        done += len(batch)
        if verbose and done % 100 == 0:
            print(f"  {done}/{len(llm_pairs)} LLM pairs defined")

        time.sleep(delay)

    # Diagonal: same label → same label = NONE
    for l in labels:
        matrix[l][l] = "NONE"

    if verbose:
        # Show what relations the LLM invented
        from collections import Counter
        all_rels = [
            v for row in matrix.values()
            for v in row.values()
            if v != "NONE"
        ]
        rel_counts = Counter(all_rels)
        print(f"\n  Matrix complete.")
        print(f"  Unique relations discovered: {len(rel_counts)}")
        print(f"\n  Top relations found:")
        for rel, cnt in rel_counts.most_common(15):
            print(f"    {rel:<30s} {cnt}")

    return matrix


def save_matrix(matrix: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2, ensure_ascii=False)
    print(f"  Matrix saved → {path}")


def load_matrix(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — EXTRACT TRIPLES USING MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def build_entity_chunk_map(entities: List[Dict]) -> Dict[str, List[Dict]]:
    """Group entities by chunk_id."""
    chunk_map = defaultdict(list)
    for ent in entities:
        chunk_map[ent["chunk_id"]].append(ent)
    return chunk_map


def extract_triples_from_chunk(
    chunk_entities: List[Dict],
    matrix:         Dict[str, Dict[str, str]],
    chunk_id:       str,
    source_file:    str,
) -> List[Dict]:
    """
    For every pair of entities in the same chunk:
      1. Look up (label_A, label_B) in the matrix
      2. If relation != NONE → create triple
    """
    triples = []
    seen    = set()  # avoid duplicate triples in same chunk

    # All ordered pairs of entities in this chunk
    for ent_a, ent_b in itertools.permutations(chunk_entities, 2):
        label_a = ent_a.get("canonical_label", "")
        label_b = ent_b.get("canonical_label", "")
        text_a  = ent_a.get("text", "").strip()
        text_b  = ent_b.get("text", "").strip()

        if not label_a or not label_b or not text_a or not text_b:
            continue

        # Skip entities with meaningless labels
        if label_a in SKIP_LABELS or label_b in SKIP_LABELS:
            continue

        # Skip purely numeric entity text
        if not is_meaningful_entity(text_a) or not is_meaningful_entity(text_b):
            continue

        # Look up relation in matrix
        relation = matrix.get(label_a, {}).get(label_b, "NONE")

        # Skip if no relation defined
        if not relation or relation == "NONE":
            continue

        # Dedup within chunk
        key = (text_a.lower(), relation, text_b.lower())
        if key in seen:
            continue
        seen.add(key)

        triples.append({
            "subject":        text_a,
            "subject_label":  label_a,
            "relation":       relation,
            "object":         text_b,
            "object_label":   label_b,
            "chunk_id":       chunk_id,
            "source_file":    source_file,
        })

    return triples


def extract_all_triples(
    chunks:      List[Dict],
    entities:    List[Dict],
    matrix:      Dict[str, Dict[str, str]],
    verbose:     bool = True,
) -> List[Dict]:
    """Run matrix-based triple extraction over all chunks."""

    chunk_entity_map = build_entity_chunk_map(entities)
    chunk_lookup     = {c["chunk_id"]: c for c in chunks}

    if verbose:
        print(f"\n{'='*60}")
        print(f"PHASE 2 — Extracting triples from entity pairs")
        print(f"{'='*60}")
        print(f"  Chunks with entities: {len(chunk_entity_map)}")

    all_triples: List[Dict] = []
    global_seen: set        = set()

    for chunk_id, chunk_entities in chunk_entity_map.items():
        chunk       = chunk_lookup.get(chunk_id, {})
        source_file = chunk.get("source_file", "unknown")

        chunk_triples = extract_triples_from_chunk(
            chunk_entities, matrix, chunk_id, source_file
        )
        all_triples.extend(chunk_triples)

    # Global deduplication — same (subj, rel, obj) across chunks
    deduped  = []
    seen_global = set()
    for t in all_triples:
        key = (t["subject"].lower(), t["relation"], t["object"].lower())
        if key not in seen_global:
            seen_global.add(key)
            deduped.append(t)

    if verbose:
        from collections import Counter
        rel_dist  = Counter(t["relation"] for t in deduped)
        pair_dist = Counter(
            (t["subject_label"], t["object_label"]) for t in deduped
        )
        print(f"\n  Total triples (raw)        : {len(all_triples)}")
        print(f"  Total triples (deduplicated): {len(deduped)}")
        print(f"\n  Relation distribution:")
        for rel, cnt in rel_dist.most_common():
            bar = "█" * min(int(cnt / max(rel_dist.values()) * 30), 30)
            print(f"  {rel:<25s} {cnt:5d}  {bar}")
        print(f"\n  Top 10 label pair combinations:")
        for (a, b), cnt in pair_dist.most_common(10):
            print(f"  {a} → {b}: {cnt}")

    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse, os

    p = argparse.ArgumentParser()
    p.add_argument("--api_key",       default=os.getenv("NIM_API_KEY", ""))
    p.add_argument("--schema",        default=SCHEMA_PATH)
    p.add_argument("--entities",      default=ENTITIES_PATH)
    p.add_argument("--chunks",        default=CHUNKS_PATH)
    p.add_argument("--matrix_out",    default=MATRIX_PATH)
    p.add_argument("--triples_out",   default=TRIPLES_PATH)
    p.add_argument("--skip_matrix",   action="store_true",
                   help="Skip Phase 1 — load existing matrix instead")
    p.add_argument("--delay",         type=float, default=0.2)
    args = p.parse_args()

    # ── Load resources ────────────────────────────────────────────────────────
    labels   = load_canonical_labels(args.schema)
    entities = load_entities(args.entities)
    chunks   = load_chunks(args.chunks)

    print(f"Labels   : {len(labels)}")
    print(f"Entities : {len(entities)}")
    print(f"Chunks   : {len(chunks)}")

    # ── Phase 1: Build or load matrix ────────────────────────────────────────
    if args.skip_matrix and Path(args.matrix_out).exists():
        print(f"\nLoading existing matrix from {args.matrix_out}")
        matrix = load_matrix(args.matrix_out)
    else:
        llm    = NvidiaLLMClient(api_key=args.api_key)
        matrix = build_matrix(labels, llm, delay=args.delay)
        save_matrix(matrix, args.matrix_out)

    # ── Phase 2: Extract triples ──────────────────────────────────────────────
    triples = extract_all_triples(chunks, entities, matrix)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(args.triples_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.triples_out, "w", encoding="utf-8") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {args.triples_out}")
    print(f"Total triples: {len(triples)}")


if __name__ == "__main__":
    main()
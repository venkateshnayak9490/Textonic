import re
import time
from neo4j_client import Neo4jClient
from cypher_generator import generate_cypher, get_schema
from llm_handler import generate_answer
from nim_client import NvidiaLLMClient

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "Goparaju_12"

LLM_API_KEY = "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"
# LLM_MODEL   = "meta/llama-3.2-1b-instruct"
LLM_MODEL = "Qwen/Qwen2.5-1B-Instruct"
NO_INFO_MSG = "No information found in the knowledge graph."

llm        = NvidiaLLMClient(api_key=LLM_API_KEY, model=LLM_MODEL)
neo_client = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)


# ─────────────────────────────────────────────
# KEYWORD EXTRACTION
# ─────────────────────────────────────────────
def extract_keywords(text: str) -> list[str]:
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop = {
        'what', 'why', 'how', 'who', 'does', 'the', 'this', 'that',
        'with', 'from', 'into', 'about', 'when', 'are', 'and', 'for',
        'would', 'could', 'should', 'which', 'their', 'there', 'have',
        'been', 'toward', 'influence', 'explain', 'describe', 'tell',
        'important', 'mean', 'definition'
    }
    words = text.lower().split()
    return [w for w in words if w not in stop and len(w) > 3]


# ─────────────────────────────────────────────
# FORMAT KG TRIPLES
# Converts raw Neo4j rows → clean (subject, relation, object)
# ─────────────────────────────────────────────
def format_kg_triples(results: list) -> list:
    triples = []
    for r in results:
        if not isinstance(r, dict):
            continue

        # 1-hop result
        if 'entity' in r and 'related_entity' in r:
            triples.append({
                "subject":  r.get('entity', ''),
                "relation": r.get('relationship', ''),
                "object":   r.get('related_entity', '')
            })

        # 2-hop result — capture both hops
        if 'second_related_entity' in r and r.get('second_related_entity'):
            triples.append({
                "subject":  r.get('related_entity', ''),
                "relation": r.get('second_relationship', ''),
                "object":   r.get('second_related_entity', '')
            })

    # Deduplicate while preserving order
    seen   = set()
    unique = []
    for t in triples:
        key = (t['subject'], t['relation'], t['object'])
        if key not in seen and all(key):  # skip empty triples
            seen.add(key)
            unique.append(t)

    return unique


# ─────────────────────────────────────────────
# RELEVANCE CHECK
# ─────────────────────────────────────────────
def results_are_relevant(question: str, results: list) -> bool:
    if not results:
        return False

    keywords = extract_keywords(question)
    if not keywords:
        return False

    results_text = " ".join(
        str(v).lower()
        for r in results
        for v in (r.values() if isinstance(r, dict) else [r])
    )

    return any(kw in results_text for kw in keywords)


# ─────────────────────────────────────────────
# SAFE NEO4J QUERY
# ─────────────────────────────────────────────
def safe_neo_query(cypher: str) -> list:
    if not cypher or len(cypher.strip()) < 10:
        return []
    try:
        return neo_client.query(cypher)
    except Exception as e:
        print(f"  [Neo4j error]: {e}")
        return []


# ─────────────────────────────────────────────
# STEP 1 QUERIES
# ─────────────────────────────────────────────
def build_step1_queries(phrase: str, keywords: list, schema: dict) -> list[str]:
    safe_phrase = phrase.replace("'", "\\'")
    rel_list    = schema.get('relationships', [])
    queries     = []

    # 1a. Exact phrase, 1-hop, directed
    queries.append(f"""
        MATCH (n:Entity)-[r]->(m:Entity)
        WHERE toLower(n.name) CONTAINS '{safe_phrase}'
        RETURN n.name AS entity,
               type(r) AS relationship,
               m.name  AS related_entity
        ORDER BY n.name
        LIMIT 15
    """)

    # 1b. Exact phrase, 2-hop traversal
    queries.append(f"""
        MATCH (n:Entity)-[r1]->(mid:Entity)-[r2]->(m:Entity)
        WHERE toLower(n.name) CONTAINS '{safe_phrase}'
        RETURN n.name   AS entity,
               type(r1) AS relationship,
               mid.name AS related_entity,
               type(r2) AS second_relationship,
               m.name   AS second_related_entity
        LIMIT 10
    """)

    # 1c. Phrase on either side, directed, 1-hop
    queries.append(f"""
        MATCH (n:Entity)-[r]->(m:Entity)
        WHERE toLower(n.name) CONTAINS '{safe_phrase}'
           OR toLower(m.name) CONTAINS '{safe_phrase}'
        RETURN n.name AS entity,
               type(r) AS relationship,
               m.name  AS related_entity
        LIMIT 15
    """)

    # 1d. Anchor keyword filtered by schema relationship types
    if rel_list and keywords:
        anchor     = keywords[0].replace("'", "\\'")
        rel_filter = " OR ".join([f"type(r) = '{rel}'" for rel in rel_list[:8]])
        queries.append(f"""
            MATCH (n:Entity)-[r]->(m:Entity)
            WHERE toLower(n.name) CONTAINS '{anchor}'
              AND ({rel_filter})
            RETURN n.name AS entity,
                   type(r) AS relationship,
                   m.name  AS related_entity
            LIMIT 15
        """)

    return queries


# ─────────────────────────────────────────────
# STEP 2 QUERIES
# ─────────────────────────────────────────────
def build_step2_queries(keywords: list) -> list[str]:
    queries = []

    for kw in keywords[:3]:
        safe_kw = kw.replace("'", "\\'")
        queries.append(f"""
            MATCH (n:Entity)-[r]->(m:Entity)
            WHERE toLower(n.name) CONTAINS '{safe_kw}'
            RETURN n.name AS entity,
                   type(r) AS relationship,
                   m.name  AS related_entity
            LIMIT 12
        """)

    if keywords:
        safe_kw = keywords[0].replace("'", "\\'")
        queries.append(f"""
            MATCH (n:Entity)-[r1]->(mid:Entity)-[r2]->(m:Entity)
            WHERE toLower(n.name) CONTAINS '{safe_kw}'
               OR toLower(mid.name) CONTAINS '{safe_kw}'
            RETURN n.name   AS entity,
                   type(r1) AS relationship,
                   mid.name AS related_entity,
                   type(r2) AS second_relationship,
                   m.name   AS second_related_entity
            LIMIT 10
        """)

    return queries


# ─────────────────────────────────────────────
# STEP 3 QUERIES
# ─────────────────────────────────────────────
def build_step3_queries(keywords: list) -> list[str]:
    if not keywords:
        return []

    clauses = []
    for kw in keywords[:5]:
        safe_kw = kw.replace("'", "\\'")
        clauses.append(f"toLower(n.name) CONTAINS '{safe_kw}'")
        clauses.append(f"toLower(m.name) CONTAINS '{safe_kw}'")

    where = " OR ".join(clauses)

    return [f"""
        MATCH (n:Entity)-[r]-(m:Entity)
        WHERE {where}
        RETURN n.name AS entity,
               type(r) AS relationship,
               m.name  AS related_entity
        LIMIT 25
    """]


# ─────────────────────────────────────────────
# RUN QUERIES UNTIL RELEVANT RESULT FOUND
# ─────────────────────────────────────────────
def run_queries_until_relevant(queries: list, question: str) -> list:
    for cypher in queries:
        results = safe_neo_query(cypher)
        if results and results_are_relevant(question, results):
            return results
    return []


# ─────────────────────────────────────────────
# MAIN PIPELINE
# Returns: tuple (answer: str, kg_triples: list)
# kg_triples is [] when nothing was found in KG
# ─────────────────────────────────────────────
def kg_rag_pipeline(question: str) -> tuple[str, list]:
    time.sleep(0.5)

    try:
        keywords = extract_keywords(question)
        if not keywords:
            return NO_INFO_MSG, []

        phrase = " ".join(keywords[:3])

        # ── STEP 1: LLM-generated Cypher ──
        print("  [Step 1] LLM generating Cypher...")
        try:
            llm_cypher = generate_cypher(question, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            results    = safe_neo_query(llm_cypher)
            if results and results_are_relevant(question, results):
                print("  [Step 1] LLM Cypher hit")
                return generate_answer(question, results), format_kg_triples(results)
        except Exception as e:
            print(f"  [Step 1] Cypher generator failed: {e}")

        # ── STEP 1b: Schema-guided directed queries ──
        print("  [Step 1b] Schema-guided directed queries...")
        try:
            schema     = get_schema(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            s1_queries = build_step1_queries(phrase, keywords, schema)
        except Exception:
            schema     = {'relationships': []}
            s1_queries = build_step1_queries(phrase, keywords, schema)

        results = run_queries_until_relevant(s1_queries, question)
        if results:
            print("  [Step 1b] Schema-guided hit")
            return generate_answer(question, results), format_kg_triples(results)

        # ── STEP 2: Per-keyword directed traversal ──
        print("  [Step 2] Per-keyword directed queries...")
        s2_queries = build_step2_queries(keywords)
        results    = run_queries_until_relevant(s2_queries, question)
        if results:
            print("  [Step 2] Keyword traversal hit")
            return generate_answer(question, results), format_kg_triples(results)

        # ── STEP 3: Loose fallback + LLM synonym expansion ──
        print("  [Step 3] Loose fallback + LLM synonyms...")
        try:
            expanded_q   = llm.generate(
                f"Give 3 synonyms or closely related scientific terms for: {question}"
            ).strip()
            all_keywords = extract_keywords(question + " " + expanded_q)
        except Exception:
            all_keywords = keywords

        s3_queries = build_step3_queries(all_keywords)
        results    = run_queries_until_relevant(s3_queries, question)
        if results:
            print("  [Step 3] Fallback hit")
            return generate_answer(question, results), format_kg_triples(results)

        print("  [All steps exhausted] No relevant data in KG")
        return NO_INFO_MSG, []

    except Exception as e:
        return f"Error: {str(e)}", []
from nim_client import NvidiaLLMClient
from schema_discovery import SchemaDiscovery
import re

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
LLM_API_KEY = "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"  # REPLACE WITH YOUR ACTUAL NVIDIA API KEY
# LLM_MODEL = "meta/llama-3.2-1b-instruct"
LLM_MODEL = "Qwen/Qwen2.5-1B-Instruct"  # REPLACE WITH YOUR DESIRED MODEL
llm = NvidiaLLMClient(
    api_key=LLM_API_KEY,
    model=LLM_MODEL
)
# from qwen_client import QwenLLMClient

# llm = QwenLLMClient()

# ─────────────────────────────────────────────────────────────
# Cache schema so we don't query Neo4j every single time
# ─────────────────────────────────────────────────────────────
_cached_schema = None

def get_schema(neo4j_uri, neo4j_user, neo4j_password):
    global _cached_schema
    if _cached_schema is None:
        print("  📋 Loading schema from Neo4j...")
        schema_client = SchemaDiscovery(neo4j_uri, neo4j_user, neo4j_password)
        _cached_schema = schema_client.get_schema()
        schema_client.close()
        print(f"  ✅ Schema loaded: {len(_cached_schema['relationships'])} relationship types")
    return _cached_schema


def validate_cypher(cypher, allowed_relationships):
    if not cypher or len(cypher.strip()) < 10:
        return False, "Query is too short"
    
    # Check for basic clauses
    up = cypher.upper()
    if 'MATCH' not in up or 'RETURN' not in up:
        return False, "Missing MATCH or RETURN"
    
    # Check if the query is actually just conversational text
    if len(cypher.split()) < 4:
        return False, "Query is not a complete statement"

    # Strict check on labels to prevent Neo4j property/label errors
    # Llama-1b loves to invent labels like :DuckCurve
    found_labels = re.findall(r':(\w+)', cypher)
    for lbl in found_labels:
        # If it's not our base label and not a relationship, it's a hallucination
        if lbl != "Entity" and lbl not in allowed_relationships:
             return False, f"Illegal label used: :{lbl}"

    return True, "OK"


def generate_cypher(question, neo4j_uri, neo4j_user, neo4j_password, max_retries=2):
    schema = get_schema(neo4j_uri, neo4j_user, neo4j_password)
    rel_list = ", ".join(schema['relationships'])

    prompt = f"""Task: Write a Neo4j Cypher query for this question.
Relationships: [{rel_list}]
Label: Use ONLY :Entity

Example:
Question: What is agroforestry?
Query: MATCH (n:Entity)-[r]->(m:Entity) WHERE toLower(n.name) CONTAINS 'agroforestry' RETURN n.name AS entity, type(r) AS relationship, m.name AS result LIMIT 5

Question: {question}
Query:"""

    for attempt in range(max_retries):
        raw_response = llm.generate(prompt, max_tokens=200)
        query = clean_cypher_output(raw_response)
        
        is_valid, _ = validate_cypher(query, schema['relationships'])
        if is_valid:
            return query
            
    # If it still fails, return a basic keyword search
    words = re.findall(r'\b[a-zA-Z]{4,}\b', question.lower())
         
    stop_words = {
    'what','is','are','why','how','does','do','the','this','that',
    'here','heres','there','their','about','which','when'
    }
    keyword = [w for w in words if w not in stop_words]
    return f"MATCH (n:Entity)-[r]->(m:Entity) WHERE toLower(n.name) CONTAINS '{keyword}' RETURN n.name, type(r), m.name LIMIT 5"


def clean_cypher_output(raw_text):
    raw_text = re.sub(r'```(?:cypher)?(.*?)```', r'\1', raw_text, flags=re.DOTALL | re.IGNORECASE)

    # Extract only MATCH...LIMIT
    match = re.search(r'(MATCH[\s\S]*?LIMIT\s+\d+)', raw_text, re.IGNORECASE)
    if not match:
        return ""

    query = match.group(1)

    # 🚨 REMOVE dangerous quotes
    query = re.sub(r"[“”]", "'", query)

    # 🚨 REMOVE words like "here's"
    query = re.sub(r"\'s\b", "", query)

    # 🚨 ESCAPE single quotes properly
    # query = query.replace("'", "\\'")

    return query.strip()


def generate_fallback_cypher(question, relationships):
    # Expanded list of words to ignore so we don't search for "here", "is", "question"
    stop_words = {
        'what', 'is', 'are', 'how', 'why', 'does', 'do', 'the', 'a', 'an', 
        'in', 'for', 'to', 'of', 'and', 'or', 'with', 'about', 'can', 'will', 
        'was', 'were', 'related', 'here', 'heres', 'improved', 'rewritten', 'question'
    }

    # Clean the question: remove non-alphanumeric and split
    clean_q = re.sub(r'[^a-zA-Z0-9\s]', '', question)
    words = clean_q.lower().split()
    keywords = [w for w in words if w not in stop_words and len(w) > 3]

    where_clauses = []
    if keywords:
        for keyword in keywords[:3]:
            # Use double backslashes to escape any potential internal quotes
            safe_keyword = keyword.replace("'", "\\'")
            where_clauses.append(f"toLower(n.name) CONTAINS '{safe_keyword}'")

    where_clause_str = " OR ".join(where_clauses) if where_clauses else "true"

    fallback = f"MATCH (n:Entity)-[r]->(m:Entity) WHERE {where_clause_str} " \
               f"RETURN n.name AS entity, type(r) AS relationship_type, m.name AS related_entity LIMIT 10"
    
    return ' '.join(fallback.split())
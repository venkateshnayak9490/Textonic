"""
Centralized configuration for KG + Text RAG hybrid system.
All paths, API keys, model names, and hyperparameters in one place.
Windows and Linux compatible.
"""

import os
from pathlib import Path

# ============================================================================
# WORKSPACE PATHS (Auto-detected, Windows & Linux compatible)
# ============================================================================

# Root directory of the workspace
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]  # src/configs/config.py -> KG_Implementation

# KG pipeline directories (archived)
KG_DATA_PREP = WORKSPACE_ROOT / "archived" / "Data_preparation"
KG_ENTITY_RELATION = WORKSPACE_ROOT / "archived" / "Entity_relation"
KG_OUTPUT = WORKSPACE_ROOT / "data" / "kg"  # New location for KG outputs

# Text RAG directories
RAG_ROOT = WORKSPACE_ROOT / "src" / "rag"  # Renamed from inlp_rag
RAG_SRC = RAG_ROOT / "src"
RAG_FILTERED = RAG_ROOT / "filtered"

# Hybrid system directories
HYBRID_DIR = WORKSPACE_ROOT / "src" / "hybrid"
CONFIGS_DIR = WORKSPACE_ROOT / "src" / "configs"
EVAL_DIR = WORKSPACE_ROOT / "eval"

# ============================================================================
# KG INPUT/OUTPUT FILES
# ============================================================================

# KG inputs
CHUNKS_JSON = WORKSPACE_ROOT / "data" / "text_chunks" / "chunks.json"
CANONICAL_SCHEMA = KG_OUTPUT / "canonical_schema.json"

# KG outputs
ENTITIES_JSON = KG_OUTPUT / "entities.json"
TRIPLES_JSON = KG_OUTPUT / "triples_2.json"
LABEL_RELATION_MATRIX = KG_OUTPUT / "label_relation_matrix.json"

# ============================================================================
# RAG INPUT/OUTPUT FILES
# ============================================================================

RAG_TEST_DATASET = WORKSPACE_ROOT / "climate_hallucination_dataset_test_230_v2.json"
RAG_OUTPUTS_DIR = RAG_ROOT / "outputs"

# ============================================================================
# HYBRID SYSTEM SETTINGS
# ============================================================================

# Fusion strategy
FUSION_ALPHA = 0.4  # Weight for KG score in [score_kg * alpha + score_rag * (1-alpha)]
KG_TOP_K = 6        # Number of KG facts to retrieve
RAG_TOP_K = 6       # Number of text chunks to retrieve
RERANKER_TOP_K = 4  # Final reranked results

# ============================================================================
# LLM & MODEL SETTINGS
# ============================================================================

# Model selection
PREFERRED_MODEL = "qwen"  # "llama" or "qwen"
LLAMA_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
QWEN_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# Hugging Face token (set via environment variable)
HF_TOKEN = os.getenv("HF_TOKEN", "")  # YOU MUST SET THIS IN YOUR ENVIRONMENT

# LLM Generation settings
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 120
LLM_DO_SAMPLE = False

# ============================================================================
# RAG EMBEDDING & RETRIEVAL SETTINGS
# ============================================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Robust embeddings for chunks
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-encoder for reranking

# FAISS index settings
FAISS_INDEX_TYPE = "IndexFlatL2"  # L2 distance (can use IndexHNSW for faster search)

# ============================================================================
# ENTITY LINKING & ENTITY MATCHING SETTINGS
# ============================================================================

# For KG entity linking to queries
ENTITY_SIMILARITY_THRESHOLD = 0.7  # Cosine/string similarity threshold
USE_FUZZY_MATCHING = True          # Enable approximate entity matching

# ============================================================================
# HYBRID ANSWER GENERATION PROMPT TEMPLATES
# ============================================================================

SYSTEM_PROMPT = """\
You are an expert climate science assistant. Answer questions based on the provided
knowledge graph facts and text evidence. Be precise, cite sources, and express uncertainty
if evidence is insufficient.\
"""

HYBRID_PROMPT_TEMPLATE = """\
Use the following information to answer the question accurately and concisely.

Knowledge Graph Facts:
{kg_context}

Text Evidence from Documents:
{text_context}

Question:
{question}

Answer requirements:
- Keep the answer to 3-4 sentences.
- Do not include inline citation tags in the middle of sentences.
- Add one final line as: Sources: TEXT:1, KG:entity_name
- If evidence is weak, clearly say uncertainty in one short sentence.
\
"""

# ============================================================================
# LOGGING & DEBUG SETTINGS
# ============================================================================

VERBOSE = True  # Print progress
LOG_LEVEL = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

EVAL_METRICS = ["bleu", "rouge", "citation_precision", "hallucination_rate"]
TEST_SAMPLE_SIZE = 50  # Number of test cases to evaluate

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_paths_exist():
    """Create necessary directories if they don't exist."""
    for path in [RAG_OUTPUTS_DIR, EVAL_DIR, KG_OUTPUT, HYBRID_DIR]:
        path.mkdir(parents=True, exist_ok=True)

def validate_kb_files():
    """Check if all required KG files exist."""
    required = [CHUNKS_JSON, CANONICAL_SCHEMA, ENTITIES_JSON, TRIPLES_JSON]
    missing = [f for f in required if not f.exists()]
    if missing:
        print(f"WARNING: Missing KG files: {missing}")
        return False
    return True

def validate_rag_files():
    """Check if RAG chunked PDFs exist."""
    if not RAG_FILTERED.exists():
        print(f"WARNING: RAG filtered PDFs not found at {RAG_FILTERED}")
        return False
    pdfs = list(RAG_FILTERED.glob("*.pdf"))
    if not pdfs:
        print(f"WARNING: No PDFs found in {RAG_FILTERED}")
        return False
    return True

def validate_test_dataset_file():
    """Check if configured test dataset file exists."""
    if not RAG_TEST_DATASET.exists():
        print(f"WARNING: Test dataset not found at {RAG_TEST_DATASET}")
        return False
    return True

def get_model_id(model_name=None):
    """Return a Hugging Face model ID for alias or direct model name input."""
    model_name = (model_name or PREFERRED_MODEL).strip()
    models = {
        "llama": LLAMA_MODEL_ID,
        "qwen": QWEN_MODEL_ID
    }

    # Alias support (llama/qwen)
    if model_name.lower() in models:
        return models[model_name.lower()]

    # Direct model ID support, e.g. "meta-llama/..." or "Qwen/..."
    if "/" in model_name:
        return model_name

    # Safe fallback
    return QWEN_MODEL_ID

def print_config():
    """Print out configuration summary."""
    print("\n" + "="*70)
    print("HYBRID KG + TEXT RAG CONFIGURATION")
    print("="*70)
    print(f"Workspace root     : {WORKSPACE_ROOT}")
    print(f"KG outputs dir     : {KG_OUTPUT}")
    print(f"RAG root dir       : {RAG_ROOT}")
    print(f"Hybrid module dir  : {HYBRID_DIR}")
    print(f"\nFusion settings:")
    print(f"  KG weight        : {FUSION_ALPHA}")
    print(f"  KG top-k         : {KG_TOP_K}")
    print(f"  RAG top-k        : {RAG_TOP_K}")
    print(f"\nModel settings:")
    print(f"  Preferred model  : {PREFERRED_MODEL}")
    print(f"  Model ID         : {get_model_id()}")
    print(f"  Embedding model  : {EMBEDDING_MODEL}")
    print(f"  Reranker model   : {RERANKER_MODEL}")
    print("="*70 + "\n")

if __name__ == "__main__":
    ensure_paths_exist()
    print_config()
    print(f"KG files valid: {validate_kb_files()}")
    print(f"RAG files valid: {validate_rag_files()}")
    print(f"Test dataset valid: {validate_test_dataset_file()}")

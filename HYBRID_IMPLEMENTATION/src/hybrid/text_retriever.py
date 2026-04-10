"""
Text Retriever: Load and query text RAG index with GPU support.
Wrapper around FAISS + embeddings from the inlp_rag system.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
import json
import numpy as np
import torch
import faiss

# Add parent directories to path for compatibility when running file directly.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))

try:
    from src.configs.config import (
        RAG_FILTERED, RAG_SRC, RAG_OUTPUTS_DIR, EMBEDDING_MODEL, RERANKER_MODEL,
        VERBOSE
    )
except ImportError:
    from configs.config import (
        RAG_FILTERED, RAG_SRC, RAG_OUTPUTS_DIR, EMBEDDING_MODEL, RERANKER_MODEL,
        VERBOSE
    )

try:
    # Prefer package-safe imports for normal execution from workspace root.
    from src.rag.src.retrieval import build_faiss_index, retrieve_top_k
    from src.rag.src.rerank import rerank
    from src.rag.src.pdf_processing import load_and_chunk_pdfs
except ImportError as e:
    try:
        # Legacy fallback for older path setups.
        from rag.src.retrieval import build_faiss_index, retrieve_top_k
        from rag.src.rerank import rerank
        from rag.src.pdf_processing import load_and_chunk_pdfs
    except ImportError:
        print(f"WARNING: RAG modules not found. Text retrieval will be limited. Error: {e}")
        build_faiss_index = None
        retrieve_top_k = None
        rerank = None
        load_and_chunk_pdfs = None


class TextRetriever:
    """Load and query text chunks via FAISS with GPU acceleration."""

    def __init__(
        self,
        pdf_folder: str = None,
        use_reranker: bool = True,
        use_gpu: bool = True,
        force_rebuild: bool = False
    ):
        """
        Initialize text retriever with GPU support.
        
        Args:
            pdf_folder: Path to folder with PDF files (optional, for reloading)
            use_reranker: If True, use cross-encoder reranker
            use_gpu: If True, use GPU acceleration (default: True)
            force_rebuild: If True, ignore cache and rebuild text index
        """
        self.pdf_folder = pdf_folder or str(RAG_FILTERED)
        self.use_reranker = use_reranker
        self.use_gpu = use_gpu and self._quick_cuda_available()
        self.force_rebuild = force_rebuild
        
        self.chunks = []
        self.index = None
        self.embedding_model = None
        self._loaded = False  # Lazy loading flag
        self._disabled = False
        self._disable_reason = ""
        self.cache_dir = Path(RAG_OUTPUTS_DIR)
        self.chunks_cache_path = self.cache_dir / "text_chunks_cache.json"
        self.index_cache_path = self.cache_dir / "text_index.faiss"
        self.meta_cache_path = self.cache_dir / "text_index_meta.json"

    def _quick_cuda_available(self) -> bool:
        """Fast, non-blocking CUDA presence check for text retrieval."""
        if os.environ.get("HYBRID_FORCE_CPU", "0") == "1":
            return False
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible == "":
            return False
        if cuda_visible and cuda_visible not in {"-1", "none", "None"}:
            return True
        return any(Path("/dev").glob("nvidia*"))

    def _is_disk_quota_error(self, exc: Exception) -> bool:
        """Detect filesystem quota errors from OS or wrapped library exceptions."""
        msg = str(exc).lower()
        return getattr(exc, "errno", None) == 122 or "disk quota exceeded" in msg

    def _disable_retrieval(self, reason: str):
        """Disable text retrieval for this retriever instance after hard failures."""
        self._disabled = True
        self._disable_reason = reason
        self.chunks = []
        self.index = None
        self.embedding_model = None
        print(f"WARNING: Text retrieval disabled for this run: {reason}")

    def _current_pdf_state(self) -> Dict:
        """Return deterministic metadata for source PDFs to validate cache."""
        pdf_dir = Path(self.pdf_folder)
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        files = []
        for p in pdf_files:
            stat = p.stat()
            files.append({
                "name": p.name,
                "size": stat.st_size,
                "mtime": int(stat.st_mtime)
            })
        return {
            "pdf_folder": str(pdf_dir.resolve()),
            "file_count": len(files),
            "files": files,
            "embedding_model": EMBEDDING_MODEL
        }

    def _load_embedding_model(self):
        """Load query embedding model (used with cached or rebuilt FAISS index)."""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        if self.use_gpu:
            model = model.cuda()
            if VERBOSE:
                print("✓ Embedding model loaded on GPU")
        return model

    def _try_load_cache(self) -> bool:
        """Load cached chunks + FAISS index when source PDFs are unchanged."""
        if self._disabled:
            return False

        if not (self.chunks_cache_path.exists() and self.index_cache_path.exists() and self.meta_cache_path.exists()):
            return False

        try:
            with open(self.meta_cache_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if meta.get("pdf_state") != self._current_pdf_state():
                if VERBOSE:
                    print("Text cache invalidated (PDFs changed). Rebuilding index...")
                return False

            with open(self.chunks_cache_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)

            self.index = faiss.read_index(str(self.index_cache_path))
            self.embedding_model = self._load_embedding_model()

            if VERBOSE:
                print(f"Loaded cached text index: {len(self.chunks)} chunks")

            return True

        except Exception as e:
            if self._is_disk_quota_error(e):
                self._disable_retrieval(f"Disk quota exceeded while loading cache/model: {e}")
                return False
            print(f"Warning: failed to load text cache ({e}). Rebuilding index...")
            self.chunks = []
            self.index = None
            self.embedding_model = None
            return False

    def _save_cache(self):
        """Persist chunks and FAISS index to avoid rebuilding on each run."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            with open(self.chunks_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=True)

            faiss.write_index(self.index, str(self.index_cache_path))

            meta = {
                "pdf_state": self._current_pdf_state()
            }
            with open(self.meta_cache_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=True)

            if VERBOSE:
                print(f"Saved text cache to {self.cache_dir}")

        except Exception as e:
            print(f"Warning: failed to save text cache ({e})")
    
    def _load_text_chunks(self):
        """Load and chunk PDFs, build FAISS index with GPU acceleration."""
        if self._disabled:
            return

        if not load_and_chunk_pdfs:
            print("WARNING: RAG PDF processing not available. Skipping text index build.")
            return

        if (not self.force_rebuild) and self._try_load_cache():
            return

        if self.force_rebuild and VERBOSE:
            print("Force rebuild enabled. Ignoring cached text index...")
        
        try:
            if VERBOSE:
                print(f"Loading PDFs from {self.pdf_folder}...")
            
            self.chunks = load_and_chunk_pdfs(self.pdf_folder)
            
            if VERBOSE:
                print(f"  Loaded {len(self.chunks)} text chunks")
            
            if VERBOSE:
                gpu_str = "(GPU mode)" if self.use_gpu else "(CPU mode)"
                print(f"Building FAISS index with {EMBEDDING_MODEL} {gpu_str}...")
            
            self.index, self.embedding_model = build_faiss_index(
                self.chunks,
                use_gpu=self.use_gpu
            )

            self._save_cache()

            if VERBOSE:
                print(f"  FAISS index built with {len(self.chunks)} vectors")

        except Exception as e:
            if self._is_disk_quota_error(e):
                self._disable_retrieval(f"Disk quota exceeded while building text index: {e}")
                return
            print(f"Error loading text chunks: {e}")
            self.chunks = []
            self.index = None

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_reranker: bool = None
    ) -> List[Dict]:
        """
        Retrieve top-k text chunks for a query with GPU acceleration.

        Args:
            query: Search query text
            top_k: Number of results to return
            use_reranker: Override class setting for this query

        Returns:
            List of chunk dicts with text, source, score
        """
        # Validate input
        if not query or not isinstance(query, str) or not query.strip():
            return []
        
        # Lazy load PDFs on first retrieval
        if not self._loaded:
            self._load_text_chunks()
            self._loaded = True

        if self._disabled:
            return []

        if self.index is None or not self.chunks:
            print("WARNING: Text index not available. Returning empty results.")
            return []

        if not retrieve_top_k:
            print("WARNING: RAG retrieval function not available.")
            return []

        # Retrieve more chunks if using reranker
        retrieve_k = top_k * 4 if use_reranker or self.use_reranker else top_k

        # Get initial retrieval with GPU support
        try:
            retrieved = retrieve_top_k(
                query,
                self.index,
                self.chunks,
                self.embedding_model,
                k=min(retrieve_k, len(self.chunks)),
                use_gpu=self.use_gpu
            )
        except Exception as e:
            if self._is_disk_quota_error(e):
                self._disable_retrieval(f"Disk quota exceeded during retrieval: {e}")
                return []
            raise

        # Optionally rerank with GPU support
        use_reranker = use_reranker if use_reranker is not None else self.use_reranker
        if use_reranker and rerank:
            retrieved = rerank(
                query,
                retrieved,
                top_k=top_k,
                use_gpu=self.use_gpu
            )
        else:
            retrieved = retrieved[:top_k]

        return retrieved

    def format_text_context(self, chunks: List[Dict]) -> str:
        """
        Format retrieved text chunks into readable context.

        Args:
            chunks: List of chunk dicts from retrieve()

        Returns:
            Formatted text for insertion into prompt
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('text', '')
            source = chunk.get('source', 'unknown')
            context_parts.append(f"[Chunk {i} from {source}]")
            context_parts.append(text)
            context_parts.append("")

        return "\n".join(context_parts) if context_parts else "[No text chunks retrieved]"


# Example usage
if __name__ == "__main__":
    try:
        retriever = TextRetriever(use_reranker=True)

        # Test query
        query = "What is the impact of greenhouse gases on climate?"
        results = retriever.retrieve(query, top_k=3)

        print(f"\nText Retrieval Results: {len(results)} chunks")

        context = retriever.format_text_context(results)
        print("\nFormatted Context:")
        print(context)

    except Exception as e:
        print(f"Error in text retriever demo: {e}")
           
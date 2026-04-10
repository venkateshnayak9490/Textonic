"""
Hybrid KG + Text RAG Pipeline: Orchestrates entity linking, retrieval, fusion, and answer generation.
Multi-GPU optimized for efficient execution on 2+ GPUs.
"""

import sys
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import re
from concurrent.futures import ThreadPoolExecutor
import torch

# Add parent directories to path for compatibility when running file directly.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "rag"))


def _configure_model_cache_paths() -> None:
    """Set cache paths for HF/Transformers/SentenceTransformers before model imports."""
    cache_root = Path(
        os.environ.get("HYBRID_CACHE_ROOT")
        or os.environ.get("XDG_CACHE_HOME")
        or (Path(tempfile.gettempdir()) / "hybrid_hf_cache")
    )

    hf_home = cache_root / "hf_home"
    transformers_cache = cache_root / "transformers"
    sentence_cache = cache_root / "sentence_transformers"

    # Respect explicit user-provided env vars; only set defaults when missing.
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(sentence_cache))

    # Best effort directory creation to avoid repeated cache initialization errors.
    try:
        Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["SENTENCE_TRANSFORMERS_HOME"]).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"WARNING: Could not create model cache directories: {e}")


_configure_model_cache_paths()


def _quick_cuda_available() -> bool:
    """Fast, non-blocking CUDA presence check for startup decisions."""
    if os.environ.get("HYBRID_FORCE_CPU", "0") == "1":
        return False
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible == "":
        return False
    if cuda_visible and cuda_visible not in {"-1", "none", "None"}:
        return True
    return any(Path("/dev").glob("nvidia*"))

try:
    from src.configs.config import (
        FUSION_ALPHA, KG_TOP_K, RAG_TOP_K, RERANKER_TOP_K,
        SYSTEM_PROMPT, HYBRID_PROMPT_TEMPLATE,
        LLM_TEMPERATURE, LLM_DO_SAMPLE, LLM_MAX_TOKENS,
        VERBOSE, HF_TOKEN, get_model_id, PREFERRED_MODEL
    )
    from src.hybrid.kg_retriever import KGRetriever
    from src.hybrid.text_retriever import TextRetriever
except ImportError:
    from configs.config import (
        FUSION_ALPHA, KG_TOP_K, RAG_TOP_K, RERANKER_TOP_K,
        SYSTEM_PROMPT, HYBRID_PROMPT_TEMPLATE,
        LLM_TEMPERATURE, LLM_DO_SAMPLE, LLM_MAX_TOKENS,
        VERBOSE, HF_TOKEN, get_model_id, PREFERRED_MODEL
    )
    from hybrid.kg_retriever import KGRetriever
    from hybrid.text_retriever import TextRetriever

try:
    # Prefer package-safe imports for normal execution from workspace root.
    from src.rag.src.llm import load_model
except ImportError as e:
    try:
        # Legacy fallback for older path setups.
        from rag.src.llm import load_model
    except ImportError:
        print(f"WARNING: RAG LLM module not found. Limited answer generation. Error: {e}")
        load_model = None


class HybridPipeline:
    """End-to-end hybrid KG + Text RAG pipeline with multi-GPU support."""

    def __init__(
        self,
        use_kg: bool = True,
        use_rag: bool = True,
        use_reranker: bool = True,
        model_name: Optional[str] = None,
        use_gpu: bool = True,
        force_rebuild_text_index: bool = False
    ):
        """
        Initialize hybrid pipeline with GPU optimization.
        
        Args:
            use_kg: Enable knowledge graph retrieval
            use_rag: Enable text RAG retrieval
            use_reranker: Enable cross-encoder reranking
            model_name: LLM model name ("llama", "qwen", or model ID)
            use_gpu: Enable GPU acceleration (default: True if CUDA available)
            force_rebuild_text_index: If True, rebuild text index and skip cache
        """
        self.use_kg = use_kg
        self.use_rag = use_rag
        self.use_reranker = use_reranker
        self.model_name = model_name or PREFERRED_MODEL
        self.use_gpu = use_gpu and _quick_cuda_available()
        self.force_rebuild_text_index = force_rebuild_text_index
        
        # Print GPU status
        if self.use_gpu:
            print(f"\n{'='*70}")
            print(f"GPU ACCELERATION ENABLED")
            print("  CUDA quick check passed")
            print(f"{'='*70}\n")
        else:
            print("\nGPU acceleration disabled or CUDA not available. Using CPU.\n")
        
        self.kg_retriever = None
        self.text_retriever = None
        self.tokenizer = None
        self.model = None
        
        self._initialize_retrievers()
        self._initialize_llm()
    
    def _initialize_retrievers(self):
        """Load KG and text retrievers with GPU support."""
        if self.use_kg:
            if VERBOSE:
                print("Initializing KG retriever...")
            try:
                self.kg_retriever = KGRetriever()
            except Exception as e:
                print(f"Error loading KG retriever: {e}")
                self.use_kg = False
        
        if self.use_rag:
            if VERBOSE:
                print("Initializing text retriever...")
            try:
                self.text_retriever = TextRetriever(
                    use_reranker=self.use_reranker,
                    use_gpu=self.use_gpu,
                    force_rebuild=self.force_rebuild_text_index
                )
            except Exception as e:
                print(f"Error loading text retriever: {e}")
                self.use_rag = False
    
    def _initialize_llm(self):
        """Load LLM for answer generation with multi-GPU support."""
        if not load_model:
            print("WARNING: LLM loading not available.")
            return
        
        try:
            if VERBOSE:
                print(f"Loading LLM: {self.model_name}...")
            
            model_id = get_model_id(self.model_name)
            self.tokenizer, self.model = load_model(
                model_id, 
                token=HF_TOKEN or None,
                use_multi_gpu=self.use_gpu
            )
            
            if VERBOSE:
                print(f"  LLM loaded successfully")
        
        except Exception as e:
            print(f"Error loading LLM: {e}")
            self.tokenizer = None
            self.model = None
    
    def retrieve_kg(self, query: str, top_k: Optional[int] = None) -> Dict:
        """Retrieve from knowledge graph."""
        if not self.use_kg or not self.kg_retriever:
            return {'entities': [], 'triples': [], 'num_entities': 0, 'num_triples': 0, 'query': query}
        
        top_k = top_k or KG_TOP_K
        kg_result = self.kg_retriever.query_kg(query, top_k=top_k)
        kg_result['query'] = query  # Ensure query key is always present
        return kg_result
    
    def retrieve_text(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve from text RAG with GPU acceleration."""
        if not self.use_rag or not self.text_retriever:
            return []
        
        top_k = top_k or RAG_TOP_K
        try:
            return self.text_retriever.retrieve(
                query,
                top_k=top_k,
                use_reranker=self.use_reranker
            )
        except Exception as e:
            print(f"WARNING: text retrieval failed; disabling RAG for this run. Error: {e}")
            self.use_rag = False
            return []

    def _retrieve_both_sources(
        self,
        question: str,
        retrieval_mode: str,
        kg_top_k: Optional[int] = None,
        rag_top_k: Optional[int] = None,
    ) -> tuple:
        """Retrieve KG and text evidence using the requested ordering strategy."""
        mode = (retrieval_mode or "seq_kg_first").strip().lower()

        if mode == "parallel":
            with ThreadPoolExecutor(max_workers=2) as executor:
                kg_future = executor.submit(self.retrieve_kg, question, kg_top_k) if self.use_kg else None
                text_future = executor.submit(self.retrieve_text, question, rag_top_k) if self.use_rag else None
                kg_results = kg_future.result() if kg_future else {'entities': [], 'triples': [], 'num_entities': 0, 'num_triples': 0, 'query': question}
                text_results = text_future.result() if text_future else []
            return kg_results, text_results

        if mode == "seq_rag_first":
            text_results = self.retrieve_text(question, rag_top_k) if self.use_rag else []
            kg_results = self.retrieve_kg(question, kg_top_k) if self.use_kg else {'entities': [], 'triples': [], 'num_entities': 0, 'num_triples': 0, 'query': question}
            return kg_results, text_results

        # Default: sequential KG first.
        kg_results = self.retrieve_kg(question, kg_top_k) if self.use_kg else {'entities': [], 'triples': [], 'num_entities': 0, 'num_triples': 0, 'query': question}
        text_results = self.retrieve_text(question, rag_top_k) if self.use_rag else []
        return kg_results, text_results

    def _tokenize(self, text: str) -> List[str]:
        """Simple lowercase tokenizer for lexical overlap scoring."""
        if not text:
            return []
        return re.findall(r"[a-z0-9]+", text.lower())

    def _lexical_score(self, query: str, text: str) -> float:
        """Compute overlap ratio between query tokens and text tokens."""
        q_tokens = set(self._tokenize(query))
        t_tokens = set(self._tokenize(text))
        if not q_tokens or not t_tokens:
            return 0.0
        return len(q_tokens.intersection(t_tokens)) / max(1, len(q_tokens))

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-max normalize scores to [0,1] with stable fallback."""
        if not scores:
            return []

        min_s = min(scores)
        max_s = max(scores)

        if max_s == min_s:
            return [1.0 for _ in scores]

        return [(s - min_s) / (max_s - min_s) for s in scores]

    def _ensure_source_diversity(self, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Prefer at least one KG and one text evidence item in final fused selection when available."""
        if not candidates:
            return []

        selected = candidates[:top_k]
        has_kg = any(c.get('source') == 'kg' for c in selected)
        has_text = any(c.get('source') == 'text' for c in selected)

        if has_kg and has_text:
            return selected

        # Add missing source types without overwriting existing candidates
        if not has_kg and len(selected) < top_k:
            kg_candidate = next((c for c in candidates if c.get('source') == 'kg'), None)
            if kg_candidate is not None and kg_candidate not in selected:
                selected.append(kg_candidate)

        if not has_text and len(selected) < top_k:
            text_candidate = next((c for c in candidates if c.get('source') == 'text'), None)
            if text_candidate is not None and text_candidate not in selected:
                selected.append(text_candidate)

        return selected[:top_k]
    
    def fuse_results(
        self,
        kg_results: Dict,
        text_results: List[Dict],
        alpha: Optional[float] = None
    ) -> Dict:
        """
        Fuse KG and text results with weighted score combination.
        
        Args:
            kg_results: Output from retrieve_kg()
            text_results: Output from retrieve_text()
            alpha: Weight for KG results [0, 1]
        
        Returns:
            Fused results with normalized scores
        """
        alpha = alpha if alpha is not None else FUSION_ALPHA

        kg_entities = kg_results.get('entities', [])
        kg_triples = kg_results.get('triples', [])

        # Build KG candidates with lexical scores (entity + triple text)
        kg_candidates: List[Dict] = []
        for i, entity in enumerate(kg_entities):
            entity_text = entity.get('text', '')
            label = entity.get('canonical_label', '')
            score = self._lexical_score(query=kg_results.get('query', ''), text=f"{entity_text} {label}")
            if score == 0.0:
                score = 1.0 / (i + 1)
            kg_candidates.append({
                'source': 'kg',
                'kind': 'entity',
                'raw': entity,
                'display': f"{entity_text} ({label})",
                'score_raw': score
            })

        for i, triple in enumerate(kg_triples):
            subj = triple.get('subject', '')
            rel = triple.get('relation', '')
            obj = triple.get('object', '')
            triple_text = f"{subj} {rel} {obj}"
            score = self._lexical_score(query=kg_results.get('query', ''), text=triple_text)
            if score == 0.0:
                score = 1.0 / (i + 1)
            kg_candidates.append({
                'source': 'kg',
                'kind': 'triple',
                'raw': triple,
                'display': f"{subj} --[{rel}]--> {obj}",
                'score_raw': score
            })

        # Build text candidates with model score when available; rank fallback otherwise
        text_candidates: List[Dict] = []
        for i, chunk in enumerate(text_results):
            raw_score = chunk.get('score')
            if raw_score is None:
                raw_score = chunk.get('rerank_score')
            if raw_score is None:
                raw_score = 1.0 / (i + 1)

            text_candidates.append({
                'source': 'text',
                'kind': 'chunk',
                'raw': chunk,
                'display': chunk.get('text', '')[:300],
                'score_raw': float(raw_score)
            })

        kg_norm = self._normalize_scores([c['score_raw'] for c in kg_candidates])
        text_norm = self._normalize_scores([c['score_raw'] for c in text_candidates])

        for idx, c in enumerate(kg_candidates):
            c['score_norm'] = kg_norm[idx] if idx < len(kg_norm) else 0.0
            c['score_kg'] = c['score_norm']
            c['score_text'] = 0.0
            c['combined_score'] = alpha * c['score_kg'] + (1.0 - alpha) * c['score_text']

        for idx, c in enumerate(text_candidates):
            c['score_norm'] = text_norm[idx] if idx < len(text_norm) else 0.0
            c['score_kg'] = 0.0
            c['score_text'] = c['score_norm']
            c['combined_score'] = alpha * c['score_kg'] + (1.0 - alpha) * c['score_text']

        all_candidates = kg_candidates + text_candidates
        all_candidates = sorted(all_candidates, key=lambda x: x['combined_score'], reverse=True)

        min_score = 0.15
        strong_candidates = [c for c in all_candidates if c.get('combined_score', 0.0) >= min_score]
        pool = strong_candidates if strong_candidates else all_candidates
        top_fused = self._ensure_source_diversity(pool, RERANKER_TOP_K)

        fused = {
            'kg_entities': kg_entities,
            'kg_triples': kg_triples,
            'text_chunks': text_results,
            'fusion_alpha': alpha,
            'fused_evidence': top_fused
        }

        return fused
    
    def generate_answer(
        self,
        query: str,
        fused_results: Dict,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate answer using fused context.
        
        Args:
            query: User question
            fused_results: Fused results from fuse_results()
            max_tokens: Max tokens to generate
        
        Returns:
            Generated answer string
        """
        max_tokens = max_tokens or LLM_MAX_TOKENS

        # Format KG context
        kg_context = ""
        fused_evidence = fused_results.get('fused_evidence', [])

        # Prefer top fused KG evidence when available
        fused_kg_items = [e for e in fused_evidence if e.get('source') == 'kg']
        if fused_kg_items:
            kg_parts = [f"- {item.get('display', '')}" for item in fused_kg_items]
            kg_context = "\n".join(kg_parts)
        elif fused_results['kg_entities'] or fused_results['kg_triples']:
            kg_parts = []
            for entity in fused_results['kg_entities']:
                kg_parts.append(f"- {entity.get('text')} ({entity.get('canonical_label')})")
            for triple in fused_results['kg_triples']:
                kg_parts.append(
                    f"- {triple.get('subject')} --[{triple.get('relation')}]--> {triple.get('object')}"
                )
            kg_context = "\n".join(kg_parts)
        
        # Format text context
        text_context = ""
        fused_text_items = [e for e in fused_evidence if e.get('source') == 'text']
        if fused_text_items:
            text_parts = []
            for i, item in enumerate(fused_text_items, 1):
                text_parts.append(f"[Chunk {i}] {item.get('display', '')}...")
            text_context = "\n".join(text_parts)
        elif fused_results['text_chunks']:
            text_parts = []
            for i, chunk in enumerate(fused_results['text_chunks'], 1):
                text_parts.append(f"[Chunk {i}] {chunk.get('text', '')[:200]}...")
            text_context = "\n".join(text_parts)
        
        # Build prompt
        prompt = HYBRID_PROMPT_TEMPLATE.format(
            kg_context=kg_context or "[No knowledge graph matches]",
            text_context=text_context or "[No text matches]",
            question=query
        )
        
        # Generate using fused prompt (KG + text) with GPU acceleration
        if self.model and self.tokenizer:
            try:
                full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
                inputs = self.tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=min(getattr(self.tokenizer, "model_max_length", 4096), 4096)
                )
                
                # Handle both DataParallel and regular models
                generation_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
                device = next(generation_model.parameters()).device
                
                inputs = {k: v.to(device) for k, v in inputs.items()}

                gen_kwargs = {
                    "max_new_tokens": max_tokens,
                    "do_sample": LLM_DO_SAMPLE,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.eos_token_id
                }
                if LLM_DO_SAMPLE:
                    gen_kwargs["temperature"] = LLM_TEMPERATURE

                with torch.no_grad():
                    outputs = generation_model.generate(
                        **inputs,
                        **gen_kwargs
                    )

                # Decode only newly generated tokens to avoid prompt/system leakage.
                prompt_len = inputs["input_ids"].shape[1]
                generated_ids = outputs[0][prompt_len:]
                answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                answer = self._complete_truncated_answer(full_prompt, answer)
                answer = self._limit_generated_answer(answer, max_words=30)
                return self._postprocess_answer(answer)
            except Exception as e:
                print(f"Error generating answer: {e}")
                return self._postprocess_answer(self._extractive_fallback_answer(query, fused_results))
        else:
            return self._postprocess_answer(self._extractive_fallback_answer(query, fused_results))

    def _is_sentence_complete(self, text: str) -> bool:
        """Return True when text ends with sentence punctuation."""
        if not text:
            return False
        return bool(re.search(r"[.!?][\"'\)\]]*\s*$", text.strip()))

    def _complete_truncated_answer(
        self,
        base_prompt: str,
        partial_answer: str,
        max_rounds: int = 2,
        extra_tokens: int = 40,
    ) -> str:
        """Continue generation briefly when output appears cut mid-sentence."""
        if not partial_answer or self._is_sentence_complete(partial_answer):
            return partial_answer
        if not self.model or not self.tokenizer:
            return partial_answer

        completion_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        device = next(completion_model.parameters()).device
        current = partial_answer.strip()

        for _ in range(max_rounds):
            if self._is_sentence_complete(current):
                break

            continuation_prompt = (
                f"{base_prompt}\n\n"
                f"Current answer:\n{current}\n\n"
                "Continue the answer with only the next short fragment to complete the sentence."
            )
            inputs = self.tokenizer(
                continuation_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=min(getattr(self.tokenizer, "model_max_length", 4096), 4096),
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            gen_kwargs = {
                "max_new_tokens": extra_tokens,
                "do_sample": False,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.eos_token_id,
            }

            with torch.no_grad():
                outputs = completion_model.generate(**inputs, **gen_kwargs)

            prompt_len = inputs["input_ids"].shape[1]
            delta = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
            if not delta:
                break

            current = f"{current} {delta}".strip()
            if len(current.split()) > 90:
                break

        return current

    def _limit_generated_answer(self, answer: str, max_words: int = 30) -> str:
        """Limit generated answer by words while keeping full sentence boundaries."""
        if not answer:
            return answer

        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        selected: List[str] = []
        words_so_far = 0

        for sentence in sentences:
            s = sentence.strip()
            if not s:
                continue

            selected.append(s)
            words_so_far += len(s.split())

            # Include the full sentence that crosses the threshold.
            if words_so_far >= max_words:
                break

        return " ".join(selected).strip() if selected else answer.strip()

    def _extractive_fallback_answer(self, query: str, fused_results: Dict) -> str:
        """Build a deterministic answer directly from retrieved evidence when LLM is unavailable."""
        fused_evidence = fused_results.get('fused_evidence', []) or []
        kg_items = [e for e in fused_evidence if e.get('source') == 'kg']
        text_items = [e for e in fused_evidence if e.get('source') == 'text']

        # If no fused evidence, try raw buckets.
        if not kg_items and not text_items:
            kg_entities = fused_results.get('kg_entities', []) or []
            kg_triples = fused_results.get('kg_triples', []) or []
            text_chunks = fused_results.get('text_chunks', []) or []

            if not (kg_entities or kg_triples or text_chunks):
                return (
                    "Uncertainty: There is insufficient retrieved evidence to answer this question confidently. "
                    "Sources: TEXT:0, KG:0"
                )

            # Use raw evidence when fused list is unavailable.
            if kg_entities:
                kg_items = [{"display": f"{e.get('text', '')} ({e.get('canonical_label', '')})"} for e in kg_entities[:2]]
            if kg_triples:
                kg_items.extend(
                    [{"display": f"{t.get('subject', '')} --[{t.get('relation', '')}]--> {t.get('object', '')}"} for t in kg_triples[:2]]
                )
            if text_chunks:
                text_items = [{"display": c.get('text', '')[:220]} for c in text_chunks[:2]]

        candidates: List[Tuple[float, str, str]] = []
        for item in kg_items:
            line = (item.get('display') or '').strip()
            if line:
                candidates.append((self._lexical_score(query, line), line, 'kg'))
        for item in text_items:
            line = (item.get('display') or '').strip()
            if line:
                candidates.append((self._lexical_score(query, line), line, 'text'))

        # Prefer lines that lexically match the question to reduce generic/noisy fallback output.
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score = candidates[0][0] if candidates else 0.0

        # Avoid producing misleading fallback answers from low-relevance evidence.
        if best_score < 0.2:
            return (
                "Uncertainty: Retrieved evidence does not strongly match this question. "
                "Sources: TEXT:0, KG:0"
            )

        evidence_lines = [line for _, line, _ in candidates[:3]]

        if not evidence_lines:
            return (
                "Uncertainty: Retrieved evidence is too weak to produce a reliable answer. "
                "Sources: TEXT:0, KG:0"
            )

        # Keep concise, readable fallback while preserving question-specific content.
        summary_sentences: List[str] = []
        summary_sentences.append(f"For '{query}', retrieved evidence indicates: {evidence_lines[0]}.")
        if len(evidence_lines) > 1:
            summary_sentences.append(f"Additional supporting evidence: {evidence_lines[1]}.")
        if len(evidence_lines) > 2:
            summary_sentences.append(f"Further context: {evidence_lines[2]}.")

        top_kinds = {kind for _, _, kind in candidates[:3]}
        sources = []
        if 'text' in top_kinds:
            sources.append("TEXT:1")
        if 'kg' in top_kinds:
            sources.append("KG:entity_name")
        source_line = f"Sources: {', '.join(sources)}" if sources else "Sources: TEXT:0, KG:0"

        return " ".join(summary_sentences[:3]) + "\n" + source_line

    def _postprocess_answer(self, answer: str) -> str:
        """Keep answers concise and move inline citations to a final Sources line."""
        if not answer:
            return answer

        citations = re.findall(r"\[(TEXT:\d+|KG:[^\]]+)\]", answer)
        unique_citations = []
        for c in citations:
            if c not in unique_citations:
                unique_citations.append(c)

        cleaned = re.sub(r"\s*\[(TEXT:\d+|KG:[^\]]+)\]", "", answer)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Remove leaked system prompt/preamble fragments at the start.
        for _ in range(3):
            before = cleaned
            cleaned = re.sub(
                r"^you are an expert climate science assistant\.?\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(r"^be precise, cite sources, and express uncertainty if evidence is insufficient\.?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"^use proper grammar and punctuation\.?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"^use proper grammar and spelling\.?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"^\s*answer\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"^\s*rewritten\s+answer\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
            if cleaned == before:
                break

        # Remove leaked wrappers/instructions that appear inline before sentence split.
        cleaned = re.sub(r"\buse proper grammar and punctuation\.?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\buse proper grammar and spelling\.?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(?:answer|rewritten\s+answer)\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)

        # Remove common step-by-step / chain-of-thought style headings.
        cleaned = re.sub(r"^#+\s*", "", cleaned)
        cleaned = re.sub(r"^step\s*\d+\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bstep\s*\d+\s*:\s*[^.?!]*[.?!]?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^let(?:'|’)s\s+think\s+step\s+by\s+step\.?\s*", "", cleaned, flags=re.IGNORECASE)
        # Remove stray markdown heading markers that can appear inline.
        cleaned = re.sub(r"\s*##+\s*", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Fix common prompt leakage where instruction text is concatenated.
        cleaned = re.sub(
            r"(Sources:\s*TEXT:\d+,\s*KG:[^\s.,;:]+)\s*Write\s+",
            r"\1 ",
            cleaned,
            flags=re.IGNORECASE,
        )

        # Remove leaked instruction text if the model echoes prompt bullets.
        cleaned = re.sub(r"^-\s*Do not use phrases like[^.]*\.\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^-\s*Do not include inline citation tags[^.]*\.\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^-\s*Keep the answer to\s*3-4\s*sentences\.?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^-\s*Add one final line as:[^.]*\.\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^-\s*If evidence is weak[^.]*\.\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.lstrip("- ").strip()

        # Remove leaked imperative/instruction-style sentences anywhere in output.
        instruction_patterns = [
            r"^do not make up information\.?$",
            r"^do not make unsupported claims\.?$",
            r"^you are an expert climate science assistant\b.*$",
            r"^be precise, cite sources\b.*$",
            r"^use proper grammar and punctuation\.?$",
            r"^use proper grammar and spelling\.?$",
            r"^step\s*\d+\s*:.*$",
            r"^let(?:'|’)s\s+think\s+step\s+by\s+step\.?$",
            r"^answer requirements:?$",
            r"^answer:?$",
            r"^write a response\b.*$",
            r"^write a short summary\b.*$",
            r"^question:?\b.*$",
        ]
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        filtered: List[str] = []
        for sentence in sentences:
            s = sentence.strip()
            if not s:
                continue
            if any(re.match(pat, s, flags=re.IGNORECASE) for pat in instruction_patterns):
                continue
            filtered.append(s)

        # De-duplicate repeated sentence fragments often caused by generation loops.
        deduped: List[str] = []
        seen = set()
        for sentence in filtered:
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(sentence)

        concise = " ".join(deduped).strip()
        concise = self._trim_to_sentence_boundary(concise)

        if unique_citations:
            return f"{concise}\nSources: {', '.join(unique_citations)}"
        return concise

    def _trim_to_sentence_boundary(self, text: str) -> str:
        """Ensure output does not end mid-sentence."""
        if not text:
            return text

        t = text.strip()
        if t.endswith((".", "!", "?")):
            return t

        # Keep up to the last complete sentence-ending punctuation.
        last = max(t.rfind("."), t.rfind("!"), t.rfind("?"))
        if last >= 0:
            trimmed = t[: last + 1].strip()
            if trimmed:
                return trimmed

        # If model produced no sentence punctuation, cut at a natural clause boundary.
        clause_points = [t.rfind(","), t.rfind(";"), t.rfind(":")]
        for marker in [" because ", " which ", " that ", " and "]:
            idx = t.rfind(marker)
            if idx >= 20:
                clause_points.append(idx)

        clause_cut = max(clause_points)
        if clause_cut >= 20:
            candidate = t[:clause_cut].rstrip(" ,;:-")
            if len(candidate.split()) >= 6:
                return candidate + "."

        # Final fallback: keep a concise prefix instead of trailing partial clause.
        words = t.split()
        if len(words) > 28:
            return " ".join(words[:28]).rstrip(" ,;:-") + "."

        return t.rstrip(" ,;:-") + "."
    
    def query(self, question: str, retrieval_mode: str = "seq_kg_first") -> Dict:
        """
        Full pipeline: question -> retrieve -> fuse -> answer.
        
        Args:
            question: User question
            retrieval_mode: Retrieval strategy (seq_kg_first, seq_rag_first, parallel)
        
        Returns:
            Dict with 'answer', 'kg_results', 'text_results', 'fused_results'
        """
        # Validate input
        if not question or not isinstance(question, str) or not question.strip():
            return {
                'answer': 'Error: Empty or invalid question provided.',
                'kg_results': {'entities': [], 'triples': [], 'num_entities': 0, 'num_triples': 0, 'query': ''},
                'text_results': [],
                'fused_results': {'kg_entities': [], 'kg_triples': [], 'text_chunks': [], 'fused_evidence': []}
            }
        
        question = question.strip()
        
        if VERBOSE:
            print(f"\n{'='*70}")
            print(f"Processing question: {question}")
            print(f"Retrieval mode: {retrieval_mode}")
            print(f"{'='*70}")
        
        # Retrieve from both sources
        kg_results, text_results = self._retrieve_both_sources(question, retrieval_mode)
        
        if VERBOSE:
            print(f"KG: {len(kg_results.get('entities', []))} entities, "
                  f"{len(kg_results.get('triples', []))} triples")
            print(f"Text: {len(text_results)} chunks")
        
        # Fuse results
        fused_results = self.fuse_results(kg_results, text_results)

        # One-step adaptive expansion when top evidence is weak.
        fused_evidence = fused_results.get('fused_evidence', [])
        best_score = fused_evidence[0].get('combined_score', 0.0) if fused_evidence else 0.0
        if best_score < 0.25:
            if VERBOSE:
                print("Weak evidence detected; expanding retrieval once (KG +4, Text +2).")
            kg_results, text_results = self._retrieve_both_sources(
                question,
                retrieval_mode,
                kg_top_k=KG_TOP_K + 4,
                rag_top_k=RAG_TOP_K + 2,
            )
            fused_results = self.fuse_results(kg_results, text_results)
        
        # Generate answer
        answer = self.generate_answer(question, fused_results)
        
        if VERBOSE:
            display_answer = answer
            if len(answer) > 400:
                display_answer = answer[:400] + "..."
            print(f"\nGenerated answer:\n{display_answer}\n")
        
        return {
            'question': question,
            'answer': answer,
            'kg_results': kg_results,
            'text_results': text_results,
            'fused_results': fused_results
        }


# Example usage
if __name__ == "__main__":
    # Initialize pipeline with multi-GPU support (2 GPUs if available)
    pipeline = HybridPipeline(
        use_kg=True, 
        use_rag=True, 
        use_reranker=True,
        model_name='qwen',
        use_gpu=True  # Enable GPU acceleration (2+ GPUs)
    )
    
    test_questions = [
        "What are the main sources of greenhouse gas emissions?",
        "How does renewable energy impact climate change?",
        "What policies help reduce carbon emissions?"
    ]
    
    for q in test_questions[:1]:  # Test one question
        result = pipeline.query(q)
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}\n")
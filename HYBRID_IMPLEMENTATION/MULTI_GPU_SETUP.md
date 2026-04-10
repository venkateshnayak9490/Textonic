# Multi-GPU Setup Guide for Hybrid KG + RAG Pipeline

## Overview
The Hybrid_Implementation is now fully optimized for **2-GPU execution**. All components leverage GPU acceleration for efficient processing:
- **Text Embeddings**: GPU-accelerated FAISS indexing
- **LLM Inference**: Multi-GPU distributed with DataParallel
- **Cross-Encoder Reranking**: GPU-accelerated with batch processing

---

## GPU Acceleration Components

### 1. **LLM (llm.py)**
- ✅ Multi-GPU DataParallel distribution
- ✅ Auto device mapping across GPUs
- ✅ Automatic fallback to single GPU if needed

**Features:**
```python
load_model(model_id, token=HF_TOKEN, use_multi_gpu=True)
```

### 2. **Cross-Encoder Reranker (rerank.py)**
- ✅ GPU inference with batch processing (32 items/batch)
- ✅ Lazy model loading (loaded only on first use)
- ✅ Automatic multi-GPU distribution

**Features:**
```python
rerank(query, chunks, top_k=5, use_gpu=True, batch_size=32)
```

### 3. **FAISS Index (retrieval.py)**
- ✅ GPU-accelerated vector similarity search
- ✅ Dense embedding computation on GPU
- ✅ Fallback to CPU if GPU FAISS unavailable

**Features:**
```python
build_faiss_index(chunks, use_gpu=True, gpu_memory_fraction=0.5)
retrieve_top_k(query, index, chunks, model, use_gpu=True)
```

### 4. **Text Retriever (text_retriever.py)**
- ✅ GPU-aware TextRetriever class
- ✅ Automatic GPU detection and usage

**Features:**
```python
TextRetriever(pdf_folder=None, use_reranker=True, use_gpu=True)
```

### 5. **Hybrid Pipeline Orchestration (pipeline.py)**
- ✅ Multi-GPU status printing on initialization
- ✅ GPU info for all components
- ✅ Unified GPU management

---

## Usage Examples

### Basic Usage (Auto-GPU)
```python
from src.hybrid.pipeline import HybridPipeline

# Automatically detects and uses 2 GPUs if available
pipeline = HybridPipeline(
    use_kg=True,
    use_rag=True,
    use_reranker=True,
    model_name='llama',
    use_gpu=True  # Auto-detect CUDA
)

result = pipeline.query("How can carbon pricing reduce greenhouse gas emissions?")
print(result['answer'])
```

### Command Line Test
```bash
cd /home2/venkatesh.nayak/work/Hybrid_Implementation

python -c "
from src.hybrid.pipeline import HybridPipeline
import torch

# Check GPU availability
print(f'GPUs Available: {torch.cuda.device_count()}')

# Initialize pipeline with multi-GPU
p = HybridPipeline(use_kg=True, use_rag=True, use_reranker=True, model_name='llama', use_gpu=True)

# Run query
r = p.query('How can carbon pricing reduce greenhouse gas emissions?')

# Print results
print('\n' + '='*70)
print('QUERY RESULTS')
print('='*70)
print(f'Question: {r[\"question\"]}')
print(f'Answer: {r[\"answer\"]}')
print(f'KG Entities: {len(r[\"kg_results\"].get(\"entities\", []))}')
print(f'Text Chunks: {len(r[\"text_results\"])}')
print(f'Fused Evidence: {len(r[\"fused_results\"][\"fused_evidence\"])}')
"
```

### Disable GPU (CPU-only Mode)
```python
pipeline = HybridPipeline(
    use_kg=True,
    use_rag=True,
    use_reranker=True,
    model_name='llama',
    use_gpu=False  # Force CPU mode
)
```

---

## Performance Characteristics

### Expected Speedup with 2 GPUs
- **LLM Generation**: ~1.8-2.0x faster with DataParallel
- **Reranking**: ~2.0x faster (batch processing + GPU)
- **Embeddings**: ~3-4x faster (dense operations)
- **Overall Pipeline**: ~1.5-2.0x faster (I/O bound in places)

### GPU Memory Usage
- **GPU 0**: LLM (distributed across GPUs)
- **GPU 1**: LLM + Reranker + Embeddings

If OOM errors occur:
1. Reduce batch_size in rerank: `rerank(..., batch_size=16)`
2. Reduce GPU memory fraction: `build_faiss_index(..., gpu_memory_fraction=0.3)`
3. Use single GPU: `use_gpu=False`

---

## Configuration Variables

Set these environment variables to customize GPU behavior:

```bash
# Set HuggingFace token
export HF_TOKEN="your_hf_token"

# Control multi-GPU (optional)
export CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1

# Monitor GPU usage
nvidia-smi -l 1  # Monitor every 1 second
```

---

## Monitoring GPU Usage

### Real-time Monitoring
```bash
# Watch GPU utilization
nvidia-smi dmon

# Monitor with refresh
watch -n 1 nvidia-smi

# Detailed memory info
nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv,noheader
```

### In Python
```python
import torch
import subprocess

# Check GPU count
print(f"GPUs: {torch.cuda.device_count()}")

# Current GPU memory
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Memory: {torch.cuda.memory_allocated(i) / 1e9:.2f}GB / {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f}GB")
```

---

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution:**
```python
# Reduce batch size
rerank(query, chunks, batch_size=8)  # Instead of 32

# Disable reranker
TextRetriever(use_reranker=False)

# Use CPU
pipeline = HybridPipeline(use_gpu=False)
```

### Issue: GPU Not Detected
**Check:**
```bash
nvidia-smi  # Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Slower with GPU than CPU
**Likely Cause:** Small batch sizes, data transfer overhead
**Solution:**
- Ensure batch_size is large enough (≥32)
- Reduce number of CPU-GPU transfers
- Check nvidia-smi for actual GPU usage

---

## Architecture Diagram

```
Query
  ↓
[Text Retrieval (GPU)]  ← Embeddings on GPU
  ├→ Dense search (GPU FAISS)
  ├→ Reranking (GPU CrossEncoder)
  ↓
[KG Retrieval (CPU)]  ← Knowledge graph search
  ↓
[Fusion]
  ↓
[LLM Generation (Multi-GPU)]  ← Distributed across 2 GPUs
  ↓
Answer
```

---

## Files Modified for Multi-GPU

✅ `src/rag/src/llm.py` - Multi-GPU LLM loading
✅ `src/rag/src/rerank.py` - GPU reranker with batch processing
✅ `src/rag/src/retrieval.py` - GPU-accelerated FAISS
✅ `src/hybrid/text_retriever.py` - GPU-aware text retrieval
✅ `src/hybrid/pipeline.py` - Multi-GPU orchestration

---

## Next Steps

1. **Run the test command** to verify multi-GPU setup
2. **Monitor GPU usage** with `nvidia-smi`
3. **Benchmark performance** for your use case
4. **Adjust batch sizes** based on your GPU memory

For issues or optimization tips, check the logs and monitor GPU utilization!

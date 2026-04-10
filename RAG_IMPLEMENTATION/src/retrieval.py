import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_faiss_index(chunks):
    embedding_model = SentenceTransformer("BAAI/bge-base-en")

    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, embedding_model

def retrieve_top_k(query, index, chunks, embedding_model, k=20):
    query_emb = embedding_model.encode([query])
    query_emb = np.array(query_emb).astype("float32")

    distances, indices = index.search(query_emb, k)

    return [chunks[i] for i in indices[0]]

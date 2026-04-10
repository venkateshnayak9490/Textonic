from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, retrieved_chunks, top_k=3):
    pairs = [(query, c["text"]) for c in retrieved_chunks]
    scores = reranker_model.predict(pairs)

    ranked = sorted(zip(retrieved_chunks, scores),
                    key=lambda x: x[1],
                    reverse=True)

    return [x[0] for x in ranked[:top_k]]

import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict
from typing import Optional


class Retriever:
    def __init__(self, index_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.index = faiss.read_index(index_path)
        meta_path = index_path + ".meta.json"
        with open(meta_path, "r", encoding="utf-8") as fin:
            self.metadata = json.load(fin)
        self.model = SentenceTransformer(model_name)

    def query(self, text: str, top_k: int = 5) -> Dict:
        emb = self.model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        D, I = self.index.search(emb, top_k)
        scores = D[0].tolist()
        inds = I[0].tolist()
        results = []
        for s, idx in zip(scores, inds):
            if idx < 0 or idx >= len(self.metadata):
                continue
            m = self.metadata[idx]
            results.append({
                "chunk_id": m["chunk_id"],
                "source": m["source"],
                "page": m["page"],
                "text": m["text"],
                "score": float(s),
            })
        return {"query": text, "results": results}


def rerank_results(query: str, results: List[Dict], model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> List[Dict]:
    """Rerank a list of result dicts using a CrossEncoder. Returns results sorted by rerank score.

    This is lazy: the CrossEncoder model will be downloaded on first call.
    WARNING: This requires 8GB+ RAM. May cause kernel crashes on limited-memory systems.
    """
    if not results:
        return results
    
    # Check available memory
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        if available_gb < 2.0:
            raise RuntimeError(f"Insufficient RAM: {available_gb:.1f}GB available, but reranking requires ~4GB. Skipping reranking.")
    except ImportError:
        # psutil not available, proceed with warning
        print("⚠ Warning: psutil not available. Cannot verify memory. Reranking may cause crashes on low-memory systems.")
    
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        raise RuntimeError("CrossEncoder is required for reranking. Install sentence-transformers and retry.")

    texts = [r["text"] for r in results]
    pairs = [[query, t] for t in texts]
    reranker = CrossEncoder(model_name)
    scores = reranker.predict(pairs)
    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)
    results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return results


def synthesize_answer(retrieval: Dict, score_threshold: float = 0.25) -> Dict:
    results = retrieval["results"]
    if not results:
        return {"answer": "I couldn't find relevant information.", "citations": [], "low_confidence": True}
    top_score = results[0]["score"]
    citations = []
    snippets = []
    for r in results:
        citations.append(f"[{r['source']}:{r['page']}]")
        snippets.append(r["text"])
    if top_score < score_threshold:
        return {
            "answer": "I don't have strong evidence to answer confidently. Could you clarify?",
            "citations": citations,
            "low_confidence": True,
            "retrieval": retrieval,
        }
    # Better synthesis: preserve line breaks and emit bullet/point-form when appropriate
    top_snip = snippets[0].strip()
    # Normalize common whitespace
    top_snip = '\n'.join(line.rstrip() for line in top_snip.splitlines() if line.strip())

    # If the snippet contains multiple short lines, format as bullets for readability
    lines = [ln for ln in top_snip.splitlines() if ln.strip()]
    if len(lines) > 1:
        # Heuristic: if lines are reasonably short, present as bullet points
        avg_len = sum(len(ln) for ln in lines) / len(lines)
        if avg_len < 140 or len(lines) <= 10:
            formatted = '\n'.join(f"- {ln}" for ln in lines[:50])
        else:
            # long paragraphs: truncate to keep answers concise
            formatted = '\n'.join(lines[:5])
    else:
        # Single-line or long paragraph: truncate safely
        formatted = top_snip if len(top_snip) <= 1000 else top_snip[:1000].rsplit(' ', 1)[0] + '...'

    answer = f"{formatted}\n\nCitations: {'; '.join(citations)}"
    return {"answer": answer, "citations": citations, "low_confidence": False, "retrieval": retrieval}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("index_path")
    p.add_argument("query")
    args = p.parse_args()
    r = Retriever(args.index_path)
    ret = r.query(args.query, top_k=5)
    out = synthesize_answer(ret)
    print(json.dumps(out, ensure_ascii=False, indent=2))

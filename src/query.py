import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Optional


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
    if not results:
        return results
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        if available_gb < 2.0:
            raise RuntimeError(f"Insufficient RAM: {available_gb:.1f}GB available. Skipping reranking.")
    except ImportError:
        pass
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        raise RuntimeError("CrossEncoder is required for reranking.")
    texts = [r["text"] for r in results]
    pairs = [[query, t] for t in texts]
    reranker = CrossEncoder(model_name)
    scores = reranker.predict(pairs)
    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)
    results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return results


def _call_claude(prompt: str, max_tokens: int = 400) -> str:
    """Call Claude API to synthesize a focused answer. Returns plain text."""
    import urllib.request
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"].strip()
    except Exception as e:
        return ""


def _extract_focused_answer(query: str, chunks: List[Dict]) -> str:
    """
    Use Claude API to extract a focused, direct answer from retrieved chunks.
    Falls back to best-effort text extraction if API unavailable.
    """
    context_parts = []
    for i, r in enumerate(chunks[:3], 1):
        context_parts.append(f"[Chunk {i} — {r['source']} page {r['page']}]\n{r['text']}")
    context = "\n\n".join(context_parts)

    prompt = f"""You are a document assistant. Answer the question below using ONLY the provided document chunks.
Be direct and concise. Give a 1-3 sentence answer. Do NOT mention the chunks or say "according to". 
Just answer the question plainly. If the answer is not in the chunks, say "Not found in the documents."

Question: {query}

Document chunks:
{context}

Answer:"""

    answer = _call_claude(prompt, max_tokens=300)
    if answer and answer != "Not found in the documents." and len(answer) > 10:
        return answer

    # Fallback: find the single best sentence from top chunk by keyword overlap
    import re
    top_text = chunks[0]["text"]
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n', top_text) if len(s.strip()) > 25]
    if not sentences:
        return top_text[:300].strip()

    query_words = set(re.sub(r'[^\w\s]', '', query.lower()).split())
    stop = {'what','is','the','a','an','of','in','for','to','and','or','how','does','do',
            'did','was','were','are','who','when','where','which','with','that','has',
            'have','had','can','could','would','should','me','my','his','her','their'}
    keywords = query_words - stop

    def score(s):
        words = set(re.sub(r'[^\w\s]', '', s.lower()).split())
        return len(words & keywords)

    best = max(sentences, key=score)
    # Take best + the sentence after it for context
    idx = sentences.index(best)
    selected = sentences[idx: idx + 2]
    return ' '.join(selected)


def synthesize_answer(retrieval: Dict, score_threshold: float = 0.25) -> Dict:
    results = retrieval.get("results", [])

    if not results:
        return {"answer": "I couldn't find relevant information.", "citations": [], "low_confidence": True}

    # Filter results that pass threshold
    passing = [r for r in results if r["score"] >= score_threshold]

    # Build deduplicated citations
    seen_keys = set()
    citations = []
    for r in (passing if passing else results):
        key = f"{r['source']}:{r['page']}"
        if key not in seen_keys:
            citations.append(f"[{r['source']}:{r['page']}]")
            seen_keys.add(key)

    if not passing:
        return {
            "answer": "I don't have strong evidence to answer confidently. Could you clarify your question?",
            "citations": citations,
            "low_confidence": True,
            "retrieval": retrieval,
        }

    query_text = retrieval.get("query", "")
    focused = _extract_focused_answer(query_text, passing)

    return {
        "answer": focused,
        "citations": citations,
        "low_confidence": False,
        "retrieval": retrieval
    }


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

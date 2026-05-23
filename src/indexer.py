import json
import os
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


def load_chunks(chunks_path: str) -> List[dict]:
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as fin:
        for line in fin:
            chunks.append(json.loads(line))
    return chunks


def build_faiss_index(chunks_path: str, index_path: str, model_name: str = "all-MiniLM-L6-v2"):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    model = SentenceTransformer(model_name)
    chunks = load_chunks(chunks_path)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # normalize for cosine similarity using inner product
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    meta_path = index_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as fout:
        json.dump(chunks, fout, ensure_ascii=False, indent=2)
    print(f"Wrote index to {index_path} and metadata to {meta_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("chunks_path")
    p.add_argument("index_path")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    args = p.parse_args()
    build_faiss_index(args.chunks_path, args.index_path, args.model)

import pandas as pd
import os
import json


def load_eval_set(path: str):
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load evaluation CSV at {path}: {exc}") from exc


def citation_in_topk(citation: str, topk_results: list) -> bool:
    # citation format: filename:page
    return any(f"{r['source']}:{r['page']}" == citation for r in topk_results)


def precision_at_k(eval_csv: str, index_path: str, k: int = 3):
    from src.query import Retriever

    df = load_eval_set(eval_csv)
    r = Retriever(index_path)
    hits = 0
    for _, row in df.iterrows():
        q = row['question']
        gold = row['gold_citation']
        res = r.query(q, top_k=k)
        if citation_in_topk(gold, res['results']):
            hits += 1
    return hits / len(df)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('eval_csv')
    ap.add_argument('index_path')
    args = ap.parse_args()
    for k in (3, 5):
        p = precision_at_k(args.eval_csv, args.index_path, k=k)
        print(f'Precision@{k}: {p:.3f}')

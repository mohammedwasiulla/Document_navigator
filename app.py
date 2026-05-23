import streamlit as st
from pathlib import Path
import os
import pandas as pd
from src.query import Retriever, synthesize_answer
from src.ingest import ingest_pdfs
from src.indexer import build_faiss_index
from src.evaluate import load_eval_set


BASE = Path(__file__).parent
PDF_DIR = BASE / "pdfs"
DATA_DIR = BASE / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
INDEX_PATH = DATA_DIR / "faiss.index"
EVAL_CSV = BASE / "eval_set.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Document Navigator", layout="wide")
st.title("Document Navigator — Transparent RAG Assistant")

with st.sidebar:
    st.header("Indexing")
    st.markdown("**Upload a PDF**")
    uploaded_file = st.file_uploader("Upload PDF to index", type=["pdf"])
    if uploaded_file is not None:
        save_path = PDF_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved {uploaded_file.name} to {PDF_DIR}")
        if st.button("Ingest uploaded PDFs and build index"):
            st.info("Ingesting PDFs and building FAISS index — this may take a while")
            ingest_pdfs(str(PDF_DIR), str(CHUNKS_PATH))
            build_faiss_index(str(CHUNKS_PATH), str(INDEX_PATH))
            st.success("Index built from uploaded PDFs")
    if st.button("(Re)ingest PDFs and build index"):
        st.info("Ingesting PDFs and building FAISS index — this may take a while")
        ingest_pdfs(str(PDF_DIR), str(CHUNKS_PATH))
        build_faiss_index(str(CHUNKS_PATH), str(INDEX_PATH))
        st.success("Index built")
    st.markdown("---")
    st.markdown("Settings")
    top_k = st.number_input("Top-k", min_value=1, max_value=20, value=3)
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25)
    dedupe = st.checkbox("Deduplicate by source/page", value=True, key="dedupe_checkbox")
    use_reranker = st.checkbox("Use cross-encoder reranker (may download model)", value=False, key="reranker_checkbox")

if not INDEX_PATH.exists():
    st.warning("Index not found. Run (Re)ingest from the sidebar first.")

query = st.text_input("Ask a question about the PDFs:")
if query:
    retriever = Retriever(str(INDEX_PATH))
    # If reranker is enabled, retrieve a larger candidate set for reranking
    if use_reranker:
        candidate_k = max(10, top_k * 4)
        ret = retriever.query(query, top_k=candidate_k)
    else:
        ret = retriever.query(query, top_k=top_k)
    # Filter by score threshold to reduce noisy low-similarity hits
    ret["results"] = [r for r in ret["results"] if r["score"] >= threshold]

    # Optional deduplication: keep first chunk per (source,page)
    if dedupe:
        seen = set()
        filtered = []
        for r in ret["results"]:
            key = (r["source"], r["page"])
            if key in seen:
                continue
            seen.add(key)
            filtered.append(r)
        ret["results"] = filtered
    # If reranker is requested and candidates exist, rerank now
    if use_reranker and ret.get("results"):
        try:
            from src.query import rerank_results
            ret["results"] = rerank_results(query, ret["results"])[:top_k]
        except Exception as e:
            st.warning(f"Reranker failed or unavailable: {e}")

    synth = synthesize_answer(ret, score_threshold=threshold)
    st.subheader("Answer")
    if synth.get("low_confidence"):
        st.error(synth["answer"])
    else:
        st.success(synth["answer"])

    # Load and display gold citation/key phrase from eval set if available
    if EVAL_CSV.exists():
        try:
            eval_df = load_eval_set(str(EVAL_CSV))
            # Try to find matching question
            matching = eval_df[eval_df["question"].str.lower().str.contains(query.lower(), na=False)]
            if not matching.empty:
                st.subheader("📋 Evaluation Reference")
                for _, row in matching.iterrows():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Gold Citation:** `{row['gold_citation']}`")
                    with col2:
                        st.markdown(f"**Key Phrase:** {row['gold_key_phrase']}")
        except Exception as e:
            pass  # Silently skip if eval_set not available

    st.subheader("Retrieval Traces")
    for r in ret["results"]:
        st.markdown(f"**Chunk**: {r['chunk_id']} — **Source**: {r['source']}:{r['page']} — **Score**: {r['score']:.4f}")
        st.write(r["text"]) 

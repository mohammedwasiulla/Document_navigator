import streamlit as st
from pathlib import Path
import os
from src.query import Retriever, synthesize_answer
from src.ingest import ingest_pdfs
from src.indexer import build_faiss_index
from src.evaluate import load_eval_set

BASE       = Path(__file__).parent
PDF_DIR    = BASE / "pdfs"
DATA_DIR   = BASE / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
INDEX_PATH  = DATA_DIR / "faiss.index"
EVAL_CSV   = BASE / "eval_set.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Document Navigator", layout="wide")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📄 Document Navigator — Transparent RAG Assistant")
st.caption("Ask questions across your PDF library — every answer is fully traceable.")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 PDF Management")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        saved = []
        for uf in uploaded_files:
            save_path = PDF_DIR / uf.name
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
            saved.append(uf.name)
        st.toast(f"✅ Saved {len(saved)} file(s): {', '.join(saved)}", icon="📄")

    # Show currently indexed PDFs
    if CHUNKS_PATH.exists():
        import json
        try:
            chunks = [json.loads(l) for l in open(CHUNKS_PATH)]
            indexed = sorted(set(c["source"] for c in chunks))
            with st.expander(f"📚 Indexed PDFs ({len(indexed)})"):
                for name in indexed:
                    st.markdown(f"• `{name}`")
        except Exception:
            pass

    if st.button("🔄 (Re)ingest & Rebuild Index", type="primary"):
        with st.spinner("Ingesting PDFs with pdfplumber and building FAISS index…"):
            try:
                ingest_pdfs(str(PDF_DIR), str(CHUNKS_PATH))
                build_faiss_index(str(CHUNKS_PATH), str(INDEX_PATH))
                st.toast("✅ Index built successfully!", icon="🔍")
                st.success("Index ready — ask your first question!")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()
    st.header("⚙️ Search Settings")
    top_k     = st.number_input("Top-k chunks to retrieve", min_value=1, max_value=20, value=5)
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.15, step=0.05,
                          help="Lower = more answers, Higher = more selective")
    dedupe       = st.checkbox("Deduplicate by source/page", value=True)
    use_reranker = st.checkbox("Use cross-encoder reranker (slower, more accurate)", value=False)

    st.divider()
    st.caption("💡 Tip: After uploading a new PDF, always click **(Re)ingest** before querying.")

# ── Index check ───────────────────────────────────────────────────────────────
if not INDEX_PATH.exists():
    st.warning("⚠️ No index found. Upload your PDFs and click **🔄 (Re)ingest & Rebuild Index** in the sidebar.")
    st.stop()

# ── Query input ───────────────────────────────────────────────────────────────
query = st.text_input(
    "💬 Ask a question about your documents",
    placeholder="e.g.  What is Mohammed's email?  /  What is the net pay?  /  What is the return policy?"
)

if query:
    with st.spinner("🔍 Searching across your documents…"):
        retriever    = Retriever(str(INDEX_PATH))
        candidate_k  = max(10, top_k * 4) if use_reranker else top_k
        ret          = retriever.query(query, top_k=candidate_k)

        # Deduplicate by (source, page) before threshold
        if dedupe:
            seen, deduped = set(), []
            for r in ret["results"]:
                key = (r["source"], r["page"])
                if key not in seen:
                    seen.add(key)
                    deduped.append(r)
            ret["results"] = deduped

        # Optional reranker
        if use_reranker and ret.get("results"):
            try:
                from src.query import rerank_results
                ret["results"] = rerank_results(query, ret["results"])[:top_k]
            except Exception as e:
                st.toast(f"⚠️ Reranker unavailable: {e}", icon="⚠️")

        synth = synthesize_answer(ret, score_threshold=threshold)

    # Results above threshold for display
    display_results = [r for r in ret["results"] if r["score"] >= threshold]

    # ── Answer box ────────────────────────────────────────────────────────────
    st.subheader("💬 Answer")

    if synth.get("low_confidence"):
        st.error(f"⚠️ {synth['answer']}")
        st.toast("Low confidence — try rephrasing or lowering the threshold.", icon="⚠️")
    else:
        st.success(synth["answer"])
        st.toast("✅ Answer found!", icon="✅")

        # Citations as pills below the answer
        if synth.get("citations"):
            cit_str = "  ·  ".join(synth["citations"])
            st.caption(f"📎 **Sources:** {cit_str}")

    st.divider()

    # ── Gold citation reference (from eval_set.csv) ───────────────────────────
    if EVAL_CSV.exists():
        try:
            eval_df  = load_eval_set(str(EVAL_CSV))
            matching = eval_df[
                eval_df["question"].str.lower().str.contains(query.lower(), na=False)
            ]
            if not matching.empty:
                st.subheader("📋 Evaluation Reference")
                for _, row in matching.iterrows():
                    col1, col2, col3 = st.columns([2, 2, 3])
                    with col1:
                        st.markdown("**Question (eval set)**")
                        st.info(row["question"])
                    with col2:
                        st.markdown("**Gold Citation**")
                        st.code(row["gold_citation"], language=None)
                    with col3:
                        st.markdown("**Gold Key Phrase**")
                        st.warning(row["gold_key_phrase"])
                st.divider()
        except Exception:
            pass

    # ── Retrieval Traces ──────────────────────────────────────────────────────
    st.subheader("🔍 Retrieval Traces")

    if not display_results:
        st.info("No chunks scored above the confidence threshold. Try lowering it in the sidebar (currently set to {:.2f}).".format(threshold))
    else:
        st.caption(f"Showing {len(display_results)} chunk(s) above threshold {threshold:.2f}")
        for i, r in enumerate(display_results, 1):
            score_color = "🟢" if r["score"] >= 0.35 else ("🟡" if r["score"] >= 0.20 else "🔴")
            header = f"{score_color} **Chunk {i}** — `{r['source']}` page {r['page']} — score: `{r['score']:.4f}`"
            with st.expander(header, expanded=(i == 1)):
                st.markdown(f"**Chunk ID:** `{r['chunk_id']}`")
                st.text_area("Chunk text", r["text"], height=160, disabled=True, key=f"chunk_{i}")

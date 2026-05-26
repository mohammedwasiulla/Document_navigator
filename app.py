import streamlit as st
from pathlib import Path
import pandas as pd

try:
    from src.query import Retriever, synthesize_answer
    from src.ingest import ingest_pdfs
    from src.indexer import build_faiss_index
    from src.evaluate import load_eval_set
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

BASE        = Path(__file__).parent
PDF_DIR     = BASE / "pdfs"
DATA_DIR    = BASE / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
INDEX_PATH  = DATA_DIR / "faiss.index"
EVAL_CSV    = BASE / "eval_set.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="DocNav · RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Toast Message ─────────────────────────────
if "toast_shown" not in st.session_state:
    st.session_state.toast_shown = False

if not st.session_state.toast_shown:
    st.toast(
        "Use the sidebar to upload your PDF documents or adjust the precision and confidence settings for improved retrieval accuracy.",
        icon="ℹ️"
    )

    st.session_state.toast_shown = True

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── page background ── */
[data-testid="stAppViewContainer"] {
  background-color: #0f1117 !important;
  background-image:
    radial-gradient(circle at 15% 10%, rgba(99,179,237,0.13) 0%, transparent 40%),
    radial-gradient(circle at 85% 90%, rgba(252,129,74,0.11) 0%, transparent 38%),
    radial-gradient(circle, #1a2032 1px, transparent 1px) !important;
  background-size: auto, auto, 28px 28px !important;
}
[data-testid="stAppViewContainer"] > .main {
  background: transparent !important;
}
.block-container {
  position: relative;
  z-index: 1;
  max-width: 860px !important;
  padding-top: 2rem !important;
}

/* ── fix text input label visibility ── */
[data-testid="stTextInput"] label,
[data-testid="stTextInput"] label p {
  color: #e6edf3 !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
}
[data-testid="stTextInput"] input {
  background: rgba(30, 37, 50, 0.95) !important;
  border: 1.5px solid rgba(99,179,237,0.35) !important;
  border-radius: 10px !important;
  color: #e6edf3 !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 1rem !important;
  padding: 0.75rem 1rem !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stTextInput"] input::placeholder {
  color: #4a5568 !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: #63b3ed !important;
  box-shadow: 0 0 0 3px rgba(99,179,237,0.18) !important;
  outline: none !important;
}

/* ── hide chrome ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    st.divider()

    st.subheader("📂 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file:
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        with open(PDF_DIR / uploaded_file.name, "wb") as fh:
            fh.write(uploaded_file.getbuffer())
        st.success(f"Saved: {uploaded_file.name}")
        if st.button("⬆️ Ingest & Build Index", use_container_width=True):
            with st.spinner("Building index…"):
                if MODULES_AVAILABLE:
                    ingest_pdfs(str(PDF_DIR), str(CHUNKS_PATH))
                    build_faiss_index(str(CHUNKS_PATH), str(INDEX_PATH))
            st.success("Index built!")
            st.rerun()

    if st.button("🔄 Re-ingest All PDFs", use_container_width=True):
        with st.spinner("Ingesting all PDFs…"):
            if MODULES_AVAILABLE:
                ingest_pdfs(str(PDF_DIR), str(CHUNKS_PATH))
                build_faiss_index(str(CHUNKS_PATH), str(INDEX_PATH))
        st.success("Index rebuilt!")
        st.rerun()

    st.divider()
    st.subheader("🔧 Settings")
    top_k = st.number_input("Top-K chunks", min_value=1, max_value=20, value=3, step=1)
    threshold = st.number_input(
        "Confidence threshold (0–1)",
        min_value=0.00, max_value=1.00,
        value=0.25, step=0.05, format="%.2f"
    )
    dedupe       = st.checkbox("Deduplicate by source/page", value=True)
    use_reranker = st.checkbox("Use cross-encoder reranker", value=False)

    st.divider()
    idx_ok    = INDEX_PATH.exists()
    pdf_count = len(list(PDF_DIR.glob("*.pdf"))) if PDF_DIR.exists() else 0
    st.metric("PDFs indexed", pdf_count)
    if idx_ok:
        st.success("✅ Index is ready")
    else:
        st.warning("⚠️ No index found")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
idx_ok = INDEX_PATH.exists()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.8rem;padding-bottom:1.2rem;
            border-bottom:1px solid rgba(255,255,255,0.07);">
  <p style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
            letter-spacing:0.2em;text-transform:uppercase;color:#63b3ed;
            margin:0 0 0.4rem;">Transparent Retrieval-Augmented Generation</p>
  <h1 style="font-family:'Inter',sans-serif;font-weight:700;
             font-size:clamp(1.8rem,4vw,2.5rem);
             background:linear-gradient(135deg,#e6edf3 30%,#63b3ed 100%);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;margin:0;line-height:1.15;">
    Document Navigator
  </h1>
  <p style="font-family:'Inter',sans-serif;color:#8b949e;
            font-size:0.9rem;margin:0.4rem 0 0;">
    Ask questions across your PDF library — every answer is traceable.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Search bar (always visible, fully readable label) ────────────────────────
query = st.text_input(
    "💬 Ask a question about your documents",
    placeholder="e.g. What are the key findings in chapter 3?",
)

# ─────────────────────────────────────────────────────────────────────────────
# WELCOME PAGE  — shown when no query typed yet
# ─────────────────────────────────────────────────────────────────────────────
if not query:
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # hero card
    st.markdown("""
    <div style="
      background: linear-gradient(135deg, rgba(99,179,237,0.08) 0%, rgba(252,129,74,0.06) 100%);
      border: 1px solid rgba(99,179,237,0.2);
      border-radius: 16px;
      padding: 2.5rem 2rem;
      text-align: center;
      margin-bottom: 2rem;
    ">
      <div style="font-size:3rem;margin-bottom:1rem;">📚</div>
      <h2 style="font-family:'Inter',sans-serif;font-weight:700;font-size:1.5rem;
                 color:#e6edf3;margin:0 0 0.6rem;">Welcome to Document Navigator</h2>
      <p style="font-family:'Inter',sans-serif;color:#8b949e;font-size:0.95rem;
                max-width:480px;margin:0 auto 1.5rem;line-height:1.6;">
        Upload your PDFs, build the index, then ask any question in plain English.
        Every answer is backed by traceable source chunks with similarity scores.
      </p>
      <div style="display:flex;justify-content:center;gap:0.8rem;flex-wrap:wrap;">
        <span style="background:rgba(99,179,237,0.12);border:1px solid rgba(99,179,237,0.25);
                     border-radius:20px;padding:0.3rem 0.9rem;
                     font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#63b3ed;">
          FAISS Vector Search
        </span>
        <span style="background:rgba(252,129,74,0.10);border:1px solid rgba(252,129,74,0.22);
                     border-radius:20px;padding:0.3rem 0.9rem;
                     font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#fc814a;">
          Cross-Encoder Reranking
        </span>
        <span style="background:rgba(63,185,80,0.10);border:1px solid rgba(63,185,80,0.22);
                     border-radius:20px;padding:0.3rem 0.9rem;
                     font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#3fb950;">
          Transparent RAG
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 3-step guide
    c1, c2, c3 = st.columns(3)
    steps = [
        ("1️⃣", "Upload PDF", "Use the sidebar to upload one or more PDF files to your library."),
        ("2️⃣", "Build Index", "Click Re-ingest All PDFs to embed and index all documents."),
        ("3️⃣", "Ask Away",    "Type any question above and get an answer with source traces."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], steps):
        with col:
            st.markdown(f"""
            <div style="
              background:rgba(22,27,34,0.8);
              border:1px solid rgba(255,255,255,0.08);
              border-radius:12px;
              padding:1.4rem 1.2rem;
              text-align:center;
              height:100%;
            ">
              <div style="font-size:1.8rem;margin-bottom:0.6rem;">{icon}</div>
              <div style="font-family:'Inter',sans-serif;font-weight:600;
                          font-size:0.95rem;color:#e6edf3;margin-bottom:0.5rem;">{title}</div>
              <div style="font-family:'Inter',sans-serif;font-size:0.83rem;
                          color:#8b949e;line-height:1.55;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # feature highlights
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(22,27,34,0.6);border:1px solid rgba(255,255,255,0.07);
                border-radius:12px;padding:1.4rem 1.6rem;">
      <p style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                letter-spacing:0.15em;text-transform:uppercase;color:#63b3ed;
                margin:0 0 1rem;">✦ What you can do</p>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:0.8rem;">
        <div style="display:flex;align-items:flex-start;gap:0.6rem;">
          <span style="color:#3fb950;font-size:1rem;flex-shrink:0;">✓</span>
          <span style="font-family:'Inter',sans-serif;font-size:0.85rem;
                       color:#8b949e;line-height:1.5;">Ask natural-language questions across all PDFs</span>
        </div>
        <div style="display:flex;align-items:flex-start;gap:0.6rem;">
          <span style="color:#3fb950;font-size:1rem;flex-shrink:0;">✓</span>
          <span style="font-family:'Inter',sans-serif;font-size:0.85rem;
                       color:#8b949e;line-height:1.5;">See exactly which page &amp; chunk each answer comes from</span>
        </div>
        <div style="display:flex;align-items:flex-start;gap:0.6rem;">
          <span style="color:#3fb950;font-size:1rem;flex-shrink:0;">✓</span>
          <span style="font-family:'Inter',sans-serif;font-size:0.85rem;
                       color:#8b949e;line-height:1.5;">Filter by confidence threshold to reduce noise</span>
        </div>
        <div style="display:flex;align-items:flex-start;gap:0.6rem;">
          <span style="color:#3fb950;font-size:1rem;flex-shrink:0;">✓</span>
          <span style="font-family:'Inter',sans-serif;font-size:0.85rem;
                       color:#8b949e;line-height:1.5;">Optional cross-encoder reranking for higher accuracy</span>
        </div>
        <div style="display:flex;align-items:flex-start;gap:0.6rem;">
          <span style="color:#3fb950;font-size:1rem;flex-shrink:0;">✓</span>
          <span style="font-family:'Inter',sans-serif;font-size:0.85rem;
                       color:#8b949e;line-height:1.5;">Deduplicate results across pages for cleaner output</span>
        </div>
        <div style="display:flex;align-items:flex-start;gap:0.6rem;">
          <span style="color:#3fb950;font-size:1rem;flex-shrink:0;">✓</span>
          <span style="font-family:'Inter',sans-serif;font-size:0.85rem;
                       color:#8b949e;line-height:1.5;">Evaluation mode: compare against gold citations</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not idx_ok:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.warning("⚠️ No index found yet — upload a PDF and click **Re-ingest All PDFs** in the sidebar to get started.")

# ─────────────────────────────────────────────────────────────────────────────
# QUERY RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if query and MODULES_AVAILABLE and idx_ok:
    retriever   = Retriever(str(INDEX_PATH))
    candidate_k = max(10, top_k * 4) if use_reranker else top_k

    with st.spinner("Searching documents…"):
        ret = retriever.query(query, top_k=candidate_k)

    ret["results"] = [r for r in ret["results"] if r["score"] >= threshold]

    if dedupe:
        seen, filtered = set(), []
        for r in ret["results"]:
            k = (r["source"], r["page"])
            if k not in seen:
                seen.add(k); filtered.append(r)
        ret["results"] = filtered

    if use_reranker and ret.get("results"):
        try:
            from src.query import rerank_results
            ret["results"] = rerank_results(query, ret["results"])[:top_k]
        except Exception as e:
            st.warning(f"Reranker unavailable: {e}")

    synth  = synthesize_answer(ret, score_threshold=threshold)
    is_low = synth.get("low_confidence")

    st.markdown("### 💬 Answer")
    if is_low:
        st.error(synth["answer"])
    else:
        st.success(synth["answer"])
    st.markdown("""
<style>

/* ─────────────────────────────────────────────
   BRIGHT TEXT + PREMIUM UI
───────────────────────────────────────────── */

/* Main markdown text */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: #f5f7fa !important;
}

/* Answer box */
.answer-box {
    background: linear-gradient(
        135deg,
        rgba(25,35,50,0.98),
        rgba(15,20,28,0.98)
    );

    border: 1px solid rgba(99,179,237,0.35);
    border-left: 5px solid #63b3ed;

    border-radius: 16px;
    padding: 1.4rem;
    margin-bottom: 1.5rem;

    box-shadow:
        0 8px 25px rgba(0,0,0,0.35);
}

.answer-text {
    color: #ffffff !important;
    font-size: 1rem;
    line-height: 1.8;
    font-weight: 400;
}

/* Retrieval title */
.retrieval-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #ffffff !important;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

/* Expander container */
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    overflow: hidden;
    background: rgba(15,18,25,0.95) !important;
    margin-bottom: 1rem !important;
}

/* Expander tab/header */
[data-testid="stExpander"] summary {
    background: linear-gradient(
        135deg,
        rgba(35,45,65,0.98),
        rgba(25,30,40,0.98)
    ) !important;

    padding: 0.9rem 1rem !important;

    border-radius: 14px !important;

    color: #ffffff !important;

    font-weight: 600 !important;

    font-size: 0.92rem !important;
}

/* Hover */
[data-testid="stExpander"] summary:hover {
    background: linear-gradient(
        135deg,
        rgba(45,60,85,1),
        rgba(30,35,48,1)
    ) !important;
}

/* Chunk text box */
.chunk-box {
    background: rgba(10,14,20,0.96);

    border: 1px solid rgba(99,179,237,0.15);

    border-radius: 12px;

    padding: 1rem;

    margin-top: 1rem;

    color: #f8fafc !important;

    line-height: 1.9;

    font-size: 0.97rem;

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.03);
}

/* Chunk meta */
.chunk-meta {
    color: #c9d1d9 !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
}

/* Bright labels */
.bright-label {
    color: #63b3ed !important;
    font-weight: 700;
    margin-bottom: 0.8rem;
    letter-spacing: 0.08em;
    font-size: 0.78rem;
}

/* Score labels */
.high-score {
    color: #3fb950 !important;
    font-weight: 700;
}

.medium-score {
    color: #e3b341 !important;
    font-weight: 700;
}

.low-score {
    color: #f85149 !important;
    font-weight: 700;
}

/* Evaluation cards */
.eval-card {
    background: rgba(18,22,30,0.96);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 14px;

    padding: 1rem;

    margin-bottom: 1rem;
}

/* Evaluation text */
.eval-text {
    color: #ffffff !important;
    font-size: 0.95rem;
    line-height: 1.7;
}

</style>
""", unsafe_allow_html=True)


    # eval reference
    if EVAL_CSV.exists():
        try:
            eval_df  = load_eval_set(str(EVAL_CSV))
            matching = eval_df[eval_df["question"].str.lower().str.contains(query.lower(), na=False)]
            if not matching.empty:
                st.markdown("### 📋 Evaluation Reference")
                for _, row in matching.iterrows():
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Gold Citation:** `{row['gold_citation']}`")
                    with c2:
                        st.markdown(f"**Key Phrase:** {row['gold_key_phrase']}")
        except Exception:
            pass

    # retrieval traces
    if ret["results"]:

        st.markdown(f"""
        <div class="retrieval-title">
        🔍 Retrieval Traces — {len(ret["results"])} chunk(s)
        </div>
        """, unsafe_allow_html=True)

        for i, r in enumerate(ret["results"]):

            sc = r["score"]
            sc_pct = int(sc * 100)

            if sc >= 0.6:
                label = '<span class="high-score">🟢 High Confidence</span>'
                bar_col = "#3fb950"

            elif sc >= threshold:
                label = '<span class="medium-score">🟡 Medium Confidence</span>'
                bar_col = "#e3b341"

            else:
                label = '<span class="low-score">🔴 Low Confidence</span>'
                bar_col = "#f85149"

            with st.expander(
                f"#{i+1} · {r['source']} p.{r['page']} · Score {sc:.4f}"
            ):

                st.markdown(label, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="chunk-meta">

                Similarity score: {sc:.4f}
                &nbsp;&nbsp;•&nbsp;&nbsp;
                chunk_id: {r['chunk_id']}

                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="
                    background:#1f2937;
                    border-radius:999px;
                    overflow:hidden;
                    height:8px;
                    margin-top:0.8rem;
                    margin-bottom:1rem;
                ">

                <div style="
                    width:{sc_pct}%;
                    background:{bar_col};
                    height:100%;
                    border-radius:999px;
                "></div>

                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="chunk-box">
                {r["text"]}
                </div>
                """, unsafe_allow_html=True)

elif query and not idx_ok:
    st.error("Build the index first — upload a PDF and use the sidebar.")

elif query and not MODULES_AVAILABLE:
    st.info("Project modules (src/) not found. Connect your codebase to enable live search.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:4rem;padding-top:1rem;
            border-top:1px solid rgba(255,255,255,0.06);
            font-family:'JetBrains Mono',monospace;font-size:0.63rem;
            color:#2d333b;letter-spacing:0.12em;text-align:center;">
  DOCUMENT NAVIGATOR &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; CROSS-ENCODER &nbsp;·&nbsp; TRANSPARENT RAG
</div>
""", unsafe_allow_html=True)

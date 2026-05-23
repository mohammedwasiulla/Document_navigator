# Document Navigator — A Transparent RAG Assistant

Local RAG pipeline over PDFs with transparent retrieval traces, citations, and evaluation.

Quickstart

1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Put your PDFs in the `pdfs/` folder (already present in this workspace).

3. (Optional) Edit chunking params in `src/ingest.py`.

4. Run the Streamlit app and click the sidebar button to build the index:

```bash
streamlit run app.py
```

Evaluation

Compute precision@k with:

```bash
python -m src.evaluate eval_set.csv data/faiss.index
```

Demo Guide

Review `demo_guide.md` for presentation notes, output explanations, and a demo script you can use to show the project.

Notes
- Uses `sentence-transformers` for embeddings and `faiss` for vector search.
- Answers are synthesized simply from retrieved chunks; for richer generation, hook an LLM.

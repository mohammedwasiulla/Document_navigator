# Demo Guide for Document Navigator

This guide helps you explain and demonstrate the project outputs, plus recommended UI adjustments for a cleaner presentation.

## 1. What the outputs mean

### `data/chunks.jsonl`
- One JSON record per text chunk extracted from the PDFs.
- Each record contains:
  - `chunk_id`: internal chunk identifier
  - `source`: PDF filename
  - `page`: page number
  - `chunk_index`: chunk order on the page
  - `text`: the chunk content used for retrieval
- Use it to inspect how documents were split and whether chunking is working.

### `data/faiss.index`
- The FAISS vector index used to search for similar text.
- It stores embeddings for all chunks and enables fast similarity search.

### `data/faiss.index.meta.json`
- Metadata mapping index entries back to the original chunk records.
- Use it to interpret search results by filename/page/text.

### `eval_set.csv`
- A small labeled evaluation set with 15 questions.
- Contains:
  - `question`
  - `gold_citation` (filename:page)
  - `gold_key_phrase`
- Use it for retrieval quality measurement.

### `retrieval_report.md`
- Summary of retrieval evaluation.
- Contains precision@k scores, example successes/failures, and improvement ideas.

### `app.py`
- Streamlit demo app.
- Lets users ask questions, see answers, citations, and retrieval traces.

## 2. What the UI output shows

When you ask a question, the app returns:

1. **Answer**
   - Synthesized from the top retrieved chunk(s).
   - Includes citations in `[filename:page]` format.

2. **Retrieval Traces**
   - Shows the top retrieved chunks by similarity.
   - Each trace includes:
     - `Chunk`: internal chunk id
     - `Source`: PDF name and page
     - `Score`: semantic similarity score
     - `Text`: the actual text retrieved from the PDF

3. **Filters and controls**
   - `Top-k`: how many chunks to retrieve.
   - `Confidence threshold`: removes low-scoring chunks.
   - `Deduplicate by source/page`: keeps one chunk per page.
   - `Use cross-encoder reranker`: optionally improves ordering.

## 3. Output adjustments to show in demo

### A. Use a lower default `Top-k`
- Demonstrate with `Top-k = 3` so results are concise.
- This avoids showing too many low-relevance chunks.

### B. Use `Confidence threshold`
- Set threshold to `0.25` or `0.30`.
- Explain: this hides weak matches and boosts precision.

### C. Enable `Deduplicate by source/page`
- This prevents repeated chunks from the same page.
- It makes retrieval traces easier to read.

### D. Use the reranker for higher precision
- Show the optional `Use cross-encoder reranker` checkbox.
- Explain: it reranks candidates with a stronger semantic scorer.
- Note: it may download a model the first time.

### E. Show the raw chunk text for transparency
- Point to the trace text as the evidence for the answer.
- Highlight the cited chunk and explain how it supports the answer.

## 4. Demonstration script

### Step 1: Start the app
```bash
python -m streamlit run app.py
```

### Step 2: Show the sidebar settings
- `Top-k = 3`
- `Confidence threshold = 0.25`
- `Deduplicate by source/page = ON`
- `Use cross-encoder reranker = OFF` for first demo

### Step 3: Ask a sample question
- Example: "What is the standard delivery timeline?"
- Point out the answer text and the citations.

### Step 4: Explain the top retrieval trace
- Identify the most relevant chunk.
- Say: "This chunk is the evidence. The `score` shows it is the strongest match."

### Step 5: Explain additional chunks
- If other chunks appear, say:
  - "These are other related chunks returned by semantic search."
  - "They may be context, overlap, or related content."

### Step 6: Show the effect of threshold
- Raise `Confidence threshold` to `0.35`.
- Show that some weaker chunks disappear.
- Explain why this increases precision.

### Step 7: Optional reranker demo
- Turn on `Use cross-encoder reranker`.
- Ask the same question again.
- Explain that the app now reranks results using a stronger model.

## 5. How to explain the output to others

### Key messages
- This is a **transparent RAG system**: answers are grounded in actual PDF text.
- We do not just generate text; we show the evidence chunks.
- Retrieval traces make behavior debuggable and trustworthy.
- Evaluation uses `precision@3` and `precision@5` to measure quality.

### What to highlight
- `Answer` + `Citations`: final result plus sources.
- `Retrieval Traces`: the actual evidence behind the answer.
- `Score`: how close the chunks are semantically.
- `Gold citation` in `eval_set.csv`: the human-labeled correct source.

## 6. Example explanation for output

Use this wording while demoing:

- "The system first converts the question into an embedding."
- "It finds the most similar chunks in the PDF index."
- "Then it returns the best evidence and builds an answer from it."
- "The citations show exactly which PDF and page the answer came from."
- "If the evidence is weak, the app can hide low-confidence chunks."

## 7. Optional additional notes

- If you want, you can mention `pdfs/7324429_DEC_2025_PAYSLIP (1).pdf` as an example of a real invoice/payslip being ingested.
- Mention that new PDFs can be added to `pdfs/`, then re-ingested and re-indexed.
- You can also use the browser file upload in the sidebar to add PDFs.

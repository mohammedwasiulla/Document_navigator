import json
import os
import re
from typing import List, Dict


def clean_text(text: str) -> str:
    """Clean PDF-extracted text: remove stray bullets, collapse whitespace, fix broken words."""
    # Remove lines that are only whitespace/bullets/symbols (PyPDF2 artefacts)
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are empty or only bullet/symbol noise
        if not stripped:
            continue
        if re.match(r'^[•\-\*\s]+$', stripped):
            continue
        # Collapse excess internal whitespace
        stripped = re.sub(r'[ \t]{2,}', ' ', stripped)
        cleaned.append(stripped)
    result = '\n'.join(cleaned)
    # Collapse 3+ blank lines to 2
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def extract_pages_from_pdf(path: str) -> List[Dict]:
    """Extract text page-by-page using pdfplumber (layout-aware), fall back to PyPDF2."""
    pages = []
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                except Exception:
                    text = ""
                text = clean_text(text)
                pages.append({"page_num": i, "text": text})
        return pages
    except ImportError:
        pass  # fall back to PyPDF2

    # Fallback: PyPDF2
    from PyPDF2 import PdfReader
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = clean_text(text)
        pages.append({"page_num": i, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")
    # Split on paragraph/section boundaries first, then fall back to character window
    # This keeps semantically related content together
    paragraphs = re.split(r'\n{2,}', text)
    chunks = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 1 <= chunk_size:
            current = (current + "\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            # If paragraph itself is longer than chunk_size, split by character window
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = min(start + chunk_size, len(para))
                    piece = para[start:end].strip()
                    if piece:
                        chunks.append(piece)
                    if end >= len(para):
                        break
                    start = end - overlap
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c.strip()) > 30]


def ingest_pdfs(pdf_dir: str, out_chunks_path: str, chunk_size: int = 1000, overlap: int = 200):
    os.makedirs(os.path.dirname(out_chunks_path), exist_ok=True)
    chunk_id = 0
    with open(out_chunks_path, "w", encoding="utf-8") as fout:
        for fname in sorted(os.listdir(pdf_dir)):
            if not fname.lower().endswith(".pdf"):
                continue
            path = os.path.join(pdf_dir, fname)
            pages = extract_pages_from_pdf(path)
            for p in pages:
                page_num = p["page_num"]
                text = p["text"]
                if not text:
                    continue
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                for i, ch in enumerate(chunks):
                    chunk_id += 1
                    record = {
                        "chunk_id": f"C{chunk_id:06d}",
                        "source": fname,
                        "page": page_num,
                        "chunk_index": i,
                        "text": ch,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("pdf_dir")
    p.add_argument("out_chunks")
    p.add_argument("--chunk_size", type=int, default=1000)
    p.add_argument("--overlap", type=int, default=200)
    args = p.parse_args()
    ingest_pdfs(args.pdf_dir, args.out_chunks, args.chunk_size, args.overlap)

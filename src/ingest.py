import json
import os
from typing import List, Dict
from PyPDF2 import PdfReader


def extract_pages_from_pdf(path: str) -> List[Dict]:
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page_num": i, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # if we've reached the final window, stop to avoid looping when
        # chunk_size > length or when end == length and overlap pushes start
        if end >= length:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


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

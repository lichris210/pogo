import os
import pickle
import boto3
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ── Config ──────────────────────────────────────────────────────────────
DOCS_DIR = "pogo/documents"
INDEX_PATH = "faiss_index/index.faiss"
CHUNKS_PATH = "faiss_index/chunks.pkl"
S3_BUCKET = "pogo-knowledge-base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── Helpers ──────────────────────────────────────────────────────────────
def read_file(path):
    if path.suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return path.read_text(encoding="utf-8", errors="ignore")

def chunk_text(text, source_name):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if len(chunk) > 100:
            chunks.append({"text": chunk, "source": source_name})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("📚 Loading documents...")
    all_chunks = []
    for file_path in sorted(Path(DOCS_DIR).iterdir()):
        if file_path.suffix in [".pdf", ".txt"]:
            print(f"  Reading {file_path.name}...")
            text = read_file(file_path)
            chunks = chunk_text(text, file_path.name)
            all_chunks.extend(chunks)
            print(f"    → {len(chunks)} chunks")

    print(f"\n✅ Total chunks: {len(all_chunks)}")

    print("\n🔢 Generating embeddings (this takes a minute)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("\n🗂️  Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"  Index contains {index.ntotal} vectors")

    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"\n💾 Saved index locally")

    print("\n☁️  Uploading to S3...")
    s3 = boto3.client("s3")
    s3.upload_file(INDEX_PATH, S3_BUCKET, "faiss_index/index.faiss")
    s3.upload_file(CHUNKS_PATH, S3_BUCKET, "faiss_index/chunks.pkl")
    print(f"  ✅ Uploaded to s3://{S3_BUCKET}/faiss_index/")

    print("\n🎉 Ingestion complete!")

if __name__ == "__main__":
    main()

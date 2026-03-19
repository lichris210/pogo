"""
POGO Knowledge Base Indexer
- Sentence-aware chunking (no mid-sentence splits)
- Overlap between chunks for context continuity
- Text cleaning (strip HTML/JSX/markdown artifacts)
- Larger chunks for better semantic embeddings
"""

import os
import re
import json
import pickle
import boto3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# === CONFIG ===
DOCS_DIR = "pogo/documents"
S3_BUCKET = "pogo-knowledge-base"
CHUNK_SIZE = 1200       # ~6-8 sentences per chunk (was 500)
CHUNK_OVERLAP = 200     # overlap to preserve cross-boundary context
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def clean_text(text, source):
    """Remove markup, fix whitespace, strip noise."""
    # Strip HTML/JSX tags
    text = re.sub(r'<[^>]+>', '', text)
    # Strip markdown image syntax ![...](...) 
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Strip markdown link syntax but keep text [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Strip markdown headers (keep the text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Strip horizontal rules
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    # Strip leading/trailing whitespace per line
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()


def split_into_sentences(text):
    """Split text into sentences, handling common abbreviations."""
    # Split on sentence endings followed by space + uppercase or newline
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\d])', text)
    # Also split on double newlines (paragraph breaks)
    result = []
    for s in sentences:
        parts = s.split('\n\n')
        result.extend(p.strip() for p in parts if p.strip())
    return result


def chunk_sentences(sentences, source, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Build chunks from sentences with overlap, respecting size limits."""
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If a single sentence exceeds chunk_size, add it as its own chunk
        if sentence_len > chunk_size:
            if current_chunk:
                chunks.append({
                    "text": ' '.join(current_chunk),
                    "source": source
                })
            chunks.append({
                "text": sentence[:chunk_size],
                "source": source
            })
            current_chunk = []
            current_length = 0
            continue

        # If adding this sentence would exceed the limit, save current chunk
        if current_length + sentence_len + 1 > chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "source": source
            })

            # Overlap: keep trailing sentences that fit within overlap size
            overlap_chunk = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) + 1 > overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_len += len(s) + 1

            current_chunk = overlap_chunk
            current_length = overlap_len

        current_chunk.append(sentence)
        current_length += sentence_len + 1

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text) > 50:  # skip tiny trailing fragments
            chunks.append({
                "text": chunk_text,
                "source": source
            })

    return chunks


def read_document(filepath):
    """Read a document file (txt or pdf)."""
    if filepath.endswith('.pdf'):
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except ImportError:
            print(f"  [WARN] PyMuPDF not installed, skipping {filepath}")
            print(f"         Run: pip install PyMuPDF")
            return None
    else:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()


def process_all_documents(docs_dir):
    """Process all documents and return chunks."""
    all_chunks = []

    for filename in sorted(os.listdir(docs_dir)):
        filepath = os.path.join(docs_dir, filename)
        if not os.path.isfile(filepath):
            continue
        if not (filename.endswith('.txt') or filename.endswith('.pdf')):
            continue

        print(f"Processing: {filename}")
        text = read_document(filepath)
        if not text:
            continue

        # Clean
        cleaned = clean_text(text, filename)

        # Split into sentences
        sentences = split_into_sentences(cleaned)
        print(f"  Sentences: {len(sentences)}")

        # Chunk with overlap
        chunks = chunk_sentences(sentences, filename)
        print(f"  Chunks: {len(chunks)}, avg size: {sum(len(c['text']) for c in chunks) // max(len(chunks), 1)} chars")

        all_chunks.extend(chunks)

    return all_chunks


def build_faiss_index(chunks, model_name=EMBEDDING_MODEL):
    """Embed all chunks and build FAISS index."""
    print(f"\nEmbedding {len(chunks)} chunks with {model_name}...")
    embedder = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product on normalized = cosine sim
    index.add(embeddings)

    print(f"Index built: {index.ntotal} vectors, {dimension} dimensions")
    return index


def save_local(index, chunks, output_dir="pogo/faiss_output"):
    """Save index and chunks locally."""
    os.makedirs(output_dir, exist_ok=True)

    index_path = os.path.join(output_dir, "index.faiss")
    faiss.write_index(index, index_path)
    print(f"Saved: {index_path}")

    chunks_path = os.path.join(output_dir, "chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved: {chunks_path}")

    return index_path, chunks_path


def upload_to_s3(index_path, chunks_path, bucket=S3_BUCKET):
    """Upload index and chunks to S3."""
    s3 = boto3.client("s3")

    s3.upload_file(index_path, bucket, "faiss_index/index.faiss")
    print(f"Uploaded: s3://{bucket}/faiss_index/index.faiss")

    s3.upload_file(chunks_path, bucket, "faiss_index/chunks.pkl")
    print(f"Uploaded: s3://{bucket}/faiss_index/chunks.pkl")


def print_summary(chunks):
    """Print a summary of the indexed knowledge base."""
    print("\n" + "=" * 60)
    print("POGO KNOWLEDGE BASE SUMMARY")
    print("=" * 60)
    sources = {}
    for c in chunks:
        src = c["source"]
        sources[src] = sources.get(src, 0) + 1

    for src in sorted(sources.keys()):
        avg_len = sum(len(c["text"]) for c in chunks if c["source"] == src) // sources[src]
        print(f"  {src:40s} {sources[src]:4d} chunks  (avg {avg_len} chars)")

    sizes = [len(c["text"]) for c in chunks]
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Chunk sizes — min: {min(sizes)}, max: {max(sizes)}, avg: {sum(sizes)//len(sizes)}")
    print("=" * 60)


if __name__ == "__main__":
    # 1. Process documents
    chunks = process_all_documents(DOCS_DIR)

    # 2. Build FAISS index
    index = build_faiss_index(chunks)

    # 3. Save locally
    index_path, chunks_path = save_local(index, chunks)

    # 4. Print summary
    print_summary(chunks)

    # 5. Upload to S3
    confirm = input("\nUpload to S3? (y/n): ").strip().lower()
    if confirm == "y":
        upload_to_s3(index_path, chunks_path)
        print("\nDone! POGO knowledge base updated.")
    else:
        print("\nSkipped S3 upload. Local files saved in pogo/faiss_output/")

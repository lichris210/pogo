"""
POGO Knowledge Base Indexer (Lambda-compatible version)
- Uses Bedrock Titan Embeddings (no local model needed)
- Stores as numpy array (no FAISS dependency)
- Sentence-aware chunking with overlap
"""

import os
import re
import json
import pickle
import time
import boto3
import numpy as np

# === CONFIG ===
DOCS_DIR = "pogo/documents"
S3_BUCKET = "pogo-knowledge-base"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_BATCH_DELAY = 0.05  # small delay to avoid throttling


def clean_text(text, source):
    """Remove markup, fix whitespace, strip noise."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()


def split_into_sentences(text):
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\d])', text)
    result = []
    for s in sentences:
        parts = s.split('\n\n')
        result.extend(p.strip() for p in parts if p.strip())
    return result


def chunk_sentences(sentences, source, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Build chunks from sentences with overlap."""
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

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

        if current_length + sentence_len + 1 > chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "source": source
            })

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

    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text) > 50:
            chunks.append({
                "text": chunk_text,
                "source": source
            })

    return chunks


def read_document(filepath):
    """Read a document file (txt or pdf)."""
    if filepath.endswith('.pdf'):
        try:
            import fitz
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except ImportError:
            print(f"  [WARN] PyMuPDF not installed, skipping {filepath}")
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

        cleaned = clean_text(text, filename)
        sentences = split_into_sentences(cleaned)
        print(f"  Sentences: {len(sentences)}")

        chunks = chunk_sentences(sentences, filename)
        print(f"  Chunks: {len(chunks)}, avg size: {sum(len(c['text']) for c in chunks) // max(len(chunks), 1)} chars")

        all_chunks.extend(chunks)

    return all_chunks


def embed_with_titan(texts, bedrock_client, batch_size=10):
    """Embed texts using Bedrock Titan Embeddings."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = []

        for text in batch:
            # Titan has a 8192 token limit, truncate long texts
            truncated = text[:8000]
            response = bedrock_client.invoke_model(
                modelId=EMBED_MODEL_ID,
                body=json.dumps({
                    "inputText": truncated,
                    "dimensions": 256,
                    "normalize": True
                })
            )
            result = json.loads(response["body"].read())
            batch_embeddings.append(result["embedding"])
            time.sleep(EMBED_BATCH_DELAY)

        all_embeddings.extend(batch_embeddings)

        done = min(i + batch_size, len(texts))
        print(f"  Embedded {done}/{len(texts)} chunks...", end='\r')

    print()
    return np.array(all_embeddings, dtype="float32")


def save_local(embeddings, chunks, output_dir="pogo/faiss_output"):
    """Save embeddings and chunks locally."""
    os.makedirs(output_dir, exist_ok=True)

    emb_path = os.path.join(output_dir, "embeddings.npy")
    np.save(emb_path, embeddings)
    print(f"Saved: {emb_path}")

    chunks_path = os.path.join(output_dir, "chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved: {chunks_path}")

    return emb_path, chunks_path


def upload_to_s3(emb_path, chunks_path, bucket=S3_BUCKET):
    """Upload to S3."""
    s3 = boto3.client("s3")

    s3.upload_file(emb_path, bucket, "index/embeddings.npy")
    print(f"Uploaded: s3://{bucket}/index/embeddings.npy")

    s3.upload_file(chunks_path, bucket, "index/chunks.pkl")
    print(f"Uploaded: s3://{bucket}/index/chunks.pkl")


def print_summary(chunks, embeddings):
    """Print summary."""
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
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Chunk sizes — min: {min(sizes)}, max: {max(sizes)}, avg: {sum(sizes)//len(sizes)}")
    print("=" * 60)


if __name__ == "__main__":
    # 1. Process documents
    chunks = process_all_documents(DOCS_DIR)

    # 2. Embed with Bedrock Titan
    print(f"\nEmbedding {len(chunks)} chunks with Bedrock Titan...")
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    embeddings = embed_with_titan([c["text"] for c in chunks], bedrock)

    # 3. Save locally
    emb_path, chunks_path = save_local(embeddings, chunks)

    # 4. Print summary
    print_summary(chunks, embeddings)

    # 5. Upload to S3
    confirm = input("\nUpload to S3? (y/n): ").strip().lower()
    if confirm == "y":
        upload_to_s3(emb_path, chunks_path)
        print("\nDone! POGO knowledge base updated (Titan embeddings).")
    else:
        print("\nSkipped S3 upload. Local files saved in pogo/faiss_output/")

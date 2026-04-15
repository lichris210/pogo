"""S3-backed vector store for the prompt DB.

Storage layout on S3 (same bucket as the v1 research-paper index, but a
separate prefix so the old index is untouched)::

    s3://pogo-knowledge-base/prompt_db/prompts.json
    s3://pogo-knowledge-base/prompt_db/embeddings.npy

Records are kept as JSON for easy inspection; embeddings are a stacked
``float32`` array whose row *i* corresponds to record *i*.

For local development / unit tests, the store falls back to a directory
on the local filesystem when ``POGO_PROMPT_DB_LOCAL_DIR`` is set.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np

from prompt_db.schema import PromptRecord

S3_BUCKET = os.environ.get("POGO_KNOWLEDGE_BUCKET", "pogo-knowledge-base")
PROMPTS_KEY = "prompt_db/prompts.json"
EMBEDDINGS_KEY = "prompt_db/embeddings.npy"

_LOCAL_ENV = "POGO_PROMPT_DB_LOCAL_DIR"


# ---------------------------------------------------------------------------
# S3 client (cached)
# ---------------------------------------------------------------------------

_s3 = None


def _get_s3():
    global _s3
    if _s3 is None:
        import boto3

        _s3 = boto3.client("s3")
    return _s3


def _local_dir() -> Path | None:
    raw = os.environ.get(_LOCAL_ENV)
    return Path(raw) if raw else None


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------

def load() -> tuple[list[PromptRecord], np.ndarray]:
    """Load all records and their embeddings.

    Returns ``(records, embeddings)``. If the store is empty or missing,
    returns an empty list and an empty ``(0, 0)`` array.
    """
    local = _local_dir()
    if local is not None:
        return _load_local(local)
    return _load_s3()


def save(records: list[PromptRecord], embeddings: np.ndarray) -> None:
    """Persist *records* and *embeddings* to the store.

    The two arrays must be the same length (row *i* of *embeddings*
    corresponds to *records[i]*).
    """
    if len(records) != embeddings.shape[0]:
        raise ValueError(
            f"records ({len(records)}) and embeddings ({embeddings.shape[0]}) "
            "length mismatch"
        )

    local = _local_dir()
    if local is not None:
        _save_local(local, records, embeddings)
        return
    _save_s3(records, embeddings)


# ---------------------------------------------------------------------------
# Local filesystem backend
# ---------------------------------------------------------------------------

def _load_local(root: Path) -> tuple[list[PromptRecord], np.ndarray]:
    prompts_path = root / "prompts.json"
    embeds_path = root / "embeddings.npy"
    if not prompts_path.exists() or not embeds_path.exists():
        return [], np.zeros((0, 0), dtype="float32")

    with prompts_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    records = [PromptRecord.from_dict(r) for r in raw]
    embeddings = np.load(embeds_path).astype("float32")
    return records, embeddings


def _save_local(
    root: Path, records: list[PromptRecord], embeddings: np.ndarray
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "prompts.json").open("w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)
    np.save(root / "embeddings.npy", embeddings.astype("float32"))


# ---------------------------------------------------------------------------
# S3 backend
# ---------------------------------------------------------------------------

def _load_s3() -> tuple[list[PromptRecord], np.ndarray]:
    s3 = _get_s3()
    try:
        prompts_obj = s3.get_object(Bucket=S3_BUCKET, Key=PROMPTS_KEY)
        emb_obj = s3.get_object(Bucket=S3_BUCKET, Key=EMBEDDINGS_KEY)
    except Exception as e:  # pragma: no cover — AWS-dependent path
        print(f"prompt_db.store: S3 load failed: {e}")
        return [], np.zeros((0, 0), dtype="float32")

    raw = json.loads(prompts_obj["Body"].read().decode("utf-8"))
    records = [PromptRecord.from_dict(r) for r in raw]
    embeddings = np.load(io.BytesIO(emb_obj["Body"].read())).astype("float32")
    return records, embeddings


def _save_s3(records: list[PromptRecord], embeddings: np.ndarray) -> None:
    s3 = _get_s3()

    payload = json.dumps([r.to_dict() for r in records], indent=2).encode("utf-8")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=PROMPTS_KEY,
        Body=payload,
        ContentType="application/json",
    )

    buf = io.BytesIO()
    np.save(buf, embeddings.astype("float32"))
    buf.seek(0)
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=EMBEDDINGS_KEY,
        Body=buf.getvalue(),
        ContentType="application/octet-stream",
    )

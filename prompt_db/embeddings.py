"""Bedrock Titan embedding client for the prompt DB.

Thin wrapper around :func:`bedrock-runtime.invoke_model` that reuses the
same model ID and parameters as ``lambda/handler.py`` (Titan Embed Text
v2, 256 dimensions, normalised).  Kept in its own module so ingest /
retrieve share one cached client and tests can monkey-patch a single
symbol.
"""

from __future__ import annotations

import json

import numpy as np

EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 256

_bedrock = None


def _get_bedrock():
    """Return a cached ``bedrock-runtime`` client."""
    global _bedrock
    if _bedrock is None:
        import boto3

        _bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    return _bedrock


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string with Bedrock Titan.

    Returns a normalised ``float32`` vector of length :data:`EMBED_DIM`.
    """
    bedrock = _get_bedrock()
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps(
            {
                "inputText": (text or "")[:8000],
                "dimensions": EMBED_DIM,
                "normalize": True,
            }
        ),
    )
    result = json.loads(response["body"].read())
    return np.array(result["embedding"], dtype="float32")


def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a list of texts. Titan has no batch endpoint, so this loops."""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32")
    vectors = [embed_text(t) for t in texts]
    return np.stack(vectors).astype("float32")

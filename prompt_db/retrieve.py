"""Retrieval interface for the prompt DB.

The orchestrator queries the prompt DB *on behalf* of each agent using
the classified task category — not the user's raw input. This keeps
retrieval focused on prompt-engineering patterns for that task type
rather than matching the user's surface wording.

Public API:

- :func:`retrieve_reference_prompts` — top-k high-quality reference
  prompts filtered by target model.
- :func:`retrieve_few_shot_examples` — few-shot example dicts pulled
  from records that have non-empty ``few_shot_examples``.
- :func:`retrieve_similar_prompts` — top-k nearest prompts for an
  arbitrary embedding-text query, including cosine similarity scores.

Both helpers are resilient: if the store is empty or AWS is unreachable
(e.g. in local unit tests), they log a warning and return ``[]``.
"""

from __future__ import annotations

import numpy as np

from prompt_db.embeddings import embed_text
from prompt_db.schema import PromptRecord
from prompt_db.store import load

# ---------------------------------------------------------------------------
# In-process cache
# ---------------------------------------------------------------------------

_cache: tuple[list[PromptRecord], np.ndarray] | None = None


def _load_cached() -> tuple[list[PromptRecord], np.ndarray]:
    global _cache
    if _cache is None:
        _cache = load()
    return _cache


def reset_cache() -> None:
    """Drop the in-process cache. Used by tests and by ingest_single_prompt."""
    global _cache
    _cache = None


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------

_QUERY_TEMPLATES: dict[str, str] = {
    "data_analysis": (
        "best practices for data_analysis prompts: schema grounding, "
        "structured output, classification, extraction, reasoning over tables"
    ),
    "code_generation": (
        "best practices for code_generation prompts: role prompting, "
        "step-by-step reasoning, type hints, tests, structured output"
    ),
    "writing": (
        "best practices for writing prompts: audience, tone, style guides, "
        "summarization, editing, translation"
    ),
    "creative": (
        "best practices for creative writing prompts: persona, tone, "
        "narrative constraints, style"
    ),
    "web_development": (
        "best practices for web_development prompts: component structure, "
        "frameworks, UI/UX constraints, HTML/CSS/JS"
    ),
    "research": (
        "best practices for research prompts: reasoning, analysis, "
        "comparison, synthesis, investigation"
    ),
    "general": (
        "best practices for general prompts: clarity, specificity, "
        "role assignment, structured output, chain-of-thought"
    ),
}


def _build_query(task_category: str) -> str:
    return _QUERY_TEMPLATES.get(
        task_category,
        f"best practices for {task_category} prompts: structure, clarity, format",
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_reference_prompts(
    task_category: str,
    target_model: str,
    k: int = 3,
) -> list[PromptRecord]:
    """Return top-k reference prompts for *task_category* + *target_model*.

    Results are ranked by cosine similarity of the Titan embedding of a
    task-category query string against the stored embeddings, then
    filtered to ``target_model``. The list is ordered best-first.

    If the store is empty or the embedding call fails, returns ``[]``.
    """
    records, embeddings = _load_cached()
    if not records or embeddings.size == 0:
        return []

    # Pre-filter by target model so we don't waste similarity scoring on
    # records that will be dropped anyway.
    idx_keep = [
        i for i, r in enumerate(records) if r.target_model == target_model
    ]
    if not idx_keep:
        return []

    try:
        query_vec = embed_text(_build_query(task_category))
    except Exception as e:
        print(f"retrieve_reference_prompts: embedding failed ({e})")
        return []

    filtered_emb = embeddings[idx_keep]
    # Vectors from Titan Embed v2 with normalize=True are unit-length,
    # so the dot product is cosine similarity.
    sims = filtered_emb @ query_vec

    # Sort within the filtered slice, then map back to original indices.
    order = np.argsort(sims)[::-1][:k]
    return [records[idx_keep[i]] for i in order]


def retrieve_few_shot_examples(
    task_category: str,
    target_model: str,
    k: int = 2,
) -> list[dict]:
    """Return up to *k* few-shot example dicts for the given task.

    Only records with a non-empty ``few_shot_examples`` field are
    considered. Ranking reuses :func:`retrieve_reference_prompts`
    with a larger candidate pool so we can still find *k* examples even
    when most top-ranked records lack explicit few-shot examples.
    """
    records, embeddings = _load_cached()
    if not records or embeddings.size == 0:
        return []

    # Retrieve a larger pool to account for records without examples.
    pool = retrieve_reference_prompts(task_category, target_model, k=max(k * 4, 8))

    examples: list[dict] = []
    for record in pool:
        if not record.few_shot_examples:
            continue
        for ex in record.few_shot_examples:
            examples.append(ex)
            if len(examples) >= k:
                return examples
    return examples


def retrieve_similar_prompts(
    query_text: str,
    *,
    k: int = 3,
    task_category: str | None = None,
    target_model: str | None = None,
) -> list[tuple[PromptRecord, float]]:
    """Return the nearest stored prompts for *query_text*.

    Args:
        query_text: Arbitrary text to embed and compare against the store.
        k: Number of neighbors to return.
        task_category: Optional filter on ``PromptRecord.task_category``.
        target_model: Optional filter on ``PromptRecord.target_model``.

    Returns:
        A best-first list of ``(PromptRecord, cosine_similarity)`` tuples.
        Returns ``[]`` if the store is empty or the embedding call fails.
    """
    records, embeddings = _load_cached()
    if not records or embeddings.size == 0:
        return []

    idx_keep: list[int] = []
    for idx, record in enumerate(records):
        if task_category and record.task_category != task_category:
            continue
        if target_model and record.target_model != target_model:
            continue
        idx_keep.append(idx)

    if not idx_keep:
        return []

    try:
        query_vec = embed_text(query_text)
    except Exception as e:
        print(f"retrieve_similar_prompts: embedding failed ({e})")
        return []

    filtered_emb = embeddings[idx_keep]
    sims = filtered_emb @ query_vec
    order = np.argsort(sims)[::-1][:k]
    return [
        (records[idx_keep[i]], float(sims[i]))
        for i in order
    ]

"""Ingest prompts into the prompt DB.

Two entry points:

- :func:`ingest_seed_data` — bulk-ingest the curated ``seed_prompts.json``
  file shipped with the repo. Normalises external task categories and
  target-model names into the canonical POGO vocabulary.

- :func:`ingest_single_prompt` — add a single :class:`PromptRecord` to an
  existing store, used by the orchestrator's "accepted" state handler.

Runnable as a script::

    python -m prompt_db.ingest --seed-file seed_prompts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prompt_db.embeddings import embed_batch, embed_text
from prompt_db.schema import PromptRecord, to_embedding_text
from prompt_db.store import load, save

# ---------------------------------------------------------------------------
# Seed-data normalisation
# ---------------------------------------------------------------------------

# Seed categories → canonical POGO categories.
_CATEGORY_MAP: dict[str, str] = {
    "code_generation": "code_generation",
    "classification": "data_analysis",
    "analysis": "data_analysis",
    "creative_writing": "creative",
    "data_transformation": "data_analysis",
    "reasoning": "research",
    "extraction": "data_analysis",
    "summarization": "writing",
    "agentic_workflow": "general",
    "translation": "writing",
    "multimodal": "general",
    # Pass-through entries for already-canonical values
    "data_analysis": "data_analysis",
    "writing": "writing",
    "creative": "creative",
    "web_development": "web_development",
    "research": "research",
    "general": "general",
}


def _normalise_target_model(raw: str) -> str:
    """Map external model names (e.g. ``claude-opus-4-6``) to the canonical
    POGO family name (``claude``/``gpt``/``gemini``)."""
    s = (raw or "").lower()
    if "claude" in s:
        return "claude"
    if "gpt" in s or "openai" in s:
        return "gpt"
    if "gemini" in s:
        return "gemini"
    return "claude"


def _infer_format(target_model: str, prompt_text: str) -> str:
    """Best-effort guess at the prompt's wire format."""
    if target_model == "claude" and "<" in prompt_text and ">" in prompt_text:
        return "xml"
    return "markdown"


def _split_system_user(prompt_text: str) -> tuple[str, str]:
    """Split seed ``prompt_text`` into (system_prompt, user_prompt_template).

    The curated seed follows the convention ``System: ...\\n\\nUser:\\n...``.
    If no such split exists, the entire text becomes the user template.
    """
    text = prompt_text or ""
    if text.lower().startswith("system:"):
        # Strip the "System:" marker, then split on the User boundary.
        body = text[len("system:"):].lstrip()
        lower = body.lower()
        idx = lower.find("\nuser:")
        if idx != -1:
            system = body[:idx].strip()
            user = body[idx + len("\nuser:"):].lstrip()
            return system, user
        return body.strip(), ""
    return "", text.strip()


def _seed_to_record(raw: dict) -> PromptRecord:
    """Convert a seed JSON dict into a :class:`PromptRecord`."""
    raw_category = raw.get("task_category", "general")
    task_category = _CATEGORY_MAP.get(raw_category, "general")
    subcategory = raw_category  # keep the original label for granularity

    target_model = _normalise_target_model(raw.get("target_model", "claude"))
    system_prompt, user_template = _split_system_user(raw.get("prompt_text", ""))
    fmt = _infer_format(target_model, raw.get("prompt_text", ""))

    return PromptRecord(
        id=raw["id"],
        task_category=task_category,
        subcategory=subcategory,
        target_model=target_model,
        format=fmt,
        techniques=list(raw.get("techniques_used", [])),
        system_prompt=system_prompt,
        user_prompt_template=user_template,
        few_shot_examples=list(raw.get("few_shot_examples", [])),
        quality_score=float(raw.get("quality_score", 0.85)),
        source=raw.get("source_type", "curated"),
        created_at="",
    )


# ---------------------------------------------------------------------------
# Bulk ingestion
# ---------------------------------------------------------------------------

def ingest_seed_data(seed_file_path: str, *, overwrite: bool = True) -> int:
    """Ingest a seed JSON file into the prompt DB.

    Args:
        seed_file_path: Path to a JSON file containing a list of seed prompt
            dicts (see ``seed_prompts.json`` for the expected shape).
        overwrite: If ``True`` (default), replaces the entire store with the
            seed records. If ``False``, appends to the existing store.

    Returns:
        The number of records ingested.
    """
    with Path(seed_file_path).open("r", encoding="utf-8") as f:
        raw_records = json.load(f)

    print(f"Loaded {len(raw_records)} seed records from {seed_file_path}")

    records: list[PromptRecord] = []
    for raw in raw_records:
        try:
            record = _seed_to_record(raw)
            record.source = "curated"  # force curated for seed data
            if record.quality_score <= 0.0:
                record.quality_score = 0.85
            record.validate()
            records.append(record)
        except Exception as e:
            print(f"  skipping {raw.get('id', '?')}: {e}")

    print(f"Validated {len(records)} records. Generating embeddings...")

    texts = [to_embedding_text(r) for r in records]
    embeddings = embed_batch(texts)

    if not overwrite:
        existing_records, existing_emb = load()
        records = existing_records + records
        if existing_emb.size and embeddings.size:
            embeddings = np.vstack([existing_emb, embeddings]).astype("float32")
        elif existing_emb.size:
            embeddings = existing_emb

    save(records, embeddings)
    print(f"Ingested {len(records)} records.")
    return len(records)


# ---------------------------------------------------------------------------
# Single-record ingestion
# ---------------------------------------------------------------------------

def ingest_single_prompt(record: PromptRecord) -> bool:
    """Append *record* to the prompt DB.

    Validates the record, computes its embedding, and persists.
    Returns ``True`` on success.
    """
    record.validate()

    vec = embed_text(to_embedding_text(record))
    records, embeddings = load()

    records.append(record)
    if embeddings.size == 0:
        embeddings = vec.reshape(1, -1)
    else:
        embeddings = np.vstack([embeddings, vec.reshape(1, -1)]).astype("float32")

    save(records, embeddings)
    print(f"Ingested prompt {record.id} ({record.task_category}/{record.target_model}).")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:  # pragma: no cover — CLI glue
    parser = argparse.ArgumentParser(description="Ingest prompts into the POGO prompt DB.")
    parser.add_argument(
        "--seed-file",
        required=True,
        help="Path to a seed JSON file (list of prompt records).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the existing store instead of overwriting.",
    )
    args = parser.parse_args()
    ingest_seed_data(args.seed_file, overwrite=not args.append)


if __name__ == "__main__":  # pragma: no cover
    _main()

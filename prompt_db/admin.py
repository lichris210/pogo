"""Administrative utilities for the prompt DB.

Examples:

    python -m prompt_db.admin --list --category data_analysis
    python -m prompt_db.admin --remove user_abcd1234
    python -m prompt_db.admin --update-score user_abcd1234 --score 0.92
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from prompt_db.retrieve import reset_cache
from prompt_db.schema import PromptRecord
from prompt_db.store import load, save


def list_prompts(
    task_category: str | None = None,
    source: str | None = None,
) -> list[PromptRecord]:
    """Return stored prompts filtered by category and/or source."""
    records, _ = load()
    out: list[PromptRecord] = []
    for record in records:
        if task_category and record.task_category != task_category:
            continue
        if source and record.source != source:
            continue
        out.append(record)
    return out


def remove_prompt(prompt_id: str) -> bool:
    """Remove a prompt and its embedding row by ID."""
    records, embeddings = load()
    keep_records: list[PromptRecord] = []
    keep_indices: list[int] = []
    removed = False

    for idx, record in enumerate(records):
        if record.id == prompt_id:
            removed = True
            continue
        keep_records.append(record)
        keep_indices.append(idx)

    if not removed:
        return False

    new_embeddings = _select_rows(embeddings, keep_indices)
    save(keep_records, new_embeddings)
    reset_cache()
    return True


def update_score(prompt_id: str, new_score: float) -> bool:
    """Update a prompt's quality score in-place."""
    if not 0.0 <= float(new_score) <= 1.0:
        raise ValueError("new_score must be in [0.0, 1.0]")

    records, embeddings = load()
    updated = False
    for record in records:
        if record.id == prompt_id:
            record.quality_score = float(new_score)
            updated = True
            break

    if not updated:
        return False

    save(records, embeddings)
    reset_cache()
    return True


def _select_rows(embeddings: np.ndarray, indices: list[int]) -> np.ndarray:
    if embeddings.size == 0 or not indices:
        cols = embeddings.shape[1] if embeddings.ndim == 2 and embeddings.shape[1:] else 0
        return np.zeros((0, cols), dtype="float32")
    return embeddings[indices].astype("float32")


def _main() -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description="Admin tools for the POGO prompt DB.")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--list", action="store_true", help="List prompt records.")
    actions.add_argument("--remove", metavar="PROMPT_ID", help="Remove a prompt by ID.")
    actions.add_argument(
        "--update-score",
        metavar="PROMPT_ID",
        help="Update a prompt's quality score by ID.",
    )
    parser.add_argument("--category", help="Filter list results by task category.")
    parser.add_argument("--source", help="Filter list results by source.")
    parser.add_argument(
        "--score",
        type=float,
        help="New score for --update-score (must be between 0.0 and 1.0).",
    )
    args = parser.parse_args()

    if args.list:
        records = list_prompts(task_category=args.category, source=args.source)
        print(json.dumps([record.to_dict() for record in records], indent=2))
        return

    if args.remove:
        removed = remove_prompt(args.remove)
        if not removed:
            raise SystemExit(f"Prompt not found: {args.remove}")
        print(f"Removed prompt {args.remove}.")
        return

    if args.update_score:
        if args.score is None:
            raise SystemExit("--score is required with --update-score")
        updated = update_score(args.update_score, args.score)
        if not updated:
            raise SystemExit(f"Prompt not found: {args.update_score}")
        print(f"Updated score for {args.update_score} to {args.score:.2f}.")


if __name__ == "__main__":  # pragma: no cover
    _main()

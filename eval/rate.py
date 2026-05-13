"""POGO v2 eval rating CLI.

Walks an eval run file, prompts the operator to rate each unrated entry on
a 1-5 scale, and writes ratings back to the same file. Quitting mid-session
preserves the ratings already entered.

Usage:
    python eval/rate.py eval/runs/<file>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from textwrap import indent
from typing import Callable

RATING_SCALE = {
    1: "Unusable or wrong",
    2: "Below baseline quality",
    3: "Acceptable (baseline expectation)",
    4: "Above baseline expectation",
    5: "Significantly better than baseline",
}

QUIT_TOKENS = {"q", "quit", "exit"}


def load_run(path: Path) -> dict:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or "results" not in raw:
        raise ValueError(
            f"{path} is not a valid run file (missing 'results' array)"
        )
    return raw


def save_run(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def validate_score(value: str) -> int:
    """Parse and validate a score input.

    Raises:
        ValueError: when *value* is not a valid integer in ``1..5``.
    """
    stripped = (value or "").strip()
    if not stripped:
        raise ValueError("score is required")
    try:
        score = int(stripped)
    except ValueError as exc:
        raise ValueError(f"score must be an integer 1-5 (got {value!r})") from exc
    if score not in RATING_SCALE:
        raise ValueError(f"score must be in 1..5 (got {score})")
    return score


def apply_rating(
    payload: dict,
    eval_id: str,
    score: int,
    notes: str | None,
    *,
    now: datetime | None = None,
) -> dict:
    """Mutate *payload* to attach a rating to *eval_id* and return it.

    Pure-ish helper extracted so tests can exercise the file-update logic
    without driving stdin/stdout.
    """
    timestamp = (now or datetime.now(timezone.utc)).isoformat()
    found = False
    for entry in payload.get("results", []):
        if entry.get("eval_id") == eval_id:
            entry["rating"] = {
                "score": score,
                "notes": notes or None,
                "rated_at": timestamp,
            }
            found = True
            break
    if not found:
        raise KeyError(f"eval_id {eval_id!r} not found in run file")
    return payload


def _format_captured(captured: dict) -> str:
    lines = []
    if captured.get("skipped"):
        lines.append(f"SKIPPED — {captured.get('skip_reason') or 'no reason given'}")
        if captured.get("error"):
            lines.append(f"  error: {captured['error']}")
        return "\n".join(lines)

    score = captured.get("critic_score")
    if isinstance(score, (int, float)):
        lines.append(f"Critic score: {score:.2f}")
    elapsed = captured.get("elapsed_seconds")
    tokens = captured.get("token_count")
    if elapsed is not None:
        lines.append(f"Elapsed: {elapsed:.1f}s  |  Tokens: {tokens}")
    final_output = captured.get("final_output") or "(empty)"
    lines.append("\nFinal output:")
    lines.append(indent(final_output, "  "))
    feedback = captured.get("critic_feedback")
    if feedback:
        lines.append("\nCritic feedback:")
        lines.append(indent(feedback, "  "))
    return "\n".join(lines)


def _print_scale() -> None:
    print("Rating scale:")
    for value, label in RATING_SCALE.items():
        print(f"  {value}: {label}")


def rate_interactively(
    path: Path,
    *,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> int:
    """Walk unrated entries in *path* and prompt for ratings.

    Returns the number of entries rated in this session.
    """
    payload = load_run(path)
    results = payload.get("results", [])
    unrated = [r for r in results if r.get("rating", {}).get("score") is None]

    if not unrated:
        print_fn(f"All {len(results)} entries already rated. Nothing to do.")
        return 0

    print_fn(f"Rating file: {path}")
    print_fn(f"{len(unrated)} of {len(results)} entries still need rating.")
    print_fn("Enter 'q' or 'quit' at any prompt to stop and preserve progress.\n")
    _print_scale()
    print_fn("")

    rated_count = 0
    for i, entry in enumerate(unrated, start=1):
        eid = entry.get("eval_id")
        inp = entry.get("input", {})
        print_fn("=" * 72)
        print_fn(f"[{i}/{len(unrated)}] {eid}  ({inp.get('task_category')}/{inp.get('expected_path')})")
        print_fn(f"Target family: {inp.get('target_model_family')}")
        print_fn(f"\nUser prompt:\n{indent(inp.get('user_prompt', ''), '  ')}")
        print_fn(f"\nNotes: {inp.get('notes', '(none)')}")
        print_fn("")
        print_fn(_format_captured(entry.get("captured", {})))
        print_fn("")

        while True:
            raw = input_fn("Score 1-5 (or q to quit): ")
            if raw.strip().lower() in QUIT_TOKENS:
                save_run(path, payload)
                print_fn(f"Stopped. Rated {rated_count} this session. Progress saved.")
                return rated_count
            try:
                score = validate_score(raw)
                break
            except ValueError as exc:
                print_fn(f"  invalid: {exc}")

        notes_raw = input_fn("Notes (optional, 'q' to quit without saving this entry): ")
        if notes_raw.strip().lower() in QUIT_TOKENS:
            save_run(path, payload)
            print_fn(f"Stopped before saving entry {eid}. Rated {rated_count} this session.")
            return rated_count
        notes = notes_raw.strip() or None

        apply_rating(payload, eid, score, notes)
        save_run(path, payload)
        rated_count += 1
        print_fn(f"  saved: score={score}\n")

    print_fn(f"Done. Rated {rated_count} entries this session.")
    return rated_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rate POGO eval run outputs")
    parser.add_argument("run_file", type=Path, help="Path to a run file under eval/runs/")
    args = parser.parse_args(argv)

    if not args.run_file.exists():
        print(f"error: {args.run_file} does not exist", file=sys.stderr)
        return 2

    try:
        rate_interactively(args.run_file)
    except KeyboardInterrupt:
        print("\ninterrupted — progress saved up to the last completed entry")
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

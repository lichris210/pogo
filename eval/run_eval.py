"""POGO v2 eval runner.

Loads eval/inputs.json, drives the orchestrator programmatically for each
entry, captures pipeline outputs, and writes a versioned run file under
eval/runs/. The runner uses the agent_router and orchestrator entry points
directly — it does NOT call the deployed Lambda.

Each entry's pre_baked_context is fed as the user reply that would
normally come back from the Clarifier interaction. Entries without
pre_baked_context are skipped-and-logged. Infrastructure failures
(Anthropic API unreachable, missing API key, etc.) also skip-and-log
rather than failing the run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from unittest.mock import MagicMock, patch

# Ensure repo root is importable when run as `python eval/run_eval.py`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INPUTS = REPO_ROOT / "eval" / "inputs.json"
DEFAULT_RUNS_DIR = REPO_ROOT / "eval" / "runs"


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------

@dataclass
class TokenCounter:
    """Accumulates Anthropic token usage across a single eval entry."""

    total: int = 0
    input: int = 0
    output: int = 0

    def add(self, usage: dict | None) -> None:
        if not isinstance(usage, dict):
            return
        self.input += int(usage.get("input_tokens", 0) or 0)
        self.output += int(usage.get("output_tokens", 0) or 0)
        self.total = self.input + self.output


@contextmanager
def _track_tokens():
    """Patch agent_router.invoke_agent_raw to accumulate token usage.

    Yields a :class:`TokenCounter` updated in-place as Anthropic calls happen.
    Restores the original function on exit.
    """
    from orchestrator import agent_router

    counter = TokenCounter()
    original = agent_router.invoke_agent_raw

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        counter.add(result.get("usage"))
        return result

    agent_router.invoke_agent_raw = wrapped
    # invoke_agent calls invoke_agent_raw by attribute lookup on the module,
    # so the wrap propagates. But agent_router.invoke_agent has its own
    # closure-free direct call — re-bind it too so token counting works.
    try:
        yield counter
    finally:
        agent_router.invoke_agent_raw = original


# ---------------------------------------------------------------------------
# Per-entry pipeline driver
# ---------------------------------------------------------------------------

@dataclass
class CaptureResult:
    architect_draft: str = ""
    critic_score: float = 0.0
    critic_feedback: str = ""
    final_output: str = ""
    elapsed_seconds: float = 0.0
    token_count: int = 0
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    extra: dict = field(default_factory=dict)


def run_entry(entry: dict) -> CaptureResult:
    """Drive the orchestrator pipeline for a single eval entry.

    Returns a :class:`CaptureResult`. Failures are captured on the result
    rather than raised, so the caller can continue to the next entry.
    """
    pre_baked = (entry.get("pre_baked_context") or "").strip()
    if not pre_baked:
        return CaptureResult(
            skipped=True,
            skip_reason="no pre_baked_context — Clarifier interaction required",
        )

    target_model = entry.get("target_model_family", "claude").strip().lower()
    user_prompt = (entry.get("user_prompt") or "").strip()

    if not user_prompt:
        return CaptureResult(
            skipped=True,
            skip_reason="empty user_prompt",
        )

    started = perf_counter()
    try:
        with _track_tokens() as tokens:
            captured = _drive_pipeline(user_prompt, pre_baked, target_model)
        elapsed = perf_counter() - started
        captured.elapsed_seconds = round(elapsed, 3)
        captured.token_count = tokens.total
        captured.extra = {
            "input_tokens": tokens.input,
            "output_tokens": tokens.output,
        }
        return captured
    except Exception as exc:  # noqa: BLE001
        elapsed = perf_counter() - started
        return CaptureResult(
            elapsed_seconds=round(elapsed, 3),
            error=f"{type(exc).__name__}: {exc}",
            skipped=True,
            skip_reason=f"pipeline error: {type(exc).__name__}",
        )


def _drive_pipeline(
    user_prompt: str,
    pre_baked_context: str,
    target_model: str,
) -> CaptureResult:
    """Run draft → refine → critic against the live agent stack.

    Intentionally bypasses the public ``handle_message`` Lambda entry so we
    don't need DynamoDB. The pipeline calls match the orchestrator's
    one-shot path (Phase 3 of WORKFLOW_REDESIGN.md). Chain-path execution
    is out of scope for this phase — the Decomposer arrives in Phase D.
    """
    from agents import (
        clarifier,  # noqa: F401  (kept for parity with orchestrator imports)
        context_scout,  # noqa: F401
        fewshot_generator,
        format_profiles,
        guardrails,
        prompt_architect,
    )
    from orchestrator.agent_router import (
        classify_task,
        fetch_fewshot_examples,
        fetch_reference_prompts,
        invoke_agent,
        invoke_parallel,
        run_critic_review,
    )
    from orchestrator.orchestrator import _assemble_prompt_for_review
    from orchestrator.response_merger import _extract_prompt_block
    from orchestrator.session import create_session

    if target_model not in format_profiles.FORMAT_PROFILES:
        raise ValueError(f"unknown target_model_family: {target_model}")

    profile = format_profiles.FORMAT_PROFILES[target_model]
    session = create_session("eval-runner", target_model, user_prompt)
    session.task_category = classify_task(user_prompt)

    # 1. Architect draft
    reference_prompts = fetch_reference_prompts(
        session.task_category, target_model, k=3,
    )
    arch_msgs, arch_sys = prompt_architect.build_messages(
        user_intent=user_prompt,
        mode="draft",
        context={
            "target_model": target_model,
            "task_category": session.task_category,
        },
        format_profile=profile,
        reference_prompts=reference_prompts,
    )
    draft_response = invoke_agent("prompt_architect", arch_msgs, arch_sys)
    session.current_draft = _extract_prompt_block(draft_response)
    architect_draft = session.current_draft

    # 2. Refine with pre-baked context (substitute for Clarifier reply)
    session.clarification_answers = {"reply_1": pre_baked_context}
    reference_prompts = fetch_reference_prompts(
        session.task_category, target_model, k=3,
    )
    arch_msgs, arch_sys = prompt_architect.build_messages(
        user_intent=pre_baked_context,
        mode="refine",
        context={
            "target_model": target_model,
            "task_category": session.task_category,
            "current_draft": session.current_draft,
            "user_context": session.user_context,
            "clarification_answers": session.clarification_answers,
        },
        format_profile=profile,
        reference_prompts=reference_prompts,
    )
    reference_examples = fetch_fewshot_examples(
        session.task_category, target_model, k=2,
    )
    fs_msgs, fs_sys = fewshot_generator.build_messages(
        refined_prompt=session.current_draft,
        task_category=session.task_category,
        format_profile=profile,
        reference_examples=reference_examples,
    )
    refined_response, fewshot_response = invoke_parallel([
        {"agent_name": "prompt_architect", "messages": arch_msgs, "system": arch_sys},
        {"agent_name": "fewshot_generator", "messages": fs_msgs, "system": fs_sys},
    ])
    session.current_draft = _extract_prompt_block(refined_response)
    session.fewshot_examples = (fewshot_response or "").strip()

    # Guardrails are part of the production flow; mirror it but don't abort
    # on a warning — eval should still capture what the pipeline produced.
    guardrails.check_prompt(session.current_draft, target_model)

    # 3. Critic
    final_prompt = _assemble_prompt_for_review(session)
    critic_result = run_critic_review(
        final_prompt,
        session.task_category,
        profile,
        target_model,
    )
    overall = critic_result["scores"].get("overall", -1)
    normalized = (overall / 10.0) if overall >= 0 else 0.0

    return CaptureResult(
        architect_draft=architect_draft,
        critic_score=normalized,
        critic_feedback=critic_result["response"],
        final_output=final_prompt,
    )


# ---------------------------------------------------------------------------
# Result aggregation + IO
# ---------------------------------------------------------------------------

def build_entry_result(entry: dict, capture: CaptureResult) -> dict:
    """Wrap a single capture into the on-disk schema."""
    captured = {
        "architect_draft": capture.architect_draft,
        "critic_score": capture.critic_score,
        "critic_feedback": capture.critic_feedback,
        "final_output": capture.final_output,
        "elapsed_seconds": capture.elapsed_seconds,
        "token_count": capture.token_count,
    }
    if capture.extra:
        captured["token_breakdown"] = capture.extra
    if capture.skipped:
        captured["skipped"] = True
        captured["skip_reason"] = capture.skip_reason
    if capture.error:
        captured["error"] = capture.error

    return {
        "eval_id": entry["id"],
        "input": entry,
        "captured": captured,
        "rating": {
            "score": None,
            "notes": None,
            "rated_at": None,
        },
    }


def write_run_file(
    results: list[dict],
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    label: str | None = None,
    timestamp: datetime | None = None,
    commit_sha: str | None = None,
    branch: str | None = None,
) -> Path:
    """Write results to eval/runs/<date>_<sha>_<branch-or-label>.json."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.now(timezone.utc)
    date_str = ts.strftime("%Y-%m-%d")
    sha = (commit_sha or _git_short_sha() or "nosha")[:10]
    suffix = label or branch or _git_branch() or "run"
    suffix = _slug(suffix)
    out_path = runs_dir / f"{date_str}_{sha}_{suffix}.json"

    payload = {
        "metadata": {
            "captured_at": ts.isoformat(),
            "commit_sha": commit_sha or _git_short_sha(),
            "branch": branch or _git_branch(),
            "entry_count": len(results),
            "captured_count": sum(
                1 for r in results
                if not r["captured"].get("skipped")
                and not r["captured"].get("error")
            ),
            "skipped_count": sum(
                1 for r in results if r["captured"].get("skipped")
            ),
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return out_path


def _slug(text: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "-" for c in text)
    return safe.strip("-") or "run"


def _git_short_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_branch() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_inputs(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON array of eval entries")
    return raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the POGO v2 eval harness")
    parser.add_argument(
        "--inputs", type=Path, default=DEFAULT_INPUTS,
        help="Path to eval inputs JSON (default: eval/inputs.json)",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=DEFAULT_RUNS_DIR,
        help="Directory to write the run file into (default: eval/runs/)",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Override the branch-derived suffix on the output filename "
             "(e.g. 'v2-baseline').",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N entries (smoke test).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Stub all Anthropic calls. Useful when ANTHROPIC_API_KEY is "
             "absent; produces a run file with placeholder captures so the "
             "shape can be inspected.",
    )
    args = parser.parse_args(argv)

    entries = load_inputs(args.inputs)
    if args.limit:
        entries = entries[: args.limit]

    print(f"[eval] loaded {len(entries)} entries from {args.inputs}")

    if args.dry_run:
        ctx = _dry_run_patch()
    else:
        ctx = _noop_context()

    results: list[dict] = []
    with ctx:
        for i, entry in enumerate(entries, start=1):
            eid = entry.get("id", f"index_{i}")
            print(f"[eval] ({i}/{len(entries)}) {eid} …", flush=True)
            capture = run_entry(entry)
            if capture.skipped:
                print(f"[eval]   skipped: {capture.skip_reason}")
            else:
                print(
                    f"[eval]   captured (critic={capture.critic_score:.2f}, "
                    f"tokens={capture.token_count}, "
                    f"elapsed={capture.elapsed_seconds:.1f}s)"
                )
            results.append(build_entry_result(entry, capture))

    out_path = write_run_file(
        results,
        runs_dir=args.runs_dir,
        label=args.label,
    )
    print(f"[eval] wrote {out_path}")
    return 0


@contextmanager
def _noop_context():
    yield


@contextmanager
def _dry_run_patch():
    """Patch Anthropic invocation with deterministic stubs.

    For local smoke-testing of the runner shape without an
    ``ANTHROPIC_API_KEY``. Real baseline runs should NOT use this mode.
    """
    from orchestrator import agent_router

    def fake_invoke_agent_raw(agent_name, messages, system, model_id=None, max_tokens=2000):
        body = (
            f"```\n[stub:{agent_name}] dry-run output\n```\n"
            "clarity: 7\nspecificity: 7\ncompleteness: 7\n"
            "constraint_coverage: 7\nhallucination_risk: 2\noverall: 7\n"
        )
        return {
            "agent_name": agent_name,
            "model_id": model_id or "stub",
            "text": body,
            "usage": {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300},
        }

    original_raw = agent_router.invoke_agent_raw
    agent_router.invoke_agent_raw = fake_invoke_agent_raw
    try:
        with patch("orchestrator.session._get_table", return_value=MagicMock()):
            yield
    finally:
        agent_router.invoke_agent_raw = original_raw


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

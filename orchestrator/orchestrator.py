"""POGO v2 conversation orchestrator.

Implements the state-machine that routes user messages through the
multi-agent pipeline described in PLAN.md.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import re

from agents import (
    clarifier,
    context_scout,
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
from orchestrator.live_test import run_live_test
from orchestrator.response_merger import (
    _extract_prompt_block,
    format_accepted,
    merge_draft_scout_clarifier,
    merge_refinement,
    merge_review,
    strip_architect_preamble,
    strip_fewshot_preamble,
)
from orchestrator.session import (
    Session,
    create_session,
    load_session,
    save_session,
)

INGEST_THRESHOLD = float(os.environ.get("INGEST_THRESHOLD", "0.8"))

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
}


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------

def handle_message(event: dict) -> dict:
    """Process a single user message and return the next response.

    Expects the request body to contain::

        {
            "session_id": "... (omit for first message)",
            "message": "user's text",
            "target_model": "claude|gpt|gemini",
            "run_live_test": true|false (optional)
        }

    Returns an API-Gateway-shaped dict (statusCode, headers, body).
    """
    try:
        body = json.loads(event.get("body", "{}"))
        message = body.get("message", "").strip()
        target_model = body.get("target_model", "claude").strip().lower()
        run_live_test_requested = _coerce_optional_bool(body.get("run_live_test"))
        session_id = body.get("session_id")
        user_id = body.get("user_id", "anonymous")

        if not message:
            return _error(400, "message is required")

        if target_model not in format_profiles.FORMAT_PROFILES:
            return _error(400, f"target_model must be one of: {list(format_profiles.FORMAT_PROFILES)}")

        # Load or create session
        session: Session | None = None
        if session_id:
            session = load_session(session_id)

        if session is None:
            session = create_session(user_id, target_model, message)
            session.state = "initial"

        session.add_message("user", message)

        # Dispatch on state
        state = session.state
        if state == "initial":
            result = _handle_initial(session, message)
        elif state == "awaiting_context":
            result = _handle_awaiting_context(session, message)
        elif state == "review":
            result = _handle_review(session, message, run_live_test_requested)
        elif state == "iterating":
            result = _handle_iterating(session, message, run_live_test_requested)
        elif state == "accepted":
            result = _handle_accepted(session)
        else:
            return _error(400, f"Unknown session state: {state}")

        session.add_message("assistant", result["message"])
        save_session(session)

        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps(result),
        }

    except Exception as e:
        print(f"Orchestrator error: {e}")
        return _error(500, str(e))


# ---------------------------------------------------------------------------
# Architect output validation (Bug #12)
# ---------------------------------------------------------------------------

REQUIRED_ARCHITECT_SECTIONS = ("role", "context", "task", "constraints")


class ArchitectOutputError(RuntimeError):
    """Raised when the Architect produces an unusable prompt after a retry.

    "Unusable" means one or more of <role>/<context>/<task>/<constraints>
    is missing from the extracted prompt body. Bug #12 (PLAN_REDESIGN.md)
    used to manifest as the orchestrator silently dropping the Architect
    output; this exception makes the failure observable so the eval
    runner / Lambda handler can surface it instead.
    """


def _missing_architect_sections(prompt_text: str, target_model: str) -> list[str]:
    """Return any of REQUIRED_ARCHITECT_SECTIONS not detected in the prompt body."""
    if not prompt_text:
        return list(REQUIRED_ARCHITECT_SECTIONS)

    profile = format_profiles.FORMAT_PROFILES.get(target_model, {})
    wrapper = profile.get("wrapper_format", "markdown")

    missing: list[str] = []
    for section in REQUIRED_ARCHITECT_SECTIONS:
        if wrapper == "xml":
            present = re.search(rf"<{section}\b", prompt_text, re.IGNORECASE)
        else:
            present = re.search(
                rf"(?im)^\s*#+\s*{section}\b", prompt_text
            ) or re.search(rf"<{section}\b", prompt_text, re.IGNORECASE)
        if not present:
            missing.append(section)
    return missing


def _invoke_architect_with_validation(
    arch_msgs: list[dict],
    arch_sys: str,
    target_model: str,
) -> tuple[str, str]:
    """Invoke the Prompt Architect and validate the output has every required section.

    Retries the Architect call once with a stricter instruction when the
    first response is missing any of <role>, <context>, <task>, or
    <constraints>. Raises :class:`ArchitectOutputError` if both attempts
    produce a structurally incomplete prompt.

    Returns:
        ``(raw_response, extracted_prompt_body)``. The raw response is
        what the merger needs (for "Techniques Used" extraction); the
        prompt body is what gets stored as ``session.current_draft``.
    """
    raw = invoke_agent("prompt_architect", arch_msgs, arch_sys)
    cleaned = strip_architect_preamble(raw)
    prompt_text = _extract_prompt_block(cleaned)
    missing = _missing_architect_sections(prompt_text, target_model)
    if not missing:
        return raw, prompt_text

    if os.environ.get("POGO_DEBUG"):
        print(f"[orchestrator] Architect raw output (first 600 chars): {raw[:600]!r}")
    print(
        f"[orchestrator] Architect output missing sections "
        f"{missing}; retrying once with stricter instruction."
    )

    addendum = (
        "Your previous output was missing required sections: "
        f"{', '.join(missing)}. Emit the FULL prompt again with EVERY "
        "required section present and labelled correctly: "
        "<role>, <context>, <task>, <constraints> (XML wrappers for "
        "Claude, '## Role' / '## Context' / '## Task' / '## Constraints' "
        "markdown headers for GPT and Gemini). Do NOT include any preamble, "
        "chain-of-thought, or commentary before the prompt body."
    )

    retry_msgs = list(arch_msgs) + [
        {"role": "assistant", "content": [{"type": "text", "text": raw}]},
        {"role": "user", "content": [{"type": "text", "text": addendum}]},
    ]
    raw2 = invoke_agent("prompt_architect", retry_msgs, arch_sys)
    cleaned2 = strip_architect_preamble(raw2)
    prompt_text2 = _extract_prompt_block(cleaned2)
    missing2 = _missing_architect_sections(prompt_text2, target_model)
    if missing2:
        if os.environ.get("POGO_DEBUG"):
            print(f"[orchestrator] Architect retry raw output (first 600 chars): {raw2[:600]!r}")
        raise ArchitectOutputError(
            "Prompt Architect produced an incomplete prompt after one "
            f"retry. Missing sections (target={target_model}): {missing2}. "
            "This used to be silently dropped (see Bug #12 in PLAN_REDESIGN.md)."
        )
    return raw2, prompt_text2


# ---------------------------------------------------------------------------
# State handlers
# ---------------------------------------------------------------------------

def _handle_initial(session: Session, message: str) -> dict:
    """STATE: initial — first message. Draft + scout + clarifier."""
    profile = format_profiles.FORMAT_PROFILES[session.target_model]
    prompt_format = _prompt_format(profile)

    # 1. Classify task
    session.task_category = classify_task(message)
    session.subcategory = session.task_category
    session.user_intent = message

    # 2. Prompt Architect — draft (with reference prompts from the DB)
    reference_prompts = fetch_reference_prompts(
        session.task_category, session.target_model, k=3
    )
    arch_msgs, arch_sys = prompt_architect.build_messages(
        user_intent=message,
        mode="draft",
        context={
            "target_model": session.target_model,
            "task_category": session.task_category,
        },
        format_profile=profile,
        reference_prompts=reference_prompts,
    )
    draft_response, prompt_body = _invoke_architect_with_validation(
        arch_msgs, arch_sys, session.target_model
    )

    # 3. Extract draft for session
    session.current_draft = prompt_body

    # 4. Scout + Clarifier in parallel
    scout_msgs, scout_sys = context_scout.build_messages(
        draft_prompt=session.current_draft,
        task_category=session.task_category,
        format_profile=profile,
    )
    clar_msgs, clar_sys = clarifier.build_messages(
        draft_prompt=session.current_draft,
        user_intent=message,
        format_profile=profile,
    )

    scout_response, clarifier_response = invoke_parallel([
        {"agent_name": "context_scout", "messages": scout_msgs, "system": scout_sys},
        {"agent_name": "clarifier", "messages": clar_msgs, "system": clar_sys},
    ])

    # 5. Merge
    merged = merge_draft_scout_clarifier(
        draft_response,
        scout_response,
        clarifier_response,
        prompt_format=prompt_format,
    )

    # 6. Transition
    session.state = "awaiting_context"

    return _response(
        session,
        merged["message"],
        render_blocks=merged["render_blocks"],
    )


def _handle_awaiting_context(session: Session, message: str) -> dict:
    """STATE: awaiting_context — user supplied answers / context."""
    profile = format_profiles.FORMAT_PROFILES[session.target_model]
    prompt_format = _prompt_format(profile)

    # Store the user's reply as accumulated context
    session.clarification_answers[f"reply_{len(session.clarification_answers) + 1}"] = message

    # 1. Prompt Architect — refine (with reference prompts from the DB)
    reference_prompts = fetch_reference_prompts(
        session.task_category, session.target_model, k=3
    )
    arch_msgs, arch_sys = prompt_architect.build_messages(
        user_intent=message,
        mode="refine",
        context={
            "target_model": session.target_model,
            "task_category": session.task_category,
            "current_draft": session.current_draft,
            "user_context": session.user_context,
            "clarification_answers": session.clarification_answers,
        },
        format_profile=profile,
        reference_prompts=reference_prompts,
    )

    # 2. Architect refine FIRST (Bug #11: Few-Shot must consume the REFINED
    #    draft, not the stale pre-refine one). Validate sections; retry once
    #    if anything's missing (Bug #12).
    refined_response, refined_body = _invoke_architect_with_validation(
        arch_msgs, arch_sys, session.target_model
    )
    session.current_draft = refined_body

    # 3. Few-Shot Generator runs AFTER the Architect, with the refined draft.
    if _fewshot_enabled():
        reference_examples = fetch_fewshot_examples(
            session.task_category, session.target_model, k=2
        )
        fs_msgs, fs_sys = fewshot_generator.build_messages(
            refined_prompt=session.current_draft,
            task_category=session.task_category,
            format_profile=profile,
            reference_examples=reference_examples,
        )
        fewshot_response = invoke_agent("fewshot_generator", fs_msgs, fs_sys)
        fewshot_response = strip_fewshot_preamble(fewshot_response)
        session.fewshot_examples = fewshot_response.strip()
    else:
        fewshot_response = ""
        session.fewshot_examples = ""

    # 4. Guardrails
    gr = guardrails.check_prompt(session.current_draft, session.target_model)

    # 5. Merge
    merged = merge_refinement(
        refined_response,
        fewshot_response,
        gr,
        prompt_format=prompt_format,
    )

    # 6. Transition
    if gr["passed"]:
        session.state = "review"
    # else stay in awaiting_context so user can fix

    return _response(
        session,
        merged["message"],
        render_blocks=merged["render_blocks"],
    )


def _handle_review(
    session: Session,
    message: str,
    run_live_test_requested: bool | None = None,
) -> dict:
    """STATE: review — run critic (and optional live test)."""
    # Check if user wants to accept directly
    if message.strip().lower() in ("accept", "accepted", "looks good", "done", "yes"):
        return _handle_accepted(session)

    should_run_live_test = _live_testing_enabled() or bool(run_live_test_requested)
    return _evaluate_review(session, should_run_live_test=should_run_live_test)


def _handle_iterating(
    session: Session,
    message: str,
    run_live_test_requested: bool | None = None,
) -> dict:
    """STATE: iterating — user requested changes."""
    profile = format_profiles.FORMAT_PROFILES[session.target_model]
    prompt_format = _prompt_format(profile)

    # Check for accept
    if message.strip().lower() in ("accept", "accepted", "looks good", "done", "yes"):
        return _handle_accepted(session)

    if run_live_test_requested:
        return _evaluate_review(session, should_run_live_test=True)

    # 1. Prompt Architect — refine with edit instructions
    arch_msgs, arch_sys = prompt_architect.build_messages(
        user_intent=message,
        mode="refine",
        context={
            "target_model": session.target_model,
            "task_category": session.task_category,
            "current_draft": session.current_draft,
            "user_context": session.user_context,
            "clarification_answers": session.clarification_answers,
        },
        format_profile=profile,
    )
    refined_response, refined_body = _invoke_architect_with_validation(
        arch_msgs, arch_sys, session.target_model
    )

    # 2. Update draft
    session.current_draft = refined_body
    session.fewshot_examples = ""

    # 3. Guardrails
    gr = guardrails.check_prompt(session.current_draft, session.target_model)

    # 4. Build response
    merged = merge_refinement(
        refined_response,
        "",
        gr,
        prompt_format=prompt_format,
    )

    # 5. Transition back to review
    session.state = "review"

    return _response(
        session,
        merged["message"],
        render_blocks=merged["render_blocks"],
    )


def _handle_accepted(session: Session) -> dict:
    """STATE: accepted — finalise and (optionally) ingest into the prompt DB."""
    already_accepted = session.state == "accepted"
    session.state = "accepted"
    profile = format_profiles.FORMAT_PROFILES.get(session.target_model, {})

    overall = session.scores.get("overall", -1)
    quality = overall / 10.0 if overall >= 0 else 0.0
    ingested = bool(session.ingested)

    if not already_accepted and quality >= INGEST_THRESHOLD and session.current_draft:
        ingested = _ingest_accepted_prompt(session, quality)
    session.ingested = ingested

    merged = format_accepted(
        session.current_draft,
        ingested,
        prompt_format=_prompt_format(profile),
        threshold=INGEST_THRESHOLD,
    )
    return _response(
        session,
        merged["message"],
        render_blocks=merged["render_blocks"],
    )


def _ingest_accepted_prompt(session: Session, quality_score: float) -> bool:
    """Persist an accepted, high-quality prompt into the prompt DB.

    Returns ``True`` on successful ingestion, ``False`` on any failure
    (the session is still marked accepted — ingestion is best-effort).
    """
    try:
        from prompt_db.ingest import ingest_single_prompt
        record = _build_prompt_record_from_session(session, quality_score)
        return bool(ingest_single_prompt(record))
    except Exception as e:
        print(f"Prompt ingestion failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evaluate_review(session: Session, *, should_run_live_test: bool) -> dict:
    """Run the critic and optional live test, then build the review payload."""
    from agents.guardrails import check_structural_integrity
    profile = format_profiles.FORMAT_PROFILES[session.target_model]
    final_prompt = _assemble_prompt_for_review(session)

    struct_check = check_structural_integrity(final_prompt, session.target_model)
    if not struct_check["passed"]:
        defect_ids = [f["check"] for f in struct_check["findings"]]
        return _error(422, f"Structural defects in assembled prompt: {defect_ids}")

    with ThreadPoolExecutor(max_workers=2 if should_run_live_test else 1) as pool:
        critic_future = pool.submit(
            run_critic_review,
            final_prompt,
            session.task_category,
            profile,
            session.target_model,
        )
        live_test_future = (
            pool.submit(run_live_test, final_prompt, session.target_model)
            if should_run_live_test
            else None
        )

        critic_result = critic_future.result()
        live_test_result = live_test_future.result() if live_test_future else None

    critic_response = critic_result["response"]
    scores = critic_result["scores"]
    suggestions = critic_result["suggestions"]
    sample_input = live_test_result.get("sample_input") if live_test_result else None

    session.scores = scores
    session.state = "iterating"

    merged = merge_review(
        critic_response,
        scores,
        suggestions=suggestions,
        sample_input=sample_input,
        sample_output=live_test_result,
    )

    return _response(
        session,
        merged["message"],
        scores=scores,
        suggestions=suggestions,
        sample_input=sample_input,
        sample_output=live_test_result,
        live_testing_enabled=_live_testing_enabled(),
        render_blocks=merged["render_blocks"],
    )


def _assemble_prompt_for_review(session: Session) -> str:
    """Build the prompt to evaluate, including any few-shot examples."""
    prompt_text = session.current_draft.strip()
    fewshot_examples = (session.fewshot_examples or "").strip()
    if not fewshot_examples:
        return prompt_text

    examples_section = format_profiles.format_section(
        fewshot_examples,
        "examples",
        session.target_model,
    ).strip()
    if examples_section in prompt_text:
        return prompt_text
    return f"{prompt_text}\n\n{examples_section}".strip()


def _build_prompt_record_from_session(session: Session, quality_score: float):
    """Construct a PromptRecord from accepted session state."""
    from prompt_db.schema import PromptRecord

    profile = format_profiles.FORMAT_PROFILES.get(session.target_model, {})
    fmt = "xml" if profile.get("wrapper_format") == "xml" else "markdown"
    system_prompt, user_prompt_template = _split_final_draft(session.current_draft)

    techniques = (
        session.scores.get("techniques_identified")
        if isinstance(session.scores, dict)
        else None
    ) or []

    return PromptRecord(
        id=f"user_{session.session_id[:12]}",
        task_category=session.task_category or "general",
        subcategory=session.subcategory or session.task_category or "general",
        target_model=session.target_model,
        format=fmt,
        techniques=[str(t).strip() for t in techniques if str(t).strip()],
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        few_shot_examples=_parse_fewshot_examples(session.fewshot_examples),
        quality_score=max(0.0, min(1.0, quality_score)),
        source="user_generated",
    )


def _split_final_draft(prompt_text: str) -> tuple[str, str]:
    """Split a final draft into system and user-template portions.

    The current Prompt Architect does not emit a dedicated system/user split
    for every format, so this parser only separates explicit markers. When no
    such boundary is present, the full draft is kept as ``system_prompt`` and
    the user template is left empty.
    """
    text = (prompt_text or "").strip()
    if not text:
        return "", ""

    system_patterns = (
        r"(?is)^\s*system\s*:\s*(.*?)\n\s*user\s*:\s*(.*)$",
        r"(?is)^\s*##\s*system\b(.*?)\n\s*##\s*user(?:\s+prompt|\s+template)?\b(.*)$",
        r"(?is)^\s*<system>(.*?)</system>\s*<user(?:_prompt|_template)?>(.*?)</user(?:_prompt|_template)?>\s*$",
    )
    for pattern in system_patterns:
        match = re.match(pattern, text)
        if match:
            return match.group(1).strip(), match.group(2).strip()

    return text, ""


def _parse_fewshot_examples(text: str) -> list[dict]:
    """Parse the Few-Shot Generator's text output into structured examples."""
    raw = (text or "").strip()
    if not raw:
        return []

    blocks = re.split(r"(?im)^(?=Example\s+\d+)", raw)
    examples: list[dict] = []

    for block in blocks:
        chunk = block.strip()
        if not chunk:
            continue

        label_match = re.match(r"(?im)^Example\s+\d+\s+[—-]\s*(.+)$", chunk)
        label = label_match.group(1).strip() if label_match else ""

        input_match = re.search(
            r"(?is)\bInput\s*:\s*(.*?)\n\s*Output\s*:\s*(.*)$",
            chunk,
        )
        if input_match:
            example = {
                "input": input_match.group(1).strip(),
                "output": input_match.group(2).strip(),
            }
            if label:
                example["label"] = label
            examples.append(example)
            continue

        examples.append({"text": _clean_fewshot_block(chunk)})

    return examples


def _clean_fewshot_block(text: str) -> str:
    cleaned = re.sub(r"(?im)^Example\s+\d+\s+[—-]\s*", "", text).strip()
    return cleaned


def _response(
    session: Session,
    message: str,
    scores: dict | None = None,
    suggestions: list[str] | None = None,
    sample_input: str | None = None,
    sample_output: dict | None = None,
    live_testing_enabled: bool | None = None,
    render_blocks: list[dict] | None = None,
) -> dict:
    """Build the standard response payload."""
    return {
        "session_id": session.session_id,
        "state": session.state,
        "message": message,
        "prompt_draft": session.current_draft,
        "scores": scores or session.scores,
        "suggestions": suggestions or [],
        "sample_input": sample_input,
        "sample_output": sample_output,
        "ingested": session.ingested,
        "live_testing_enabled": _live_testing_enabled() if live_testing_enabled is None else live_testing_enabled,
        "render_blocks": render_blocks or [],
    }


def _error(code: int, msg: str) -> dict:
    return {
        "statusCode": code,
        "headers": CORS_HEADERS,
        "body": json.dumps({"error": msg}),
    }


def _prompt_format(profile: dict) -> str:
    return "xml" if profile.get("wrapper_format") == "xml" else "markdown"


def _live_testing_enabled() -> bool:
    return os.environ.get("POGO_ENABLE_LIVE_TEST", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _fewshot_enabled() -> bool:
    return os.environ.get("POGO_FEWSHOT_ENABLED", "false").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _coerce_optional_bool(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None

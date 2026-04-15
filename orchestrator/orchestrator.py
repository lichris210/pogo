"""POGO v2 conversation orchestrator.

Implements the state-machine that routes user messages through the
multi-agent pipeline described in PLAN.md.
"""

from __future__ import annotations

import json
import os

from agents import (
    clarifier,
    context_scout,
    critic,
    fewshot_generator,
    format_profiles,
    guardrails,
    prompt_architect,
)
from orchestrator.agent_router import (
    classify_task,
    invoke_agent,
    invoke_parallel,
)
from orchestrator.response_merger import (
    format_accepted,
    merge_draft_scout_clarifier,
    merge_refinement,
    merge_review,
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
            "target_model": "claude|gpt|gemini"
        }

    Returns an API-Gateway-shaped dict (statusCode, headers, body).
    """
    try:
        body = json.loads(event.get("body", "{}"))
        message = body.get("message", "").strip()
        target_model = body.get("target_model", "claude").strip().lower()
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
            result = _handle_review(session, message)
        elif state == "iterating":
            result = _handle_iterating(session, message)
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
# State handlers
# ---------------------------------------------------------------------------

def _handle_initial(session: Session, message: str) -> dict:
    """STATE: initial — first message. Draft + scout + clarifier."""
    profile = format_profiles.FORMAT_PROFILES[session.target_model]

    # 1. Classify task
    session.task_category = classify_task(message)
    session.user_intent = message

    # 2. Prompt Architect — draft
    arch_msgs, arch_sys = prompt_architect.build_messages(
        user_intent=message,
        mode="draft",
        context={
            "target_model": session.target_model,
            "task_category": session.task_category,
        },
        format_profile=profile,
    )
    draft_response = invoke_agent("prompt_architect", arch_msgs, arch_sys)

    # 3. Extract draft for session
    from orchestrator.response_merger import _extract_prompt_block
    session.current_draft = _extract_prompt_block(draft_response)

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
    merged = merge_draft_scout_clarifier(draft_response, scout_response, clarifier_response)

    # 6. Transition
    session.state = "awaiting_context"

    return _response(session, merged)


def _handle_awaiting_context(session: Session, message: str) -> dict:
    """STATE: awaiting_context — user supplied answers / context."""
    profile = format_profiles.FORMAT_PROFILES[session.target_model]

    # Store the user's reply as accumulated context
    session.clarification_answers[f"reply_{len(session.clarification_answers) + 1}"] = message

    # 1. Prompt Architect — refine
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

    # 2. Few-Shot Generator in parallel
    fs_msgs, fs_sys = fewshot_generator.build_messages(
        refined_prompt=session.current_draft,
        task_category=session.task_category,
        format_profile=profile,
    )

    refined_response, fewshot_response = invoke_parallel([
        {"agent_name": "prompt_architect", "messages": arch_msgs, "system": arch_sys},
        {"agent_name": "fewshot_generator", "messages": fs_msgs, "system": fs_sys},
    ])

    # 3. Update draft
    from orchestrator.response_merger import _extract_prompt_block
    session.current_draft = _extract_prompt_block(refined_response)

    # 4. Guardrails
    gr = guardrails.check_prompt(session.current_draft, session.target_model)

    # 5. Merge
    merged = merge_refinement(refined_response, fewshot_response, gr)

    # 6. Transition
    if gr["passed"]:
        session.state = "review"
    # else stay in awaiting_context so user can fix

    return _response(session, merged)


def _handle_review(session: Session, message: str) -> dict:
    """STATE: review — run critic (and optional live test)."""
    profile = format_profiles.FORMAT_PROFILES[session.target_model]

    # Check if user wants to accept directly
    if message.strip().lower() in ("accept", "accepted", "looks good", "done", "yes"):
        return _handle_accepted(session)

    # 1. Critic
    crit_msgs, crit_sys = critic.build_messages(
        final_prompt=session.current_draft,
        task_category=session.task_category,
        format_profile=profile,
    )
    critic_response = invoke_agent("critic", crit_msgs, crit_sys)
    scores = critic.parse_scores(critic_response)
    session.scores = scores

    # 2. Optional live test (simple: send the draft to the target model)
    sample_output = _run_live_test(session.current_draft, session.target_model)

    # 3. Merge
    merged = merge_review(critic_response, scores, sample_output)

    # 4. Transition — user decides next
    session.state = "iterating"

    return _response(session, merged, scores=scores, sample_output=sample_output)


def _handle_iterating(session: Session, message: str) -> dict:
    """STATE: iterating — user requested changes."""
    profile = format_profiles.FORMAT_PROFILES[session.target_model]

    # Check for accept
    if message.strip().lower() in ("accept", "accepted", "looks good", "done", "yes"):
        return _handle_accepted(session)

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
    refined_response = invoke_agent("prompt_architect", arch_msgs, arch_sys)

    # 2. Update draft
    from orchestrator.response_merger import _extract_prompt_block
    session.current_draft = _extract_prompt_block(refined_response)

    # 3. Guardrails
    gr = guardrails.check_prompt(session.current_draft, session.target_model)

    # 4. Build response
    merged = merge_refinement(refined_response, "", gr)

    # 5. Transition back to review
    session.state = "review"

    return _response(session, merged)


def _handle_accepted(session: Session) -> dict:
    """STATE: accepted — finalise."""
    session.state = "accepted"

    overall = session.scores.get("overall", -1)
    ingested = overall >= 0 and (overall / 10.0) >= INGEST_THRESHOLD

    merged = format_accepted(session.current_draft, ingested)
    return _response(session, merged)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_live_test(prompt: str, target_model: str) -> str | None:
    """Send the draft prompt to the target model with a brief test input.

    Returns the model's output, or ``None`` on failure.
    """
    try:
        test_input = (
            "This is a brief test. Respond with a short example output "
            "demonstrating how you would handle a typical request using the "
            "instructions above."
        )

        from orchestrator.agent_router import invoke_agent
        messages = [
            {"role": "user", "content": [{"type": "text", "text": test_input}]},
        ]
        return invoke_agent(
            agent_name="live_test",
            messages=messages,
            system=prompt,
            max_tokens=500,
        )
    except Exception as e:
        print(f"Live test failed: {e}")
        return None


def _response(
    session: Session,
    message: str,
    scores: dict | None = None,
    sample_output: str | None = None,
) -> dict:
    """Build the standard response payload."""
    return {
        "session_id": session.session_id,
        "state": session.state,
        "message": message,
        "prompt_draft": session.current_draft,
        "scores": scores or session.scores,
        "sample_output": sample_output,
    }


def _error(code: int, msg: str) -> dict:
    return {
        "statusCode": code,
        "headers": CORS_HEADERS,
        "body": json.dumps({"error": msg}),
    }

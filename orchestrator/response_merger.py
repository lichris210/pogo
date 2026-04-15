"""Response merger — combines outputs from parallel agents into cohesive chat messages."""

from __future__ import annotations


def merge_draft_scout_clarifier(
    draft_response: str,
    scout_response: str,
    clarifier_response: str,
) -> str:
    """Merge the initial draft, context suggestions, and clarifying questions.

    The result reads like a single assistant message, not three separate bots.

    Args:
        draft_response: Raw text from the Prompt Architect (draft mode).
        scout_response: Raw text from the Context Scout.
        clarifier_response: Raw text from the Clarifier.

    Returns:
        A merged chat message string.
    """
    parts = [
        "Here's an initial prompt draft based on your task:\n",
        "```\n" + _extract_prompt_block(draft_response) + "\n```\n",
    ]

    techniques = _extract_after(draft_response, "Techniques Used")
    if techniques:
        parts.append(f"**Techniques applied:** {techniques.strip()}\n")

    parts.append(
        "\n---\n\n"
        "To make this even stronger, here's what you could provide:\n\n"
        f"{scout_response.strip()}\n"
    )

    parts.append(
        "\n---\n\n"
        "And a few questions to sharpen things:\n\n"
        f"{clarifier_response.strip()}"
    )

    return "\n".join(parts)


def merge_refinement(
    refined_prompt: str,
    fewshot_response: str,
    guardrail_result: dict,
) -> str:
    """Merge a refined prompt with few-shot examples and guardrail results.

    Args:
        refined_prompt: Raw text from the Prompt Architect (refine mode).
        fewshot_response: Raw text from the Few-Shot Generator.
        guardrail_result: Dict from ``guardrails.check_prompt()``.

    Returns:
        A merged chat message string.
    """
    parts = [
        "Here's your refined prompt:\n",
        "```\n" + _extract_prompt_block(refined_prompt) + "\n```\n",
    ]

    techniques = _extract_after(refined_prompt, "Techniques Used")
    if techniques:
        parts.append(f"**Techniques applied:** {techniques.strip()}\n")

    if fewshot_response.strip():
        parts.append(
            "\n---\n\n"
            "**Few-shot examples** to include with the prompt:\n\n"
            f"{fewshot_response.strip()}\n"
        )

    # Guardrail warnings / errors
    warnings = guardrail_result.get("warnings", [])
    errors = guardrail_result.get("errors", [])

    if errors:
        error_lines = "\n".join(f"  - {e}" for e in errors)
        parts.append(
            "\n---\n\n"
            "**Guardrail errors** (should be fixed before proceeding):\n"
            f"{error_lines}\n"
        )

    if warnings:
        warn_lines = "\n".join(f"  - {w}" for w in warnings)
        parts.append(
            "\n---\n\n"
            "**Guardrail warnings** (consider addressing):\n"
            f"{warn_lines}\n"
        )

    if guardrail_result.get("passed", True) and not errors:
        parts.append(
            "\n---\n\n"
            "Guardrails passed. Ready for evaluation — "
            "reply to run the Critic, or make further edits."
        )

    return "\n".join(parts)


def merge_review(
    critic_response: str,
    scores: dict,
    sample_output: str | None = None,
) -> str:
    """Merge critic evaluation, scores, and optional live-test output.

    Args:
        critic_response: Raw text from the Critic agent.
        scores: Parsed scores dict from ``critic.parse_scores()``.
        sample_output: Optional live-test result.

    Returns:
        A merged chat message string.
    """
    parts = ["**Prompt Evaluation**\n"]

    # Score card
    score_lines = []
    for key in ("clarity", "specificity", "completeness",
                "constraint_coverage", "hallucination_risk", "overall"):
        val = scores.get(key, -1)
        label = key.replace("_", " ").title()
        note = " (lower is better)" if key == "hallucination_risk" else ""
        bar = _score_bar(val)
        score_lines.append(f"  {label}{note}: {bar} {val}/10")
    parts.append("\n".join(score_lines) + "\n")

    # Critic prose (suggestions etc.)
    parts.append(f"\n{critic_response.strip()}\n")

    if sample_output:
        parts.append(
            "\n---\n\n"
            "**Sample output** (live test against target model):\n\n"
            f"```\n{sample_output.strip()}\n```"
        )

    parts.append(
        "\n---\n\n"
        "Reply with edits to refine further, or say **accept** to finalise."
    )

    return "\n".join(parts)


def format_accepted(prompt_draft: str, ingested: bool) -> str:
    """Format the final accepted prompt message.

    Args:
        prompt_draft: The final prompt text.
        ingested: Whether the prompt was added to the prompt database.

    Returns:
        A clean final message string.
    """
    parts = [
        "Your prompt is finalised.\n",
        "```\n" + prompt_draft.strip() + "\n```\n",
    ]
    if ingested:
        parts.append(
            "This prompt scored high enough to be added to our reference library."
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_prompt_block(text: str) -> str:
    """Pull the content out of a markdown code fence, or return raw text."""
    import re
    m = re.search(r"```(?:\w*)\n([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _extract_after(text: str, heading: str) -> str:
    """Return everything after *heading* (case-insensitive) in *text*."""
    import re
    m = re.search(rf"(?:^|\n)\**{re.escape(heading)}\**[:\s]*\n?([\s\S]*)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def _score_bar(value: int, width: int = 10) -> str:
    """Render a simple text-based progress bar."""
    if value < 0:
        return "[??????????]"
    filled = min(value, width)
    return "[" + "#" * filled + "." * (width - filled) + "]"

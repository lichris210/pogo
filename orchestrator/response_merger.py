"""Response merger — combines outputs from parallel agents into cohesive chat messages."""

from __future__ import annotations

import re
from typing import Any


def merge_draft_scout_clarifier(
    draft_response: str,
    scout_response: str,
    clarifier_response: str,
    *,
    prompt_format: str,
) -> dict[str, Any]:
    """Merge the initial draft, context suggestions, and clarifying questions."""
    prompt_text = _extract_prompt_block(draft_response)
    techniques = _extract_after(draft_response, "Techniques Used")

    parts = [
        "Here's an initial prompt draft based on your task:\n",
        "```\n" + prompt_text + "\n```\n",
    ]

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

    return _result(
        "\n".join(parts),
        [
            {"type": "text", "text": "Here's an initial prompt draft based on your task:"},
            {
                "type": "prompt_draft",
                "title": "Prompt Draft",
                "prompt": prompt_text,
                "format": prompt_format,
                "techniques_text": techniques.strip() or None,
            },
            {
                "type": "context_checklist",
                "title": "To make this even stronger, here's what you could provide:",
                "items": _extract_list_items(scout_response),
            },
            {
                "type": "clarifier_questions",
                "title": "And a few questions to sharpen things:",
                "items": _extract_list_items(clarifier_response),
            },
        ],
    )


def merge_refinement(
    refined_prompt: str,
    fewshot_response: str,
    guardrail_result: dict,
    *,
    prompt_format: str,
) -> dict[str, Any]:
    """Merge a refined prompt with few-shot examples and guardrail results."""
    prompt_text = _extract_prompt_block(refined_prompt)
    techniques = _extract_after(refined_prompt, "Techniques Used")

    parts = [
        "Here's your refined prompt:\n",
        "```\n" + prompt_text + "\n```\n",
    ]

    if techniques:
        parts.append(f"**Techniques applied:** {techniques.strip()}\n")

    if fewshot_response.strip():
        parts.append(
            "\n---\n\n"
            "**Few-shot examples** to include with the prompt:\n\n"
            f"{fewshot_response.strip()}\n"
        )

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

    blocks: list[dict[str, Any]] = []
    if errors:
        blocks.append(
            {
                "type": "guardrail_banner",
                "severity": "error",
                "items": list(errors),
            }
        )
    if warnings:
        blocks.append(
            {
                "type": "guardrail_banner",
                "severity": "warning",
                "items": list(warnings),
            }
        )
    blocks.append(
        {
            "type": "prompt_draft",
            "title": "Refined Prompt",
            "prompt": prompt_text,
            "format": prompt_format,
            "techniques_text": techniques.strip() or None,
        }
    )
    if fewshot_response.strip():
        blocks.append(
            {
                "type": "fewshot_examples",
                "title": "Few-shot examples",
                "text": fewshot_response.strip(),
            }
        )
    if guardrail_result.get("passed", True) and not errors:
        blocks.append(
            {
                "type": "text",
                "text": "Guardrails passed. Ready for evaluation — reply to run the Critic, or make further edits.",
            }
        )

    return _result("\n".join(parts), blocks)


def merge_review(
    critic_response: str,
    scores: dict,
    *,
    suggestions: list[str] | None = None,
    sample_input: str | None = None,
    sample_output: dict | None = None,
) -> dict[str, Any]:
    """Merge critic evaluation, scores, and optional live-test output."""
    parts = ["**Prompt Evaluation**\n"]

    score_lines = []
    for key in (
        "clarity",
        "specificity",
        "completeness",
        "constraint_coverage",
        "hallucination_risk",
        "overall",
    ):
        val = scores.get(key, -1)
        label = key.replace("_", " ").title()
        note = " (lower is better)" if key == "hallucination_risk" else ""
        bar = _score_bar(val)
        score_lines.append(f"  {label}{note}: {bar} {val}/10")
    parts.append("\n".join(score_lines) + "\n")

    critic_text = _strip_json_block(critic_response)
    critic_summary = _strip_trailing_list(critic_text) if suggestions else critic_text
    if critic_summary:
        parts.append(f"\n{critic_summary.strip()}\n")

    if suggestions:
        suggestion_lines = "\n".join(
            f"{idx}. {item}" for idx, item in enumerate(suggestions, 1)
        )
        parts.append(
            "\n---\n\n"
            "**Specific improvements to consider:**\n\n"
            f"{suggestion_lines}\n"
        )

    if sample_input:
        parts.append(
            "\n---\n\n"
            "**Sample input** (used for the live test):\n\n"
            f"```\n{sample_input.strip()}\n```"
        )

    if sample_output:
        sample_text = _sample_output_text(sample_output)
        sample_meta = _sample_output_meta(sample_output)
        parts.append(
            "\n---\n\n"
            "**Sample output** (live test against target model):\n\n"
            f"```\n{sample_text}\n```"
        )
        if sample_meta:
            parts.append(f"\n{sample_meta}\n")

    parts.append(
        "\n---\n\n"
        "Reply with edits to refine further, or say **accept** to finalise."
    )

    blocks: list[dict[str, Any]] = [
        {
            "type": "scorecard",
            "title": "Prompt Evaluation",
            "scores": scores,
        }
    ]
    if critic_summary:
        blocks.append({"type": "text", "text": critic_summary})
    if suggestions:
        blocks.append(
            {
                "type": "suggestions_list",
                "title": "Specific improvements to consider",
                "items": suggestions,
            }
        )
    if sample_input:
        blocks.append(
            {
                "type": "sample_input",
                "title": "Sample input",
                "text": sample_input.strip(),
                "collapsed_by_default": True,
            }
        )
    if sample_output:
        blocks.append(
            {
                "type": "sample_output",
                "title": "Sample output",
                "text": sample_text,
                "latency_ms": sample_output.get("latency_ms"),
                "tokens_used": sample_output.get("tokens_used"),
                "collapsed_by_default": True,
            }
        )
    blocks.append(
        {
            "type": "text",
            "text": "Reply with edits to refine further, or say **accept** to finalise.",
        }
    )

    return _result("\n".join(parts), blocks)


def format_accepted(
    prompt_draft: str,
    ingested: bool,
    *,
    prompt_format: str,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Format the final accepted prompt message."""
    parts = [
        "Your prompt is finalised.\n",
        "```\n" + prompt_draft.strip() + "\n```\n",
    ]
    if ingested:
        parts.append(
            "This prompt scored high enough to be added to our reference library."
        )
    elif threshold is not None:
        parts.append(
            f"Your prompt has been saved. Prompts scoring above {threshold:.2f} "
            "are added to our reference library."
        )

    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": "Your prompt is finalised."},
        {
            "type": "final_prompt",
            "title": "Final Prompt",
            "prompt": prompt_draft.strip(),
            "format": prompt_format,
            "ingested": ingested,
        },
    ]
    if ingested:
        blocks.append(
            {
                "type": "text",
                "text": "This prompt scored high enough to be added to our reference library.",
            }
        )
    elif threshold is not None:
        blocks.append(
            {
                "type": "text",
                "text": (
                    f"Your prompt has been saved. Prompts scoring above {threshold:.2f} "
                    "are added to our reference library."
                ),
            }
        )
    return _result("\n".join(parts), blocks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(message: str, render_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "message": message,
        "render_blocks": render_blocks,
    }


def _extract_prompt_block(text: str) -> str:
    """Pull the content out of a markdown code fence, or return raw text."""
    m = re.search(r"```(?:\w*)\n([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _extract_after(text: str, heading: str) -> str:
    """Return everything after *heading* (case-insensitive) in *text*."""
    m = re.search(
        rf"(?:^|\n)\**{re.escape(heading)}\**[:\s]*\n?([\s\S]*)",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return ""


def _extract_list_items(text: str) -> list[str]:
    """Extract numbered-list items while preserving wrapped detail lines."""
    lines = text.strip().splitlines()
    if not lines:
        return []

    items: list[str] = []
    current: list[str] = []
    found_numbered = False

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue

        numbered = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if numbered:
            found_numbered = True
            if current:
                items.append("\n".join(current).strip())
            current = [numbered.group(2).strip()]
            continue

        if found_numbered and current:
            current.append(stripped)
        else:
            items.append(re.sub(r"^[-*]\s+", "", stripped))

    if current:
        items.append("\n".join(current).strip())

    return [item for item in items if item]


def _strip_json_block(text: str) -> str:
    """Remove the first JSON code block from the critic response."""
    stripped = re.sub(r"```(?:json)?\s*\{[\s\S]*?\}\s*```", "", text, count=1)
    return stripped.strip()


def _strip_trailing_list(text: str) -> str:
    """Remove a trailing bullet/numbered list from prose."""
    stripped_text = text.strip()
    stripped = re.sub(
        r"\n(?:[^\n]*?(?:suggestion|improvement)[^\n]*:)?\s*"
        r"(?:\n(?:[-*]|\d+\.)\s+.+)+\s*$",
        "",
        stripped_text,
        flags=re.IGNORECASE,
    )
    if stripped != stripped_text:
        return stripped.strip()
    stripped = re.sub(r"(?:\n(?:[-*]|\d+\.)\s+.+)+\s*$", "", stripped_text)
    return stripped.strip()


def _score_bar(value: int, width: int = 10) -> str:
    """Render a simple text-based progress bar."""
    if value < 0:
        return "[??????????]"
    filled = min(value, width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _sample_output_text(sample_output: dict) -> str:
    return str(sample_output.get("output", "")).strip()


def _sample_output_meta(sample_output: dict) -> str:
    parts = []
    latency_ms = sample_output.get("latency_ms")
    tokens_used = sample_output.get("tokens_used")
    if isinstance(latency_ms, int) and latency_ms >= 0:
        parts.append(f"Latency: {latency_ms} ms")
    if isinstance(tokens_used, int) and tokens_used >= 0:
        parts.append(f"Tokens used: {tokens_used}")
    return " | ".join(parts)

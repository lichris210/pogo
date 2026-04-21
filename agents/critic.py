"""Critic agent.

Evaluates prompt quality across multiple dimensions, returns numeric scores
with explanations, and suggests specific improvements.
"""

from __future__ import annotations

import json
import re

from agents.format_profiles import get_format_instructions

SYSTEM_PROMPT = """\
You are the Critic — you evaluate prompt quality with precision and \
specificity. Your scores must be evidence-based: cite the exact part of \
the prompt that justifies each rating.

=== SCORING DIMENSIONS ===

Rate each dimension from 0 to 10:

1. Clarity (0-10)
   Does every instruction have exactly one reasonable interpretation?
   10 = unambiguous; 0 = vague throughout.

2. Specificity (0-10)
   Are constraints, formats, and boundaries made explicit?
   10 = fully constrained; 0 = no constraints specified.

3. Completeness (0-10)
   Does the prompt cover role, context, task, constraints, and output format?
   10 = all sections present and thorough; 0 = critical sections missing.

4. Constraint Coverage (0-10)
   Are edge cases, fallback behavior, length limits, and forbidden actions \
addressed?
   10 = comprehensive; 0 = no constraints.

5. Hallucination Risk (0-10, LOWER is better)
   How likely is the model to fabricate unsupported claims?
   0 = strong grounding controls; 10 = no grounding, high risk.

6. Overall (0-10)
   Holistic quality considering all dimensions plus how well the prompt \
matches the target model's strengths.

=== REFERENCE PROMPTS ===
{reference_prompts}

=== OUTPUT FORMAT ===
Return your evaluation as a JSON block followed by prose.

```json
{{
  "clarity": <int>,
  "specificity": <int>,
  "completeness": <int>,
  "constraint_coverage": <int>,
  "hallucination_risk": <int>,
  "overall": <int>,
  "techniques_identified": ["<technique_1>", "<technique_2>", ...]
}}
```

Then for each dimension, write one sentence explaining the score.

Finally, list 2-3 specific, actionable improvement suggestions.

{format_instructions}
"""

SCORE_KEYS = [
    "clarity",
    "specificity",
    "completeness",
    "constraint_coverage",
    "hallucination_risk",
    "overall",
]


def build_messages(
    final_prompt: str,
    task_category: str,
    format_profile: dict,
    reference_prompts: list[str] | None = None,
) -> tuple[list[dict], str]:
    """Assemble messages for the Critic agent.

    Args:
        final_prompt: The prompt to evaluate.
        task_category: Classified task type (e.g. "code_generation").
        format_profile: The FORMAT_PROFILES entry for the target model.
        reference_prompts: Optional list of high-scoring reference prompt
            strings for comparison.

    Returns:
        A tuple of (messages list, system prompt string).
    """
    target_model = format_profile.get("name", "claude").lower()
    format_instructions = get_format_instructions(target_model)
    ref_block = _format_references(reference_prompts)

    system = SYSTEM_PROMPT.format(
        format_instructions=format_instructions,
        reference_prompts=ref_block,
    )

    user_text = (
        f"Task category: {task_category}\n\n"
        f"Prompt to evaluate:\n\n{final_prompt}"
    )

    messages = [
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]

    return messages, system


def parse_scores(response: str) -> dict:
    """Extract numeric scores from the Critic's response.

    Looks for a JSON code block first, then falls back to regex extraction.

    Args:
        response: The raw text response from the Critic agent.

    Returns:
        A dict with keys from :data:`SCORE_KEYS` mapped to ``int`` values,
        plus ``"techniques_identified"`` (list of str) if present.
        Missing keys default to ``-1``.
    """
    scores: dict = {}

    # Try JSON block first
    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            for key in SCORE_KEYS:
                if key in parsed:
                    scores[key] = int(parsed[key])
            if "techniques_identified" in parsed:
                scores["techniques_identified"] = parsed["techniques_identified"]
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: regex for "key: N" or "key = N" patterns
    if not scores:
        for key in SCORE_KEYS:
            pattern = rf"{key}[\"']?\s*[:=]\s*(\d+)"
            m = re.search(pattern, response, re.IGNORECASE)
            if m:
                scores[key] = int(m.group(1))

    if not scores.get("techniques_identified"):
        scores["techniques_identified"] = _extract_techniques(response)

    # Fill missing keys with -1
    for key in SCORE_KEYS:
        scores.setdefault(key, -1)
    scores.setdefault("techniques_identified", [])

    return scores


def parse_suggestions(response: str) -> list[str]:
    """Extract 2-3 actionable improvement suggestions from Critic prose."""
    text = _strip_json_block(response)
    if not text:
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    suggestions = _extract_list_after_heading(lines)
    if suggestions:
        return suggestions

    suggestions = _extract_bullets(lines)
    if suggestions:
        return suggestions[-3:]

    sentence_matches = re.findall(r"[^.!?]+[.!?]", text)
    actionable = []
    for sentence in sentence_matches:
        cleaned = sentence.strip()
        if re.search(r"\b(add|include|specify|clarify|define|tighten|state|reduce|limit)\b", cleaned, re.IGNORECASE):
            actionable.append(cleaned.rstrip(".!?"))
    return actionable[:3]


def _format_references(prompts: list[str] | None) -> str:
    if not prompts:
        return "(No reference prompts available for comparison.)"
    sections = []
    for i, p in enumerate(prompts, 1):
        sections.append(f"--- Reference {i} ---\n{p}")
    return "\n\n".join(sections)


def _strip_json_block(text: str) -> str:
    return re.sub(r"```(?:json)?\s*\{[\s\S]*?\}\s*```", "", text, count=1).strip()


def _extract_list_after_heading(lines: list[str]) -> list[str]:
    suggestions: list[str] = []
    capture = False

    for line in lines:
        lowered = line.lower().rstrip(":")
        if "suggestion" in lowered or "improvement" in lowered:
            capture = True
            remainder = re.sub(r"^.*?:\s*", "", line).strip()
            if remainder and remainder != line:
                suggestions.append(remainder)
            continue

        if not capture:
            continue

        bullet = re.match(r"^(?:[-*]|\d+\.)\s+(.*)$", line)
        if bullet:
            suggestions.append(bullet.group(1).strip())
            continue

        if suggestions:
            suggestions[-1] = f"{suggestions[-1]} {line}".strip()

    return suggestions[:3]


def _extract_bullets(lines: list[str]) -> list[str]:
    suggestions: list[str] = []
    for line in lines:
        bullet = re.match(r"^(?:[-*]|\d+\.)\s+(.*)$", line)
        if bullet:
            suggestions.append(bullet.group(1).strip())
    return suggestions


def _extract_techniques(response: str) -> list[str]:
    json_inline = re.search(
        r"techniques_identified[\"']?\s*[:=]\s*\[([^\]]*)\]",
        response,
        re.IGNORECASE,
    )
    if json_inline:
        raw_items = [
            part.strip().strip("\"'")
            for part in json_inline.group(1).split(",")
        ]
        return [item for item in raw_items if item]

    heading_match = re.search(
        r"(?im)^techniques(?:\s+identified)?\s*:\s*(.+)$",
        response,
    )
    if heading_match:
        raw = heading_match.group(1).strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        parts = [part.strip().strip("\"'") for part in raw.split(",")]
        return [part for part in parts if part]

    return []

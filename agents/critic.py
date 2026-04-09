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

    # Fill missing keys with -1
    for key in SCORE_KEYS:
        scores.setdefault(key, -1)
    scores.setdefault("techniques_identified", [])

    return scores


def _format_references(prompts: list[str] | None) -> str:
    if not prompts:
        return "(No reference prompts available for comparison.)"
    sections = []
    for i, p in enumerate(prompts, 1):
        sections.append(f"--- Reference {i} ---\n{p}")
    return "\n\n".join(sections)

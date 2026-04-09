"""Guardrails module.

Purely rule-based checks — no model calls. Runs fast and cheap.
Detects common anti-patterns that degrade prompt quality.
"""

from __future__ import annotations

import re

# Approximate context window sizes (in tokens) per model family.
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "claude": 200_000,
    "gpt": 128_000,
    "gemini": 1_000_000,
}

# Rough token estimate: 1 token ~= 0.75 words, or ~4 chars.
_CHARS_PER_TOKEN = 4

# --- Vague-instruction patterns ---------------------------------------------------

_VAGUE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"\bdo your best\b", re.IGNORECASE),
        "\"Do your best\" is vague — specify concrete success criteria instead.",
    ),
    (
        re.compile(r"\bbe creative\b(?!\s*(?:within|but|while|and\s+(?:keep|stay|limit|follow|use)))", re.IGNORECASE),
        "\"Be creative\" without constraints gives the model no guardrails — add scope, length, or style limits.",
    ),
    (
        re.compile(r"\bas (?:needed|appropriate|necessary)\b", re.IGNORECASE),
        "\"As needed\" delegates a decision to the model — make the condition explicit.",
    ),
    (
        re.compile(r"\bfeel free\b", re.IGNORECASE),
        "\"Feel free\" is permissive without direction — state what the model should do, not what it may do.",
    ),
    (
        re.compile(r"\btry to\b", re.IGNORECASE),
        "\"Try to\" implies optional effort — use direct imperatives instead.",
    ),
]

# --- Contradiction pair patterns ---------------------------------------------------

_CONTRADICTION_PAIRS: list[tuple[re.Pattern, re.Pattern, str]] = [
    (
        re.compile(r"\bbe concise\b", re.IGNORECASE),
        re.compile(r"\bprovide detailed\b", re.IGNORECASE),
        "Possible contradiction: \"be concise\" vs. \"provide detailed\" — clarify the expected depth.",
    ),
    (
        re.compile(r"\brespond in json\b", re.IGNORECASE),
        re.compile(r"\buse markdown\b", re.IGNORECASE),
        "Possible contradiction: requesting both JSON and Markdown output formats.",
    ),
    (
        re.compile(r"\bnever use bullet\b", re.IGNORECASE),
        re.compile(r"\blist the items\b", re.IGNORECASE),
        "Possible contradiction: forbids bullet points but asks for a list.",
    ),
    (
        re.compile(r"\bdo not include examples\b", re.IGNORECASE),
        re.compile(r"\bprovide examples\b", re.IGNORECASE),
        "Possible contradiction: simultaneously forbids and requests examples.",
    ),
    (
        re.compile(r"\bkeep.*?short\b", re.IGNORECASE),
        re.compile(r"\bcomprehensive\b", re.IGNORECASE),
        "Possible contradiction: \"keep short\" vs. \"comprehensive\" — specify a target length.",
    ),
]

# --- Output format indicators ------------------------------------------------------

_OUTPUT_FORMAT_INDICATORS = re.compile(
    r"(?:output format|respond (?:in|with|as)|return (?:a|the|as)|"
    r"format:|json|csv|markdown|xml|table|list|bullet|numbered)",
    re.IGNORECASE,
)

# --- Role/persona indicators -------------------------------------------------------

_ROLE_INDICATORS = re.compile(
    r"(?:you are|act as|role:|persona:|your role|as a\b|"
    r"you\'re a|imagine you are|pretend you are)",
    re.IGNORECASE,
)

# Minimum prompt length to be considered non-trivial (characters).
_MIN_PROMPT_LENGTH = 40


def check_prompt(prompt: str, target_model: str) -> dict:
    """Run all rule-based checks against a prompt.

    Args:
        prompt: The prompt text to validate.
        target_model: One of ``"claude"``, ``"gpt"``, or ``"gemini"``.

    Returns:
        A dict with:
        - ``passed`` (bool): ``True`` if there are no errors.
        - ``warnings`` (list[str]): Non-blocking issues.
        - ``errors`` (list[str]): Blocking issues.
    """
    warnings: list[str] = []
    errors: list[str] = []

    # --- Empty or too short -------------------------------------------------------
    stripped = prompt.strip()
    if not stripped:
        errors.append("Prompt is empty.")
        return {"passed": False, "warnings": warnings, "errors": errors}

    if len(stripped) < _MIN_PROMPT_LENGTH:
        errors.append(
            f"Prompt is very short ({len(stripped)} chars). "
            f"Prompts under {_MIN_PROMPT_LENGTH} characters rarely contain enough "
            "instruction for reliable results."
        )

    # --- Vague instructions -------------------------------------------------------
    for pattern, message in _VAGUE_PATTERNS:
        if pattern.search(prompt):
            warnings.append(message)

    # --- Missing output format specification --------------------------------------
    if not _OUTPUT_FORMAT_INDICATORS.search(prompt):
        warnings.append(
            "No output format specification detected. "
            "Consider specifying the expected response format (JSON, markdown, list, etc.)."
        )

    # --- Missing role/persona -----------------------------------------------------
    if not _ROLE_INDICATORS.search(prompt):
        warnings.append(
            "No role or persona definition detected. "
            "Assigning a role (e.g. \"You are a senior data analyst\") can "
            "improve tone, reasoning style, and domain focus."
        )

    # --- Contradictory instructions -----------------------------------------------
    for pat_a, pat_b, message in _CONTRADICTION_PAIRS:
        if pat_a.search(prompt) and pat_b.search(prompt):
            warnings.append(message)

    # --- Token length estimate vs context window ----------------------------------
    estimated_tokens = len(prompt) // _CHARS_PER_TOKEN
    context_limit = MODEL_CONTEXT_LIMITS.get(target_model, 128_000)

    # Warn if the prompt alone uses more than 50% of the context window.
    if estimated_tokens > context_limit * 0.5:
        warnings.append(
            f"Prompt is ~{estimated_tokens:,} tokens (estimated), which is over "
            f"50% of {target_model}'s {context_limit:,}-token context window. "
            "This leaves limited room for the model's response."
        )

    # Fail if the prompt itself exceeds the context window.
    if estimated_tokens > context_limit:
        errors.append(
            f"Prompt is ~{estimated_tokens:,} tokens (estimated), which exceeds "
            f"{target_model}'s {context_limit:,}-token context window."
        )

    return {
        "passed": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
    }

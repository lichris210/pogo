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

# Tokens reserved for the model's response when computing usable context.
_OUTPUT_BUFFER_TOKENS = 2_000

# Minimum prompt length to be considered non-trivial (characters).
_MIN_PROMPT_LENGTH = 40

# --- Severity levels ---------------------------------------------------------------
# "info"    — suggestions for improvement, not blocking
# "warning" — likely issues that should be addressed
# "error"   — critical problems that will cause poor results

# --- Vague-instruction patterns ---------------------------------------------------

_VAGUE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"\bdo your best\b", re.IGNORECASE),
        "\"Do your best\" is vague — specify concrete success criteria instead.",
    ),
    (
        re.compile(
            r"\bbe creative\b(?!\s*(?:within|but|while|and\s+(?:keep|stay|limit|follow|use)))",
            re.IGNORECASE,
        ),
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
    (
        re.compile(r"\bdo not use (?:any )?(?:formatting|markdown)\b", re.IGNORECASE),
        re.compile(r"\buse (?:headers?|bullets?|bold|italic)\b", re.IGNORECASE),
        "Possible contradiction: forbids formatting but requests formatted output.",
    ),
    (
        re.compile(r"\bno (?:code|programming)\b", re.IGNORECASE),
        re.compile(r"\bwrite (?:a |the )?(?:function|class|script|code)\b", re.IGNORECASE),
        "Possible contradiction: forbids code but requests a code artifact.",
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

# --- Ambiguous pronoun: sentence-initial pronoun + modal/copula without clear antecedent ---

_AMBIGUOUS_PRONOUN = re.compile(
    r"(?:^|\.\s+|\n\s*)(?:It|This|That|They)\s+"
    r"(?:should|must|will|can|may|is|are|was|were|needs?\s+to|has\s+to)\b",
    re.IGNORECASE | re.MULTILINE,
)

# --- Creative / generative keywords -----------------------------------------------

_CREATIVE_KEYWORDS = re.compile(
    r"\b(?:write|create|generate|draft|compose|design|invent|produce)\b",
    re.IGNORECASE,
)

_CONSTRAINT_KEYWORDS = re.compile(
    r"\b(?:words?|characters?|sentences?|paragraphs?|pages?|brief|concise|short|long|"
    r"detailed|format|structure|style|tone|audience|length|limit|maximum|minimum|"
    r"no more than|at least|exactly|restrict)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Estimate token count as word_count * 1.3."""
    return int(len(text.split()) * 1.3)


def _has_duplicate_instructions(text: str) -> bool:
    """Return True if the prompt contains near-duplicate instruction sentences.

    Splits on sentence boundaries, normalises each chunk, and flags any pair
    with Jaccard word-set similarity >= 0.75 (both sentences at least 6 words).
    """
    raw = re.split(r"[.!?\n]+", text)
    normed: list[frozenset[str]] = []
    for s in raw:
        clean = re.sub(r"[^\w\s]", "", s.strip().lower())
        words = clean.split()
        if len(words) >= 6:
            normed.append(frozenset(words))

    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            s1, s2 = normed[i], normed[j]
            union = len(s1 | s2)
            if union and len(s1 & s2) / union >= 0.75:
                return True
    return False


# ---------------------------------------------------------------------------
# Internal check runner
# ---------------------------------------------------------------------------

def _run_checks(prompt: str, target_model: str) -> list[dict]:
    """Run all checks and return a list of finding dicts.

    Each finding has keys:
        ``severity`` — ``"info"``, ``"warning"``, or ``"error"``
        ``check``    — short snake_case identifier
        ``message``  — human-readable description
    """
    findings: list[dict] = []

    def _add(severity: str, check: str, message: str) -> None:
        findings.append({"severity": severity, "check": check, "message": message})

    stripped = prompt.strip()

    # --- Empty --------------------------------------------------------------------
    if not stripped:
        _add("error", "empty_prompt", "Prompt is empty.")
        return findings

    # --- Too short ----------------------------------------------------------------
    if len(stripped) < _MIN_PROMPT_LENGTH:
        _add(
            "error",
            "too_short",
            f"Prompt is very short ({len(stripped)} chars). "
            f"Prompts under {_MIN_PROMPT_LENGTH} characters rarely contain enough "
            "instruction for reliable results.",
        )

    # --- Vague instructions -------------------------------------------------------
    for pattern, message in _VAGUE_PATTERNS:
        if pattern.search(prompt):
            _add("warning", "vague_instruction", message)

    # --- Missing output format specification --------------------------------------
    if not _OUTPUT_FORMAT_INDICATORS.search(prompt):
        _add(
            "info",
            "missing_output_format",
            "No output format specification detected. "
            "Consider specifying the expected response format (JSON, markdown, list, etc.).",
        )

    # --- Missing role/persona -----------------------------------------------------
    if not _ROLE_INDICATORS.search(prompt):
        _add(
            "info",
            "missing_role",
            "No role or persona definition detected. "
            "Assigning a role (e.g. \"You are a senior data analyst\") can "
            "improve tone, reasoning style, and domain focus.",
        )

    # --- Contradictory instructions -----------------------------------------------
    for pat_a, pat_b, message in _CONTRADICTION_PAIRS:
        if pat_a.search(prompt) and pat_b.search(prompt):
            _add("warning", "contradiction", message)

    # --- Token length estimate vs context window ----------------------------------
    estimated_tokens = _estimate_tokens(prompt)
    context_limit = MODEL_CONTEXT_LIMITS.get(target_model, 128_000)
    usable_limit = context_limit - _OUTPUT_BUFFER_TOKENS

    if estimated_tokens > context_limit:
        _add(
            "error",
            "exceeds_context_window",
            f"Prompt is ~{estimated_tokens:,} tokens (estimated), which exceeds "
            f"{target_model}'s {context_limit:,}-token context window.",
        )
    elif estimated_tokens > usable_limit * 0.5:
        _add(
            "warning",
            "large_prompt",
            f"Prompt is ~{estimated_tokens:,} tokens (estimated), which is over "
            f"50% of {target_model}'s usable context ({usable_limit:,} tokens after "
            f"reserving {_OUTPUT_BUFFER_TOKENS:,} for output). "
            "This leaves limited room for the model's response.",
        )

    # --- Ambiguous pronoun detection ----------------------------------------------
    if _AMBIGUOUS_PRONOUN.search(prompt):
        _add(
            "info",
            "ambiguous_pronoun",
            "Ambiguous pronoun detected (\"it\", \"this\", \"that\", or \"they\" "
            "used at sentence start without a clear antecedent). Refer to subjects "
            "explicitly to avoid misinterpretation.",
        )

    # --- Missing constraints for creative/generative output -----------------------
    if _CREATIVE_KEYWORDS.search(prompt) and not _CONSTRAINT_KEYWORDS.search(prompt):
        _add(
            "info",
            "missing_constraints",
            "Prompt requests creative or generative output but specifies no "
            "constraints on length, format, tone, or scope. Adding constraints "
            "reduces variance in model output.",
        )

    # --- Duplicate instruction detection -----------------------------------------
    if _has_duplicate_instructions(prompt):
        _add(
            "warning",
            "duplicate_instruction",
            "Near-duplicate instructions detected. Repeated instructions can "
            "confuse the model or inflate prompt token usage — consolidate them.",
        )

    return findings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_prompt(prompt: str, target_model: str) -> dict:
    """Run all rule-based checks against a prompt.

    Args:
        prompt: The prompt text to validate.
        target_model: One of ``"claude"``, ``"gpt"``, or ``"gemini"``.

    Returns:
        A dict with:
        - ``passed`` (bool): ``True`` if there are no errors.
        - ``warnings`` (list[str]): Messages for severity ``"warning"`` findings.
        - ``errors`` (list[str]): Messages for severity ``"error"`` findings.
        - ``findings`` (list[dict]): Full structured findings; each dict has
          keys ``severity``, ``check``, and ``message``. Includes all severity
          levels (info, warning, error).
    """
    findings = _run_checks(prompt, target_model)

    errors = [f["message"] for f in findings if f["severity"] == "error"]
    warnings = [f["message"] for f in findings if f["severity"] == "warning"]

    return {
        "passed": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
        "findings": findings,
    }

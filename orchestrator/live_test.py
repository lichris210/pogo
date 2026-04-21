"""Live testing utilities for prompt review."""

from __future__ import annotations

import re
from time import perf_counter

from orchestrator.agent_router import (
    LIGHT_MODEL_ID,
    invoke_agent_raw,
    resolve_target_model_id,
)

INPUT_GENERATOR_SYSTEM = """\
You create short, realistic test inputs for prompt evaluation.

Given a prompt, generate one brief input that would exercise its main
functionality. Keep the input under 200 words. Make it concrete and
representative of a normal request, not adversarial edge-case fuzzing.

Return only the test input text. Do not add commentary, labels, or code
fences.
"""


def run_live_test(prompt: str, target_model: str) -> dict:
    """Run a live test against the configured target model.

    Returns a dict with:
      - ``output``: the model output or a graceful failure message
      - ``latency_ms``: total elapsed time for input generation + inference
      - ``tokens_used``: total tokens across both calls when available
      - ``sample_input``: the generated test input used for the live test
    """
    started = perf_counter()
    total_tokens = 0
    sample_input = ""

    try:
        input_result = invoke_agent_raw(
            agent_name="live_test_input",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Given this prompt, generate a realistic but brief "
                                "test input that would exercise its main "
                                "functionality. Keep it under 200 words.\n\n"
                                f"Prompt:\n{prompt}"
                            ),
                        }
                    ],
                }
            ],
            system=INPUT_GENERATOR_SYSTEM,
            model_id=LIGHT_MODEL_ID,
            max_tokens=250,
        )
        total_tokens += input_result["usage"]["total_tokens"]
        sample_input = _clean_generated_input(input_result["text"]) or _fallback_test_input(prompt)

        output_result = invoke_agent_raw(
            agent_name="live_test",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": sample_input}],
                }
            ],
            system=prompt,
            model_id=resolve_target_model_id(target_model),
            max_tokens=700,
        )
        total_tokens += output_result["usage"]["total_tokens"]
        output_text = output_result["text"].strip()
        if not total_tokens:
            total_tokens = _estimate_tokens(prompt) + _estimate_tokens(sample_input) + _estimate_tokens(output_text)
        return {
            "output": output_text,
            "latency_ms": int((perf_counter() - started) * 1000),
            "tokens_used": total_tokens,
            "sample_input": sample_input,
        }
    except Exception as exc:
        if sample_input and not total_tokens:
            total_tokens = _estimate_tokens(prompt) + _estimate_tokens(sample_input)
        return {
            "output": f"Live test unavailable: {_clean_error(exc)}",
            "latency_ms": int((perf_counter() - started) * 1000),
            "tokens_used": total_tokens,
            "sample_input": sample_input,
        }


def _clean_generated_input(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:\w+)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"^(?:Test input|Input)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _fallback_test_input(prompt: str) -> str:
    lower = prompt.lower()
    if any(term in lower for term in ("csv", "sql", "dataset", "data analysis", "analytics")):
        return (
            "Analyze this sample dataset: 8 weeks of customer signups, churn, and "
            "support tickets for three subscription tiers. Identify the main churn "
            "driver and suggest two retention actions."
        )
    if any(term in lower for term in ("code", "python", "typescript", "function", "api", "class")):
        return (
            "Write a small Python function that groups a list of orders by customer "
            "ID and returns total spend per customer, with a short explanation."
        )
    if any(term in lower for term in ("essay", "blog", "article", "email", "documentation", "readme")):
        return (
            "Draft a concise internal email explaining a one-day product outage, "
            "its impact, and the immediate remediation plan."
        )
    if any(term in lower for term in ("research", "compare", "evaluate", "pros and cons")):
        return (
            "Compare managed PostgreSQL options for a small SaaS team with a focus "
            "on backups, scaling, and operational overhead."
        )
    return (
        "Handle this common request using the instructions above: summarize the "
        "task, produce the requested output format, and keep the result concise."
    )


def _estimate_tokens(text: str) -> int:
    words = len(text.split())
    return max(1, int(words * 1.3))


def _clean_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return re.sub(r"\s+", " ", message)[:240]

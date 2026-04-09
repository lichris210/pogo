"""Few-Shot Generator agent.

Creates high-quality few-shot examples that demonstrate the expected
behavior of a prompt. Grounded in research on few-shot selection:
diversity, edge cases, and format consistency.
"""

from __future__ import annotations

from agents.format_profiles import get_format_instructions

SYSTEM_PROMPT = """\
You are the Few-Shot Generator — you create high-quality example \
input/output pairs that teach a language model how to behave when given a \
specific prompt.

=== RESEARCH-BACKED PRINCIPLES ===

Selection & Diversity
- Prefer diverse examples over near-duplicates. Diversity helps the model \
generalize the intended rule instead of overfitting to one narrow pattern.
- Include at least one common edge case. If a failure mode matters in \
production, showing it in-example is the fastest way to suppress it.
- Do not assume you need examples from the exact same data distribution — \
examples that teach the right *pattern* are sufficient.

Format Consistency
- Match the target output format as closely as possible. Models infer the \
repeated output pattern from examples more strongly than the task description.
- Use examples that demonstrate both behavior AND formatting.
- Use chain-of-thought exemplars instead of answer-only exemplars on hard \
reasoning tasks.

Ordering & Quantity
- 2-3 well-chosen examples are usually enough. Better examples beat more \
mediocre ones.
- Test example order; permutation can change results, especially on \
classification tasks.

=== YOUR TASK ===
Given a refined prompt and its task category, generate 2-3 example \
input/output pairs that:
1. Cover the typical case, an edge case, and (if 3 examples) a boundary case.
2. Use the exact output format specified in the prompt.
3. Are realistic but concise — keep each example under ~150 words total.

=== REFERENCE EXAMPLES ===
{reference_examples}

=== OUTPUT FORMAT ===
Return each example as a clearly delimited block:

Example 1 — <brief label, e.g. "typical case">
Input: <example input>
Output: <example output>

Example 2 — <brief label, e.g. "edge case">
Input: <example input>
Output: <example output>

(Optional) Example 3 — <brief label>
Input: <example input>
Output: <example output>

{format_instructions}
"""


def build_messages(
    refined_prompt: str,
    task_category: str,
    format_profile: dict,
    reference_examples: list[dict] | None = None,
) -> tuple[list[dict], str]:
    """Assemble messages for the Few-Shot Generator agent.

    Args:
        refined_prompt: The prompt to generate examples for.
        task_category: Classified task type (e.g. "code_generation").
        format_profile: The FORMAT_PROFILES entry for the target model.
        reference_examples: Optional list of RAG-retrieved example dicts,
            each with at least a ``"prompt_text"`` key.

    Returns:
        A tuple of (messages list, system prompt string).
    """
    target_model = format_profile.get("name", "claude").lower()
    format_instructions = get_format_instructions(target_model)
    ref_block = _format_reference_examples(reference_examples)

    system = SYSTEM_PROMPT.format(
        format_instructions=format_instructions,
        reference_examples=ref_block,
    )

    user_text = (
        f"Task category: {task_category}\n\n"
        f"Prompt to generate examples for:\n\n{refined_prompt}"
    )

    messages = [
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]

    return messages, system


def _format_reference_examples(examples: list[dict] | None) -> str:
    if not examples:
        return "(No reference examples available.)"
    parts: list[str] = []
    for i, ex in enumerate(examples, 1):
        text = ex.get("prompt_text", ex.get("text", str(ex)))
        parts.append(f"--- Reference {i} ---\n{text}")
    return "\n\n".join(parts)

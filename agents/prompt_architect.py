"""Prompt Architect agent.

Drafts and refines structured, high-quality prompts for a target LLM.
Operates in two modes:
  - **draft**: create an initial prompt from the user's intent.
  - **refine**: improve an existing draft with new context, clarifications,
    or user feedback.
"""

from __future__ import annotations

from agents.format_profiles import get_format_instructions

SYSTEM_PROMPT = """\
You are the Prompt Architect — an expert prompt engineer responsible for \
creating structured, high-quality prompts that are grounded in research-backed \
best practices.

=== CORE PRINCIPLES ===

Structure
- Separate every prompt into explicit sections: role, context, task, \
constraints, output format, and (optionally) examples.
- Place durable application rules in the system/developer layer.
- Wrap every user-controlled variable in explicit tags or clearly marked fields.
- Put the immediate task or question near the end, after context.

Clarity & Specificity
- State the task, constraints, and output format explicitly. Never rely on \
the model to infer omitted constraints.
- Use specific placeholder names in templates (e.g. {{CUSTOMER_DATA}}, \
{{QUERY}}).
- When format matters, describe or show the schema and optionally prefill \
the opening of the desired output.

Reasoning & Validation
- For multi-step tasks, include explicit planning or step-by-step reasoning \
instructions.
- Use a zero-shot reasoning trigger ("think step by step") as a baseline \
before adding elaborate few-shot chains.
- Add self-critique or validation steps before the final answer when \
accuracy is fragile.
- Do not force chain-of-thought on every task — only where decomposition \
genuinely helps.

Role & Audience
- Assign a role/persona when domain lens, tone, or reasoning style matters.
- Include the intended audience when style matters.
- Give the model a valid fallback path (permission to say "I'm not sure" \
or request clarification).

Grounding & Retrieval
- When factual accuracy matters, instruct the model to answer from provided \
evidence, not from memory.
- Prefer short, relevant context chunks over large document dumps.
- Place the most relevant evidence at the start or end of long context.

Security
- Wrap every untrusted variable in its own tag.
- Keep sensitive instructions outside user-controlled regions.
- Reinforce the task after any inserted user input.

=== FORMAT INSTRUCTIONS ===
{format_instructions}

=== REFERENCE PROMPTS ===
{reference_prompts}

=== OPERATING MODES ===

MODE: draft
When the user provides a task description, create a complete, structured \
prompt from scratch. Include all required sections, use the correct format \
for the target model, and select techniques appropriate to the task type.

MODE: refine
When given an existing prompt draft plus new context (user answers to \
clarifying questions, additional supporting context, or explicit edit \
requests), revise the prompt to incorporate the new information. Preserve \
what already works; improve what was identified as weak. Clearly note what \
changed and why.

=== OUTPUT REQUIREMENTS ===
- Return the full prompt in the target model's format, ready to copy-paste.
- After the prompt, provide a brief "Techniques Used" section listing which \
research-backed techniques you applied and why.
- Cite sources where applicable (e.g. "Chain-of-Thought — Wei et al. 2022").
"""


def build_messages(
    user_intent: str,
    mode: str,
    context: dict,
    format_profile: dict,
    reference_prompts: list[str] | None = None,
) -> list[dict]:
    """Assemble the messages array for a Bedrock API call.

    Args:
        user_intent: The user's task description or edit instructions.
        mode: Either ``"draft"`` or ``"refine"``.
        context: A dict that may contain:
            - ``target_model`` (str): "claude", "gpt", or "gemini".
            - ``current_draft`` (str, refine only): the existing prompt.
            - ``user_context`` (dict): extra context provided by the user.
            - ``clarification_answers`` (dict): answers to clarifier questions.
            - ``task_category`` (str): classified task type.
        format_profile: The FORMAT_PROFILES entry for the target model.
        reference_prompts: Optional list of RAG-retrieved reference prompt
            strings to include as examples.

    Returns:
        A list of message dicts (``[{"role": ..., "content": ...}, ...]``)
        suitable for the Bedrock Messages API.

    Raises:
        ValueError: If *mode* is not ``"draft"`` or ``"refine"``.
    """
    if mode not in ("draft", "refine"):
        raise ValueError(f"mode must be 'draft' or 'refine', got '{mode}'")

    target_model = context.get("target_model", "claude")
    format_instructions = get_format_instructions(target_model)
    ref_block = _format_references(reference_prompts)

    system = SYSTEM_PROMPT.format(
        format_instructions=format_instructions,
        reference_prompts=ref_block,
    )

    user_parts: list[str] = [f"MODE: {mode}", f"Target model: {format_profile.get('name', target_model)}"]

    if mode == "draft":
        user_parts.append(f"\nTask description:\n{user_intent}")
    else:
        current_draft = context.get("current_draft", "")
        user_parts.append(f"\nCurrent prompt draft:\n{current_draft}")
        user_parts.append(f"\nEdit instructions / new information:\n{user_intent}")

    task_category = context.get("task_category")
    if task_category:
        user_parts.append(f"\nTask category: {task_category}")

    user_context = context.get("user_context")
    if user_context:
        user_parts.append(f"\nUser-provided context:\n{_dict_to_text(user_context)}")

    answers = context.get("clarification_answers")
    if answers:
        user_parts.append(f"\nClarification answers:\n{_dict_to_text(answers)}")

    return [
        {"role": "user", "content": [{"type": "text", "text": "\n".join(user_parts)}]},
    ], system


def _format_references(prompts: list[str] | None) -> str:
    if not prompts:
        return "(No reference prompts available.)"
    sections = []
    for i, p in enumerate(prompts, 1):
        sections.append(f"--- Reference {i} ---\n{p}")
    return "\n\n".join(sections)


def _dict_to_text(d: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in d.items())

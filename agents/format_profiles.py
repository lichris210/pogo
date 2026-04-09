"""Format profiles for target models.

Each profile defines the wrapper format, section delimiters, example style,
and best practices that every agent inherits when generating prompts for a
given target model (Claude, GPT, Gemini).
"""

from __future__ import annotations

FORMAT_PROFILES: dict[str, dict] = {
    "claude": {
        "name": "Claude",
        "wrapper_format": "xml",
        "section_open": "<{name}>",
        "section_close": "</{name}>",
        "example_open": "<example>",
        "example_close": "</example>",
        "best_practices": [
            "Use XML tags to separate role, context, task, constraints, and output_format sections.",
            "Place durable rules in the system/developer message.",
            "Wrap every user-controlled variable in its own XML tag.",
            "Use role/persona framing to set reasoning style and tone.",
            "Put the immediate task near the end, after context.",
            "Prefill the assistant turn with an opening tag to steer output shape.",
            "Chain-of-thought works well; ask the model to think step-by-step inside <thinking> tags.",
        ],
    },
    "gpt": {
        "name": "GPT",
        "wrapper_format": "markdown",
        "section_open": "## {name}",
        "section_close": "",
        "example_open": "---\n**Example:**",
        "example_close": "---",
        "best_practices": [
            "Use markdown headers and bullet lists for structure.",
            "State the task, constraints, and output format explicitly.",
            "Provide explicit JSON schemas when structured output is needed.",
            "Use few-shot examples with clear input/output pairs.",
            "Give step-by-step instructions rather than high-level goals.",
            "Avoid overly conversational framing; be direct and literal.",
            "Use structured outputs / JSON mode when exact format matters.",
        ],
    },
    "gemini": {
        "name": "Gemini",
        "wrapper_format": "markdown",
        "section_open": "## {name}",
        "section_close": "",
        "example_open": "---\n**Example:**",
        "example_close": "---",
        "best_practices": [
            "Use markdown headers to organize long prompts into clear sections.",
            "Place the most relevant context at the beginning or end, not the middle.",
            "Leverage long-context capability but prefer short, relevant chunks over large dumps.",
            "Use explicit context placement and structured reasoning.",
            "For multimodal tasks, ask the model to describe the input before higher-level analysis.",
            "Use document grounding; tell the model to answer from provided evidence.",
            "Reduce temperature and shorten output if hallucination is a concern.",
        ],
    },
}


def get_format_instructions(target_model: str) -> str:
    """Return a block of formatting instructions for injection into an agent system prompt.

    Args:
        target_model: One of "claude", "gpt", or "gemini".

    Returns:
        A multi-line string describing how to format the generated prompt.

    Raises:
        ValueError: If *target_model* is not a recognised key.
    """
    profile = FORMAT_PROFILES.get(target_model)
    if profile is None:
        raise ValueError(
            f"Unknown target model '{target_model}'. "
            f"Choose from: {', '.join(FORMAT_PROFILES)}"
        )

    wrapper = profile["wrapper_format"]
    practices = "\n".join(f"- {p}" for p in profile["best_practices"])

    if wrapper == "xml":
        structure_guidance = (
            "Structure the prompt using XML tags. Use tags like <role>, <context>, "
            "<task>, <constraints>, <output_format>, and <examples> to delimit "
            "each section. Wrap user-provided variables in their own descriptive "
            "tags (e.g. <user_input>, <document>)."
        )
    else:
        structure_guidance = (
            "Structure the prompt using Markdown. Use ## headers for each major "
            "section (Role, Context, Task, Constraints, Output Format, Examples). "
            "Use bullet lists for rules and numbered lists for sequential steps."
        )

    return (
        f"=== FORMAT INSTRUCTIONS (target: {profile['name']}) ===\n"
        f"Wrapper format: {wrapper}\n\n"
        f"{structure_guidance}\n\n"
        f"Best practices for {profile['name']}:\n"
        f"{practices}\n"
        f"=== END FORMAT INSTRUCTIONS ==="
    )


def format_section(content: str, section_name: str, target_model: str) -> str:
    """Wrap *content* in the appropriate section delimiters for *target_model*.

    Args:
        content: The text to wrap.
        section_name: A short label (e.g. "context", "task").
        target_model: One of "claude", "gpt", or "gemini".

    Returns:
        The content wrapped with the model's section tags/headers.

    Raises:
        ValueError: If *target_model* is not a recognised key.
    """
    profile = FORMAT_PROFILES.get(target_model)
    if profile is None:
        raise ValueError(
            f"Unknown target model '{target_model}'. "
            f"Choose from: {', '.join(FORMAT_PROFILES)}"
        )

    opening = profile["section_open"].format(name=section_name)
    closing = profile["section_close"].format(name=section_name) if profile["section_close"] else ""

    if closing:
        return f"{opening}\n{content}\n{closing}"
    return f"{opening}\n\n{content}"

"""POGO v2 agent modules.

Each module defines a SYSTEM_PROMPT and a ``build_messages()`` function
(except guardrails, which is purely rule-based).

Quick-start::

    from agents import format_profiles, prompt_architect, critic

    profile = format_profiles.FORMAT_PROFILES["claude"]
    messages, system = prompt_architect.build_messages(
        user_intent="Summarize meeting notes",
        mode="draft",
        context={"target_model": "claude"},
        format_profile=profile,
    )
"""

from agents import (  # noqa: F401
    clarifier,
    context_scout,
    critic,
    fewshot_generator,
    format_profiles,
    guardrails,
    prompt_architect,
)

__all__ = [
    "clarifier",
    "context_scout",
    "critic",
    "fewshot_generator",
    "format_profiles",
    "guardrails",
    "prompt_architect",
]

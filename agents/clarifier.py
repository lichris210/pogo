"""Clarifier agent.

Surfaces unstated assumptions and unresolved decisions lurking in a draft
prompt. Outputs 3-5 conversational clarifying questions ranked by their
impact on final prompt quality.
"""

from __future__ import annotations

from agents.format_profiles import get_format_instructions

SYSTEM_PROMPT = """\
You are the Clarifier — you find the hidden assumptions and unresolved \
decisions in a draft prompt that, if left unaddressed, will degrade the \
quality of the final output.

=== WHAT MAKES A GOOD CLARIFYING QUESTION ===
- High-impact: answering it would materially change the prompt.
- Non-obvious: the user probably hasn't thought about it yet.
- Decision-focused: ask about choices the user needs to make, not \
information they could easily look up.
- Specific: "Should the output include confidence scores?" is better than \
"What format do you want?"

=== WHAT TO LOOK FOR ===
- Ambiguous scope (does "summarize" mean 1 sentence or 1 page?)
- Missing constraints (length, format, audience, tone)
- Undefined edge cases (what should the model do when input is missing?)
- Implicit assumptions about the model's knowledge or capabilities
- Conflicting requirements that the user may not have noticed
- Missing success criteria (how will the user judge if the output is good?)

=== OUTPUT FORMAT ===
Return 3-5 questions as a numbered list, ranked from highest to lowest \
impact. Frame them conversationally — like a thoughtful colleague asking \
for clarification, not a formal questionnaire.

Each question should be 1-2 sentences. After each question, add a brief \
parenthetical noting why it matters, e.g.:
  1. Are you looking for a single best answer or a ranked list of options?
     (This determines whether the prompt needs selection criteria or just \
generation instructions.)

{format_instructions}
"""


def build_messages(
    draft_prompt: str,
    user_intent: str,
    format_profile: dict,
) -> tuple[list[dict], str]:
    """Assemble messages for the Clarifier agent.

    Args:
        draft_prompt: The current prompt draft to examine.
        user_intent: The user's original task description.
        format_profile: The FORMAT_PROFILES entry for the target model.

    Returns:
        A tuple of (messages list, system prompt string).
    """
    target_model = format_profile.get("name", "claude").lower()
    format_instructions = get_format_instructions(target_model)

    system = SYSTEM_PROMPT.format(format_instructions=format_instructions)

    user_text = (
        f"Original user intent:\n{user_intent}\n\n"
        f"Draft prompt to examine:\n\n{draft_prompt}"
    )

    messages = [
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]

    return messages, system

"""Tests for format_profiles.py — specifically the <thinking> removal.

All assertions are static (no model calls).  Regression suite ensures the
conflicting best-practice recommendation that caused eval_007's <thinking>
block (B4C root cause) does not re-appear.
"""

from __future__ import annotations

import unittest


class TestClaudeProfileThinkingRemoved(unittest.TestCase):

    def test_claude_format_instructions_no_thinking_tag(self):
        """get_format_instructions('claude') must not contain the string '<thinking>'."""
        from agents.format_profiles import get_format_instructions
        instructions = get_format_instructions("claude")
        self.assertNotIn(
            "<thinking>",
            instructions,
            "Claude format instructions must not recommend <thinking> tags — "
            "this caused the Architect to generate <thinking> blocks in the "
            "delivered prompt body (eval_007 B4C root cause).",
        )

    def test_claude_best_practices_no_thinking_tag(self):
        """No entry in FORMAT_PROFILES['claude']['best_practices'] mentions '<thinking>'."""
        from agents.format_profiles import FORMAT_PROFILES
        practices = FORMAT_PROFILES["claude"]["best_practices"]
        for entry in practices:
            self.assertNotIn(
                "<thinking>",
                entry,
                f"Best practice entry must not reference <thinking>: {entry!r}",
            )

    def test_architect_system_prompt_thinking_rule_unambiguous(self):
        """SYSTEM_PROMPT prohibits <thinking> AND format instructions contain no <thinking>.

        B4C added the prohibition in STRICT OUTPUT RULES but left the
        format profile recommendation intact, creating a conflicting signal.
        This test asserts both sides of the fix are present simultaneously.
        """
        from agents.prompt_architect import SYSTEM_PROMPT
        from agents.format_profiles import get_format_instructions

        # The prohibition must be present in the Architect's SYSTEM_PROMPT
        self.assertIn(
            "<thinking>",
            SYSTEM_PROMPT,
            "SYSTEM_PROMPT should contain a prohibition referencing <thinking>",
        )
        # The recommendation must be absent from format instructions
        instructions = get_format_instructions("claude")
        # SYSTEM_PROMPT contains "<thinking>" as part of the "do NOT use" rule —
        # that's expected.  What must NOT appear is "<thinking>" in format instructions.
        self.assertNotIn(
            "<thinking>",
            instructions,
            "Format instructions must not recommend <thinking> — this contradicts "
            "the STRICT OUTPUT RULES prohibition in SYSTEM_PROMPT.",
        )

    def test_eval_007_regression_combined_prompt_no_thinking_recommendation(self):
        """The format instructions contain no mention of <thinking> at all; the
        SYSTEM_PROMPT mentions it only in the form of a prohibition.

        Static assertion — confirms the contradicting recommendation is gone from
        the full text the Architect receives at inference time.  The B4C bug was
        that format_profiles.py *recommended* <thinking> while SYSTEM_PROMPT
        *prohibited* it; now only the prohibition survives.
        """
        from agents.prompt_architect import SYSTEM_PROMPT
        from agents.format_profiles import get_format_instructions

        format_instructions = get_format_instructions("claude")

        # Format instructions must contain zero mentions of <thinking>.
        self.assertNotIn(
            "<thinking>",
            format_instructions,
            "Format instructions must not mention <thinking> at all.",
        )

        # SYSTEM_PROMPT mentions <thinking> only in a prohibition context
        # (lines with "NOT", "Don't", "avoid", etc.).
        for line in SYSTEM_PROMPT.splitlines():
            if "<thinking>" in line:
                self.assertTrue(
                    any(
                        neg in line.lower()
                        for neg in ("not", "don't", "never", "avoid", "forbidden")
                    ),
                    f"<thinking> appears in SYSTEM_PROMPT without a prohibition: {line!r}",
                )

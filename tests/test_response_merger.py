"""Tests for orchestrator/response_merger.py.

Covers _extract_prompt_block extraction logic, including the eval_011 regression
(Gemini prompts that start directly with ## Role and contain inner code fences).
"""

from __future__ import annotations

import pathlib
import unittest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"

# eval_011 attempt-1 raw Architect response (3,811-char Gemini translation
# prompt). Starts with ## Role — no outer code fence. Contains nested
# ```markdown / ```bash inner fences in the ## Example section.
# This is the exact shape that caused ArchitectOutputError across B4C and B4D:
# the old regex found the first ```markdown fence and extracted only the
# example content, making all four required sections appear missing.
_EVAL_011_RAW = (_FIXTURES_DIR / "eval_011_architect_raw.txt").read_text(
    encoding="utf-8"
)


# ---------------------------------------------------------------------------
# TestExtractPromptBlockEval011Regression
# ---------------------------------------------------------------------------

class TestExtractPromptBlockEval011Regression(unittest.TestCase):
    """Regression guard for the eval_011 extraction failure.

    Root cause (B4D investigation): _extract_prompt_block scanned the full
    text for any code fence, found the first ```markdown fence inside the
    ## Example section, and extracted only that fragment — discarding
    ## Role / ## Context / ## Task / ## Constraints.

    Fix: only attempt fence extraction when the text *starts* with a fence.
    """

    def _extract(self, text: str) -> str:
        from orchestrator.response_merger import _extract_prompt_block
        return _extract_prompt_block(text)

    def test_extract_prompt_block_markdown_no_outer_fence(self):
        """eval_011 raw response: all four required headers present in result."""
        result = self._extract(_EVAL_011_RAW)
        for header in ("## Role", "## Context", "## Task", "## Constraints"):
            self.assertIn(
                header,
                result,
                f"_extract_prompt_block dropped '{header}' from a no-outer-fence "
                f"Gemini response — inner fence extraction fired incorrectly.",
            )

    def test_missing_architect_sections_empty_after_fix(self):
        """After extraction, _missing_architect_sections returns [] for eval_011."""
        from orchestrator.response_merger import _extract_prompt_block
        from orchestrator.orchestrator import _missing_architect_sections
        extracted = _extract_prompt_block(_EVAL_011_RAW)
        missing = _missing_architect_sections(extracted, "gemini")
        self.assertEqual(
            missing,
            [],
            f"_missing_architect_sections still reports missing sections after "
            f"extraction fix: {missing}",
        )

    def test_full_text_returned_when_no_outer_fence(self):
        """A response starting with ## Role is returned unchanged (stripped)."""
        text = (
            "## Role\n\nYou are a translator.\n\n"
            "## Task\n\nTranslate the text.\n\n"
            "## Constraints\n\nDo not translate code.\n\n"
            "## Context\n\nThe document is markdown."
        )
        result = self._extract(text)
        self.assertEqual(result, text.strip())

    def test_inner_fence_not_mistaken_for_outer(self):
        """Content starting with ## headers but containing inner ```markdown
        fences must return the full text, not just the inner fence content."""
        text = (
            "## Role\n\nYou are a specialist.\n\n"
            "## Task\n\nTranslate the following:\n\n"
            "## Constraints\n\nPreserve code verbatim.\n\n"
            "## Context\n\nSee example below.\n\n"
            "## Examples\n\n"
            "**Input:**\n"
            "```markdown\n"
            "# Release Notes v1.0\n"
            "- Feature A\n"
            "```\n\n"
            "**Output:**\n"
            "```markdown\n"
            "# Notas de Versão v1.0\n"
            "- Recurso A\n"
            "```\n"
        )
        result = self._extract(text)
        self.assertIn("## Role", result)
        self.assertIn("## Constraints", result)
        self.assertIn("## Examples", result)
        # Inner example content should also be present (not extracted in isolation)
        self.assertIn("Release Notes v1.0", result)

    def test_fence_wrapped_response_still_extracted(self):
        """A response that *does* start with a ``` fence is still unwrapped
        correctly — Claude-target Architect outputs are unaffected."""
        text = (
            "```\n"
            "<role>You are an analyst.</role>\n"
            "<context>Analyse the data.</context>\n"
            "<task>Summarise findings.</task>\n"
            "<constraints>Be concise.</constraints>\n"
            "```\n"
        )
        result = self._extract(text)
        self.assertIn("<role>", result)
        self.assertIn("<constraints>", result)
        self.assertNotIn("```", result)


# ---------------------------------------------------------------------------
# TestExtractPromptBlockExistingBehaviour (guard existing contracts)
# ---------------------------------------------------------------------------

class TestExtractPromptBlockExistingBehaviour(unittest.TestCase):
    """Confirm that pre-existing extract behaviour is unchanged for
    fence-wrapped responses (the common Claude-target path)."""

    def _extract(self, text: str) -> str:
        from orchestrator.response_merger import _extract_prompt_block
        return _extract_prompt_block(text)

    def test_simple_fence(self):
        result = self._extract("```\n<role>r</role>\n<task>t</task>\n```")
        self.assertIn("<role>", result)
        self.assertNotIn("```", result)

    def test_fence_with_language_specifier(self):
        result = self._extract("```xml\n<role>You are an analyst.</role>\n```")
        self.assertIn("<role>", result)
        self.assertNotIn("```", result)

    def test_no_fence_raw_text_returned(self):
        text = "You are a helpful assistant. Be concise and clear."
        self.assertEqual(self._extract(text), text)

    def test_short_fence_content_falls_back_to_full_text(self):
        """Extracted content < 20 chars → return full text (the fallback)."""
        text = "```\nshort\n```\n\n## Role\n\nFull prompt here."
        result = self._extract(text)
        # short extraction (5 chars) triggers fallback
        self.assertIn("## Role", result)

    def test_nested_fence_in_fence_wrapped_response(self):
        """Fence-wrapped Claude response containing inner ```markdown block
        → outer content (including inner fence text) correctly extracted."""
        text = (
            "```\n"
            "## Role\nYou are a translator.\n\n"
            "## Task\nTranslate the following:\n\n"
            "```markdown\n"
            "# Release Notes v2.3.1\n"
            "- Fixed Snowdrift bug\n"
            "```\n"
            "\n"
            "Preserve code spans verbatim.\n"
            "```"
        )
        result = self._extract(text)
        self.assertIn("## Role", result)
        self.assertIn("## Task", result)
        self.assertIn("Snowdrift", result)

"""Tests for the structural guardrail checks (check_structural_integrity).

All tests are offline — no model calls.  Adversarial fixtures use verbatim
content from the B4C eval captures (eval_001 / eval_007 final_output fields).
"""

from __future__ import annotations

import json
import pathlib
import unittest

from agents.guardrails import check_structural_integrity

# ---------------------------------------------------------------------------
# Load B4C adversarial fixtures
# ---------------------------------------------------------------------------

_RUN_FILE = (
    pathlib.Path(__file__).parent.parent
    / "eval" / "runs" / "2026-05-20_eecb37c_b4c-validation.json"
)

with _RUN_FILE.open(encoding="utf-8") as _f:
    _B4C = json.load(_f)

_RESULTS_BY_ID = {r["eval_id"]: r["captured"] for r in _B4C["results"]}

_EVAL_001_FINAL = _RESULTS_BY_ID["eval_001"]["final_output"]
_EVAL_007_FINAL = _RESULTS_BY_ID["eval_007"]["final_output"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_ids(prompt: str, target: str = "claude") -> list[str]:
    return [f["check"] for f in check_structural_integrity(prompt, target)["findings"]]


def _passed(prompt: str, target: str = "claude") -> bool:
    return check_structural_integrity(prompt, target)["passed"]


# ---------------------------------------------------------------------------
# TestDuplicateExamplesTag
# ---------------------------------------------------------------------------

class TestDuplicateExamplesTag(unittest.TestCase):

    def test_eval_001_final_output_triggers(self):
        """eval_001 final_output has two <examples> blocks → duplicate_section_tag fires."""
        checks = _check_ids(_EVAL_001_FINAL)
        self.assertIn("duplicate_section_tag", checks)

    def test_eval_007_final_output_triggers(self):
        """eval_007 final_output has a nested <examples> opening → duplicate_section_tag fires."""
        checks = _check_ids(_EVAL_007_FINAL)
        self.assertIn("duplicate_section_tag", checks)

    def test_single_examples_block_passes(self):
        """One <examples> block with balanced tags → passes."""
        prompt = (
            "<role>You are a classifier.</role>\n"
            "<task>Classify the input.</task>\n"
            "<examples>\n<example>Input: x\nOutput: y</example>\n</examples>"
        )
        self.assertTrue(_passed(prompt))

    def test_duplicate_role_triggers(self):
        prompt = "<role>A</role>\n<task>Do it.</task>\n<role>B</role>"
        checks = _check_ids(prompt)
        self.assertIn("duplicate_section_tag", checks)

    def test_markdown_duplicate_examples_triggers(self):
        prompt = (
            "## Role\nYou are an assistant.\n\n"
            "## Task\nTranslate text.\n\n"
            "## Examples\nExample 1.\n\n"
            "## Examples\nExample 2."
        )
        checks = _check_ids(prompt, "gemini")
        self.assertIn("duplicate_section_tag", checks)


# ---------------------------------------------------------------------------
# TestThinkingBlock
# ---------------------------------------------------------------------------

class TestThinkingBlock(unittest.TestCase):

    def test_eval_007_thinking_fires(self):
        """eval_007 final_output contains <thinking> inside <task> → thinking_block fires."""
        checks = _check_ids(_EVAL_007_FINAL)
        self.assertIn("thinking_block", checks)

    def test_no_thinking_passes(self):
        prompt = (
            "<role>You are a helpful assistant.</role>\n"
            "<task>Answer the question. Think step by step before answering.</task>"
        )
        checks = _check_ids(prompt)
        self.assertNotIn("thinking_block", checks)

    def test_thinking_nested_in_role(self):
        prompt = "<role>You are <thinking>an expert</thinking> assistant.</role>"
        checks = _check_ids(prompt)
        self.assertIn("thinking_block", checks)


# ---------------------------------------------------------------------------
# TestTechniquesUsedMarker
# ---------------------------------------------------------------------------

class TestTechniquesUsedMarker(unittest.TestCase):

    def test_bold_triggers(self):
        prompt = "You are a helpful assistant.\n\n**Techniques Used:** chain-of-thought"
        checks = _check_ids(prompt)
        self.assertIn("techniques_used_marker", checks)

    def test_bare_triggers(self):
        prompt = "You are a helpful assistant.\n\nTechniques Used: few-shot, role-play"
        checks = _check_ids(prompt)
        self.assertIn("techniques_used_marker", checks)

    def test_no_marker_passes(self):
        prompt = (
            "<role>You are a helpful assistant.</role>\n"
            "<task>Answer questions concisely.</task>"
        )
        checks = _check_ids(prompt)
        self.assertNotIn("techniques_used_marker", checks)


# ---------------------------------------------------------------------------
# TestUnbalancedXmlTags
# ---------------------------------------------------------------------------

class TestUnbalancedXmlTags(unittest.TestCase):

    def test_unclosed_context_fires(self):
        prompt = "<role>Engineer</role>\n<context>Some context\n<task>Do it.</task>"
        checks = _check_ids(prompt)
        self.assertIn("unbalanced_xml_tag", checks)

    def test_extra_closing_fires(self):
        prompt = "<role>Engineer</role>\n<task>Do it.</task>\n</task>"
        checks = _check_ids(prompt)
        self.assertIn("unbalanced_xml_tag", checks)

    def test_balanced_prompt_passes(self):
        prompt = (
            "<role>You are an engineer.</role>\n"
            "<context>Build a Flask app.</context>\n"
            "<task>Write the code.</task>\n"
            "<constraints>Use Python 3.11.</constraints>"
        )
        self.assertTrue(_passed(prompt))

    def test_nested_example_pair_passes(self):
        prompt = (
            "<role>You are a classifier.</role>\n"
            "<task>Classify input.</task>\n"
            "<examples><example>Input: x\nOutput: y</example></examples>"
        )
        self.assertTrue(_passed(prompt))


# ---------------------------------------------------------------------------
# TestTruncatedExampleTag
# ---------------------------------------------------------------------------

class TestTruncatedExampleTag(unittest.TestCase):

    def test_empty_example_fires(self):
        prompt = (
            "<role>You are a classifier.</role>\n"
            "<task>Classify input.</task>\n"
            "<examples><example></example></examples>"
        )
        checks = _check_ids(prompt)
        self.assertIn("truncated_example_tag", checks)

    def test_whitespace_example_fires(self):
        prompt = (
            "<role>You are a classifier.</role>\n"
            "<task>Classify input.</task>\n"
            "<examples><example>   \n  </example></examples>"
        )
        checks = _check_ids(prompt)
        self.assertIn("truncated_example_tag", checks)

    def test_populated_example_passes(self):
        prompt = (
            "<role>You are a classifier.</role>\n"
            "<task>Classify input.</task>\n"
            "<examples><example>Input: foo\nOutput: bar</example></examples>"
        )
        checks = _check_ids(prompt)
        self.assertNotIn("truncated_example_tag", checks)


# ---------------------------------------------------------------------------
# TestCheckStructuralIntegrityAPI
# ---------------------------------------------------------------------------

class TestCheckStructuralIntegrityAPI(unittest.TestCase):

    def test_return_shape(self):
        result = check_structural_integrity("any prompt text", "claude")
        self.assertIn("passed", result)
        self.assertIn("errors", result)
        self.assertIn("findings", result)
        self.assertIsInstance(result["passed"], bool)
        self.assertIsInstance(result["errors"], list)
        self.assertIsInstance(result["findings"], list)

    def test_no_errors_passed_true(self):
        prompt = (
            "<role>You are a senior engineer.</role>\n"
            "<context>Code review context.</context>\n"
            "<task>Review this diff.</task>\n"
            "<constraints>Be concise.</constraints>"
        )
        result = check_structural_integrity(prompt, "claude")
        self.assertTrue(result["passed"])
        self.assertEqual(result["errors"], [])

    def test_errors_list_matches_findings(self):
        """errors list contains one entry per finding."""
        prompt = "<role>A</role>\n<role>B</role>"
        result = check_structural_integrity(prompt, "claude")
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["errors"]), len(result["findings"]))

    def test_findings_have_required_keys(self):
        prompt = "<role>A</role>\n<role>B</role>"
        result = check_structural_integrity(prompt, "claude")
        for finding in result["findings"]:
            self.assertIn("check", finding)
            self.assertIn("message", finding)

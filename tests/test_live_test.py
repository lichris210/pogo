"""Unit tests for the dedicated live-testing pipeline."""

from __future__ import annotations

import unittest
from unittest.mock import patch


class TestCleanGeneratedInput(unittest.TestCase):
    """Coverage for ``_clean_generated_input``.

    The input-generator helper strips fenced code blocks and the
    ``Test input:`` / ``Input:`` labels models tend to add.
    """

    def test_strips_fenced_code_block_and_label(self):
        from orchestrator.live_test import _clean_generated_input
        raw = "```text\nInput: Summarize the report.\n```"
        self.assertEqual(_clean_generated_input(raw), "Summarize the report.")

    def test_strips_test_input_prefix_case_insensitive(self):
        from orchestrator.live_test import _clean_generated_input
        self.assertEqual(
            _clean_generated_input("test input: Build a sort function"),
            "Build a sort function",
        )

    def test_passes_clean_text_through(self):
        from orchestrator.live_test import _clean_generated_input
        self.assertEqual(_clean_generated_input("Just plain text"), "Just plain text")


class TestFallbackTestInput(unittest.TestCase):
    """``_fallback_test_input`` routes by keyword to a domain-shaped input."""

    def test_data_branch(self):
        from orchestrator.live_test import _fallback_test_input
        out = _fallback_test_input("Analyze this CSV of customer churn")
        self.assertIn("churn", out.lower())

    def test_code_branch(self):
        from orchestrator.live_test import _fallback_test_input
        out = _fallback_test_input("You are a Python expert. Write a function.")
        self.assertIn("python", out.lower())

    def test_default_branch_when_nothing_matches(self):
        from orchestrator.live_test import _fallback_test_input
        out = _fallback_test_input("Something with no recognised keyword token")
        self.assertIn("instructions above", out.lower())


class TestRunLiveTestFallback(unittest.TestCase):
    """When the input generator returns empty/garbage, the fallback kicks in."""

    @patch("orchestrator.live_test.resolve_target_model_id", return_value="anthropic.test-model")
    @patch("orchestrator.live_test.invoke_agent_raw")
    def test_empty_generator_output_uses_fallback(self, mock_invoke_raw, _model_id):
        from orchestrator.live_test import run_live_test

        mock_invoke_raw.side_effect = [
            {"text": "```\n   \n```", "usage": {"total_tokens": 5}},
            {"text": "Sample analysis output.", "usage": {"total_tokens": 12}},
        ]

        result = run_live_test("Analyze this CSV of customer churn", "claude")

        # Fallback for data prompts mentions churn.
        self.assertIn("churn", result["sample_input"].lower())
        self.assertEqual(result["tokens_used"], 17)


class TestLiveTest(unittest.TestCase):

    @patch("orchestrator.live_test.resolve_target_model_id", return_value="anthropic.test-model")
    @patch("orchestrator.live_test.invoke_agent_raw")
    def test_run_live_test_success(self, mock_invoke_raw, mock_model_id):
        from orchestrator.live_test import run_live_test

        mock_invoke_raw.side_effect = [
            {
                "text": "Input: Summarize the attached bug report in 3 bullets.",
                "usage": {"total_tokens": 21},
            },
            {
                "text": "1. Login fails on Safari.\n2. The error started after deploy.\n3. Rolling back fixes it.",
                "usage": {"total_tokens": 34},
            },
        ]

        result = run_live_test("You are a bug triage assistant.", "gpt")

        self.assertEqual(
            result["sample_input"],
            "Summarize the attached bug report in 3 bullets.",
        )
        self.assertIn("Login fails on Safari", result["output"])
        self.assertEqual(result["tokens_used"], 55)
        self.assertIsInstance(result["latency_ms"], int)
        self.assertGreaterEqual(result["latency_ms"], 0)

    @patch("orchestrator.live_test.resolve_target_model_id", return_value="anthropic.test-model")
    @patch("orchestrator.live_test.invoke_agent_raw")
    def test_run_live_test_failure_is_graceful(self, mock_invoke_raw, mock_model_id):
        from orchestrator.live_test import run_live_test

        mock_invoke_raw.side_effect = [
            {
                "text": "Input: Review this short SQL query and explain its purpose.",
                "usage": {"total_tokens": 13},
            },
            TimeoutError("anthropic call timed out"),
        ]

        result = run_live_test("You are a SQL explainer.", "claude")

        self.assertEqual(
            result["sample_input"],
            "Review this short SQL query and explain its purpose.",
        )
        self.assertIn("Live test unavailable", result["output"])
        self.assertEqual(result["tokens_used"], 13)
        self.assertIsInstance(result["latency_ms"], int)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the dedicated live-testing pipeline."""

from __future__ import annotations

import unittest
from unittest.mock import patch


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
            TimeoutError("bedrock timed out"),
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

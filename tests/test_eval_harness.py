"""Unit tests for the eval harness (eval/run_eval.py and eval/rate.py).

These tests cover:
  - The runner's per-entry capture logic (token accumulation, skip-on-no-context).
  - The rate CLI's file-update logic (rating apply + persist).
  - The rating-scale validator (1-5 integers only).
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from eval import rate, run_eval


# ---------------------------------------------------------------------------
# Runner: per-entry capture
# ---------------------------------------------------------------------------

class TestRunEntryCapture(unittest.TestCase):

    def test_skips_when_no_pre_baked_context(self):
        entry = {
            "id": "eval_999",
            "task_category": "code_generation",
            "target_model_family": "claude",
            "expected_path": "one_shot",
            "user_prompt": "Write something.",
            "pre_baked_context": "",
            "notes": "",
        }
        result = run_eval.run_entry(entry)
        self.assertTrue(result.skipped)
        self.assertIn("pre_baked_context", result.skip_reason)
        self.assertEqual(result.token_count, 0)

    def test_captures_pipeline_outputs_with_stubbed_anthropic(self):
        """End-to-end per-entry capture using the dry-run patch.

        Verifies architect_draft, critic_score, final_output, and token
        accounting all populate when the pipeline runs cleanly.
        """
        entry = {
            "id": "eval_001",
            "task_category": "code_generation",
            "target_model_family": "claude",
            "expected_path": "one_shot",
            "user_prompt": "Write a Python function that adds two numbers.",
            "pre_baked_context": "Use type hints and include a docstring.",
            "notes": "smoke",
        }

        with run_eval._dry_run_patch():
            result = run_eval.run_entry(entry)

        self.assertFalse(result.skipped, msg=f"unexpected skip: {result.skip_reason}")
        self.assertIsNone(result.error)
        self.assertTrue(result.architect_draft)
        self.assertGreater(result.token_count, 0)
        # Stubbed critic returns overall=7 → normalized 0.7
        self.assertAlmostEqual(result.critic_score, 0.7, places=2)
        self.assertTrue(result.final_output)
        self.assertGreater(result.elapsed_seconds, 0)

    def test_overload_triggers_retry_with_backoff(self):
        """B4C: Anthropic overload (HTTP 529 / InternalServerError) retries
        with exponential backoff, then returns the successful capture.

        Other Anthropic error classes (auth, bad-request) do NOT retry —
        they pass through to the per-entry skip path on the first attempt.
        """
        import anthropic
        import httpx
        entry = {
            "id": "eval_overload",
            "task_category": "code_generation",
            "target_model_family": "claude",
            "expected_path": "one_shot",
            "user_prompt": "Write a Python add function.",
            "pre_baked_context": "Use type hints.",
            "notes": "overload-retry",
        }

        # Build a real InternalServerError instance (the SDK's 5xx class,
        # which is what surfaces a 529 overload).
        fake_response = httpx.Response(
            status_code=529,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )
        overload_exc = anthropic.InternalServerError(
            message="overloaded_error",
            response=fake_response,
            body={"error": {"type": "overloaded_error"}},
        )

        call_log = []

        def flaky_pipeline(user_prompt, pre_baked, target_model):
            call_log.append("attempt")
            if len(call_log) < 3:
                raise overload_exc
            return run_eval.CaptureResult(
                architect_draft="ok",
                final_output="done",
                critic_score=0.5,
                critic_feedback="fine",
            )

        sleep_log: list[float] = []

        with patch("eval.run_eval._drive_pipeline", side_effect=flaky_pipeline):
            result = run_eval.run_entry(
                entry,
                sleep_fn=sleep_log.append,
                backoffs=(1, 2, 4),
            )

        self.assertFalse(result.skipped, msg=f"unexpected skip: {result.skip_reason}")
        self.assertIsNone(result.error)
        self.assertEqual(result.architect_draft, "ok")
        self.assertEqual(len(call_log), 3)  # 2 overloads + 1 success
        self.assertEqual(sleep_log, [1, 2])  # slept before retries 2 and 3
        self.assertEqual(result.extra.get("overload_retries"), 2)

    def test_overload_exhausted_skips_with_clear_reason(self):
        """All overload retries fail → skip with a reason that names the cause."""
        import anthropic
        import httpx
        entry = {
            "id": "eval_overload_fail",
            "task_category": "code_generation",
            "target_model_family": "claude",
            "expected_path": "one_shot",
            "user_prompt": "x",
            "pre_baked_context": "y",
        }
        fake_response = httpx.Response(
            status_code=529,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )
        overload_exc = anthropic.InternalServerError(
            message="overloaded_error",
            response=fake_response,
            body=None,
        )
        with patch("eval.run_eval._drive_pipeline", side_effect=overload_exc):
            result = run_eval.run_entry(
                entry,
                sleep_fn=lambda _: None,
                backoffs=(0, 0, 0),
            )
        self.assertTrue(result.skipped)
        self.assertIn("overload", result.skip_reason.lower())
        self.assertIn("InternalServerError", result.error)

    def test_non_overload_api_error_does_not_retry(self):
        """Auth/bad-request/etc. fail immediately on the first attempt."""
        entry = {
            "id": "eval_bad_request",
            "task_category": "code_generation",
            "target_model_family": "claude",
            "expected_path": "one_shot",
            "user_prompt": "x",
            "pre_baked_context": "y",
        }
        call_count = {"n": 0}

        def explode(user_prompt, pre_baked, target_model):
            call_count["n"] += 1
            raise ValueError("synthetic non-retryable failure")

        with patch("eval.run_eval._drive_pipeline", side_effect=explode):
            result = run_eval.run_entry(
                entry,
                sleep_fn=lambda _: None,
                backoffs=(0, 0, 0),
            )
        self.assertTrue(result.skipped)
        self.assertEqual(call_count["n"], 1)
        self.assertIn("ValueError", result.error)

    def test_unknown_target_model_family_is_logged_not_raised(self):
        entry = {
            "id": "eval_x",
            "task_category": "code_generation",
            "target_model_family": "not_a_family",
            "expected_path": "one_shot",
            "user_prompt": "do a thing",
            "pre_baked_context": "context",
        }
        with run_eval._dry_run_patch():
            result = run_eval.run_entry(entry)
        self.assertTrue(result.skipped)
        self.assertIn("pipeline error", result.skip_reason)
        self.assertIn("ValueError", result.error)


# ---------------------------------------------------------------------------
# Runner: result aggregation + file IO
# ---------------------------------------------------------------------------

class TestWriteRunFile(unittest.TestCase):

    def test_metadata_counts_captured_vs_skipped(self):
        results = [
            run_eval.build_entry_result(
                {"id": "a"}, run_eval.CaptureResult(architect_draft="x"),
            ),
            run_eval.build_entry_result(
                {"id": "b"},
                run_eval.CaptureResult(skipped=True, skip_reason="no context"),
            ),
        ]
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out = run_eval.write_run_file(
                results,
                runs_dir=tmp_path,
                label="testlabel",
                timestamp=datetime(2026, 5, 13, tzinfo=timezone.utc),
                commit_sha="abc1234",
                branch="claude/foo",
            )
            payload = json.loads(out.read_text())

        self.assertTrue(out.name.startswith("2026-05-13_abc1234_"))
        self.assertIn("testlabel", out.name)
        self.assertEqual(payload["metadata"]["entry_count"], 2)
        self.assertEqual(payload["metadata"]["captured_count"], 1)
        self.assertEqual(payload["metadata"]["skipped_count"], 1)
        self.assertEqual(len(payload["results"]), 2)
        # Skipped entry carries skip_reason
        self.assertEqual(
            payload["results"][1]["captured"]["skip_reason"], "no context"
        )
        # Rating block is null on capture
        self.assertIsNone(payload["results"][0]["rating"]["score"])


# ---------------------------------------------------------------------------
# Rate CLI: file-update logic
# ---------------------------------------------------------------------------

class TestApplyRating(unittest.TestCase):

    def _sample_payload(self):
        return {
            "metadata": {"entry_count": 2},
            "results": [
                {
                    "eval_id": "eval_001",
                    "input": {"id": "eval_001"},
                    "captured": {},
                    "rating": {"score": None, "notes": None, "rated_at": None},
                },
                {
                    "eval_id": "eval_002",
                    "input": {"id": "eval_002"},
                    "captured": {},
                    "rating": {"score": None, "notes": None, "rated_at": None},
                },
            ],
        }

    def test_apply_rating_updates_in_place_and_round_trips_through_disk(self):
        payload = self._sample_payload()
        rate.apply_rating(
            payload,
            "eval_002",
            score=4,
            notes="solid",
            now=datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc),
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            rate.save_run(path, payload)
            reloaded = rate.load_run(path)

        self.assertEqual(reloaded["results"][0]["rating"]["score"], None)
        self.assertEqual(reloaded["results"][1]["rating"]["score"], 4)
        self.assertEqual(reloaded["results"][1]["rating"]["notes"], "solid")
        self.assertEqual(
            reloaded["results"][1]["rating"]["rated_at"],
            "2026-05-13T12:00:00+00:00",
        )

    def test_apply_rating_unknown_eval_id_raises(self):
        payload = self._sample_payload()
        with self.assertRaises(KeyError):
            rate.apply_rating(payload, "eval_999", score=3, notes=None)

    def test_quit_mid_session_preserves_prior_ratings(self):
        payload = self._sample_payload()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            rate.save_run(path, payload)

            inputs = iter([
                "5", "great work",    # rate eval_001
                "q",                  # quit before rating eval_002
            ])
            outputs: list[str] = []
            rated = rate.rate_interactively(
                path,
                input_fn=lambda _: next(inputs),
                print_fn=outputs.append,
            )

            saved = rate.load_run(path)

        self.assertEqual(rated, 1)
        self.assertEqual(saved["results"][0]["rating"]["score"], 5)
        self.assertEqual(saved["results"][0]["rating"]["notes"], "great work")
        self.assertIsNone(saved["results"][1]["rating"]["score"])


# ---------------------------------------------------------------------------
# Rate CLI: rating-scale validation
# ---------------------------------------------------------------------------

class TestValidateScore(unittest.TestCase):

    def test_valid_scores(self):
        for s in (1, 2, 3, 4, 5):
            self.assertEqual(rate.validate_score(str(s)), s)
            self.assertEqual(rate.validate_score(f" {s} "), s)

    def test_rejects_out_of_range(self):
        for bad in ("0", "6", "-1", "100"):
            with self.assertRaises(ValueError):
                rate.validate_score(bad)

    def test_rejects_non_integer(self):
        for bad in ("", "  ", "abc", "3.5", "two"):
            with self.assertRaises(ValueError):
                rate.validate_score(bad)


# ---------------------------------------------------------------------------
# B4D — Structural guardrail gate in the eval harness
# ---------------------------------------------------------------------------

class TestStructuralGuardrailGate(unittest.TestCase):
    """B4D: _drive_pipeline must run check_structural_integrity on the assembled
    prompt after the Architect step and before running the Critic.  When defects
    are found the entry is skipped and run_critic_review is NOT called."""

    _ENTRY = {
        "id": "eval_structural",
        "task_category": "code_generation",
        "target_model_family": "claude",
        "expected_path": "one_shot",
        "user_prompt": "Write a sorting function.",
        "pre_baked_context": "Use Python, return a list.",
        "notes": "",
    }

    def test_structural_check_blocks_critic(self):
        """When check_structural_integrity returns errors, run_critic_review is
        NOT called, result is skipped, and extra['structural_defects'] is set."""
        defect_finding = {
            "check": "thinking_block",
            "message": "<thinking> found in prompt",
        }
        bad_struct = {
            "passed": False,
            "errors": ["<thinking> found in prompt"],
            "findings": [defect_finding],
        }

        with run_eval._dry_run_patch(), patch(
            "agents.guardrails.check_structural_integrity",
            return_value=bad_struct,
        ), patch(
            "orchestrator.agent_router.run_critic_review",
        ) as mock_critic:
            result = run_eval.run_entry(self._ENTRY)

        mock_critic.assert_not_called()
        self.assertTrue(result.skipped)
        self.assertIn("structural defects", result.skip_reason)
        self.assertIn("structural_defects", result.extra)
        self.assertEqual(result.extra["structural_defects"], [defect_finding])

    def test_clean_prompt_proceeds_to_critic(self):
        """When check_structural_integrity passes, run_critic_review IS called."""
        clean_struct = {"passed": True, "errors": [], "findings": []}

        with run_eval._dry_run_patch(), patch(
            "agents.guardrails.check_structural_integrity",
            return_value=clean_struct,
        ), patch(
            "orchestrator.agent_router.run_critic_review",
            return_value={
                "response": "ok",
                "scores": {"overall": 7},
                "suggestions": [],
                "reference_prompts": [],
            },
        ) as mock_critic:
            result = run_eval.run_entry(self._ENTRY)

        mock_critic.assert_called_once()
        self.assertFalse(result.skipped, msg=f"unexpected skip: {result.skip_reason}")


if __name__ == "__main__":
    unittest.main()

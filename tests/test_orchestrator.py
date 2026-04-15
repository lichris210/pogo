"""Unit tests for the POGO v2 orchestrator.

Tests session CRUD, task classification, state transitions, and response
merging.  All Bedrock / DynamoDB calls are mocked so the tests run offline.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# 1. Session tests
# ---------------------------------------------------------------------------

class TestSession(unittest.TestCase):
    """Session create / serialise / deserialise."""

    def test_create_session(self):
        from orchestrator.session import create_session, VALID_STATES
        s = create_session("user-1", "claude", "Analyze CSV data")
        self.assertTrue(len(s.session_id) > 0)
        self.assertEqual(s.user_id, "user-1")
        self.assertEqual(s.target_model, "claude")
        self.assertEqual(s.state, "initial")
        self.assertIn(s.state, VALID_STATES)
        self.assertTrue(s.created_at)
        self.assertTrue(s.updated_at)

    def test_roundtrip_serialisation(self):
        from orchestrator.session import create_session, Session
        s = create_session("user-2", "gpt", "Write a poem")
        s.current_draft = "You are a poet..."
        s.user_context = {"style": "haiku"}
        s.clarification_answers = {"q1": "5-7-5 syllables"}
        s.scores = {"overall": 8}
        s.add_message("user", "Hello")
        s.add_message("assistant", "Hi there")

        d = s.to_dict()
        # JSON-serialised fields are strings in the dict
        self.assertIsInstance(d["conversation_history"], str)
        self.assertIsInstance(d["user_context"], str)

        s2 = Session.from_dict(d)
        self.assertEqual(s2.session_id, s.session_id)
        self.assertEqual(s2.current_draft, "You are a poet...")
        self.assertEqual(s2.user_context, {"style": "haiku"})
        self.assertEqual(s2.scores, {"overall": 8})
        self.assertEqual(len(s2.conversation_history), 2)
        self.assertEqual(s2.conversation_history[0]["role"], "user")

    def test_add_message(self):
        from orchestrator.session import create_session
        s = create_session("u", "claude", "test")
        s.add_message("user", "hi")
        self.assertEqual(len(s.conversation_history), 1)
        self.assertIn("timestamp", s.conversation_history[0])

    @patch("orchestrator.session._get_table")
    def test_save_session(self, mock_table_fn):
        from orchestrator.session import create_session, save_session
        mock_table = MagicMock()
        mock_table_fn.return_value = mock_table

        s = create_session("user-3", "gemini", "task")
        save_session(s)

        mock_table.put_item.assert_called_once()
        item = mock_table.put_item.call_args[1]["Item"]
        self.assertEqual(item["session_id"], s.session_id)

    @patch("orchestrator.session._get_table")
    def test_load_session_found(self, mock_table_fn):
        from orchestrator.session import create_session, load_session, Session
        s = create_session("user-4", "claude", "task")
        mock_table = MagicMock()
        mock_table.get_item.return_value = {"Item": s.to_dict()}
        mock_table_fn.return_value = mock_table

        loaded = load_session(s.session_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.session_id, s.session_id)

    @patch("orchestrator.session._get_table")
    def test_load_session_not_found(self, mock_table_fn):
        from orchestrator.session import load_session
        mock_table = MagicMock()
        mock_table.get_item.return_value = {}
        mock_table_fn.return_value = mock_table

        loaded = load_session("nonexistent-id")
        self.assertIsNone(loaded)


# ---------------------------------------------------------------------------
# 2. Task classification tests
# ---------------------------------------------------------------------------

class TestClassifyTask(unittest.TestCase):

    def test_data_analysis(self):
        from orchestrator.agent_router import classify_task
        self.assertEqual(classify_task("Analyze customer churn data in CSV"), "data_analysis")

    def test_code_generation(self):
        from orchestrator.agent_router import classify_task
        self.assertEqual(classify_task("Write a Python function to sort a list"), "code_generation")

    def test_writing(self):
        from orchestrator.agent_router import classify_task
        self.assertEqual(classify_task("Write a blog article about AI"), "writing")

    def test_creative(self):
        from orchestrator.agent_router import classify_task
        self.assertEqual(classify_task("Write a short story about a robot"), "creative")

    def test_web_development(self):
        from orchestrator.agent_router import classify_task
        self.assertEqual(classify_task("Build a React landing page"), "web_development")

    def test_research(self):
        from orchestrator.agent_router import classify_task
        self.assertEqual(classify_task("Research and compare cloud providers"), "research")

    def test_general_fallback(self):
        from orchestrator.agent_router import classify_task
        self.assertEqual(classify_task("Do something interesting"), "general")


# ---------------------------------------------------------------------------
# 3. Response merger tests
# ---------------------------------------------------------------------------

class TestResponseMerger(unittest.TestCase):

    def test_merge_draft_scout_clarifier(self):
        from orchestrator.response_merger import merge_draft_scout_clarifier
        result = merge_draft_scout_clarifier(
            draft_response="```\nYou are a data analyst...\n```\n\n**Techniques Used:**\nCoT",
            scout_response="1. Schema\n2. Sample data",
            clarifier_response="1. What format?\n2. How many rows?",
        )
        self.assertIn("initial prompt draft", result)
        self.assertIn("data analyst", result)
        self.assertIn("what you could provide", result.lower())
        self.assertIn("Schema", result)
        self.assertIn("questions to sharpen", result.lower())
        self.assertIn("What format?", result)

    def test_merge_draft_no_code_fence(self):
        """Draft without a code fence should still be included."""
        from orchestrator.response_merger import merge_draft_scout_clarifier
        result = merge_draft_scout_clarifier(
            draft_response="Plain text draft",
            scout_response="context stuff",
            clarifier_response="questions here",
        )
        self.assertIn("Plain text draft", result)

    def test_merge_refinement_passed(self):
        from orchestrator.response_merger import merge_refinement
        result = merge_refinement(
            refined_prompt="```\nRefined prompt text\n```",
            fewshot_response="Example 1: ...\nExample 2: ...",
            guardrail_result={"passed": True, "warnings": [], "errors": []},
        )
        self.assertIn("Refined prompt text", result)
        self.assertIn("Few-shot examples", result)
        self.assertIn("Guardrails passed", result)

    def test_merge_refinement_warnings(self):
        from orchestrator.response_merger import merge_refinement
        result = merge_refinement(
            refined_prompt="```\nDraft\n```",
            fewshot_response="",
            guardrail_result={
                "passed": True,
                "warnings": ["Vague instruction detected"],
                "errors": [],
            },
        )
        self.assertIn("Guardrail warnings", result)
        self.assertIn("Vague instruction", result)

    def test_merge_refinement_errors(self):
        from orchestrator.response_merger import merge_refinement
        result = merge_refinement(
            refined_prompt="```\nDraft\n```",
            fewshot_response="",
            guardrail_result={
                "passed": False,
                "warnings": [],
                "errors": ["Prompt is empty."],
            },
        )
        self.assertIn("Guardrail errors", result)
        self.assertNotIn("Guardrails passed", result)

    def test_merge_review(self):
        from orchestrator.response_merger import merge_review
        scores = {
            "clarity": 8, "specificity": 7, "completeness": 9,
            "constraint_coverage": 6, "hallucination_risk": 3, "overall": 8,
        }
        result = merge_review(
            critic_response="Good prompt. Suggestion: add output format.",
            scores=scores,
            sample_output="Here is my analysis...",
        )
        self.assertIn("Prompt Evaluation", result)
        self.assertIn("8/10", result)
        self.assertIn("Sample output", result)
        self.assertIn("accept", result.lower())

    def test_format_accepted(self):
        from orchestrator.response_merger import format_accepted
        result = format_accepted("Final prompt text", ingested=True)
        self.assertIn("finalised", result)
        self.assertIn("Final prompt text", result)
        self.assertIn("reference library", result)

    def test_format_accepted_not_ingested(self):
        from orchestrator.response_merger import format_accepted
        result = format_accepted("Final prompt text", ingested=False)
        self.assertIn("finalised", result)
        self.assertNotIn("reference library", result)


# ---------------------------------------------------------------------------
# 4. State transition tests (orchestrator)
# ---------------------------------------------------------------------------

def _make_event(body: dict) -> dict:
    return {"body": json.dumps(body), "rawPath": "/optimize"}


def _mock_invoke_agent(agent_name, messages, system, **kwargs):
    """Return deterministic responses based on agent name."""
    responses = {
        "prompt_architect": (
            "```\nYou are a data analyst. Analyze the provided data.\n```\n\n"
            "**Techniques Used:**\nRole assignment, structured output"
        ),
        "context_scout": "1. Database schema\n   Why: Grounds the analysis.",
        "clarifier": "1. What depth of analysis?\n   (Determines output length.)",
        "fewshot_generator": "Example 1 — typical\nInput: data.csv\nOutput: summary",
        "critic": (
            '```json\n{"clarity": 8, "specificity": 7, "completeness": 9, '
            '"constraint_coverage": 6, "hallucination_risk": 3, "overall": 8, '
            '"techniques_identified": ["role_assignment"]}\n```\n'
            "Good prompt overall."
        ),
        "live_test": "Here is a sample analysis of the data...",
    }
    return responses.get(agent_name, "Mock response")


@patch("orchestrator.orchestrator.save_session")
@patch("orchestrator.orchestrator.load_session", return_value=None)
@patch("orchestrator.agent_router.invoke_agent", side_effect=_mock_invoke_agent)
class TestStateTransitions(unittest.TestCase):

    def test_initial_state(self, mock_invoke, mock_load, mock_save):
        from orchestrator.orchestrator import handle_message
        event = _make_event({
            "message": "Analyze customer churn data from CSV",
            "target_model": "claude",
        })
        resp = handle_message(event)
        body = json.loads(resp["body"])

        self.assertEqual(resp["statusCode"], 200)
        self.assertEqual(body["state"], "awaiting_context")
        self.assertIn("session_id", body)
        self.assertIn("initial prompt draft", body["message"])
        self.assertIn("what you could provide", body["message"].lower())
        self.assertTrue(len(body["prompt_draft"]) > 0)

    def test_awaiting_context_state(self, mock_invoke, mock_load, mock_save):
        from orchestrator.orchestrator import handle_message
        from orchestrator.session import create_session

        # Set up a session in awaiting_context state
        session = create_session("u", "claude", "Analyze data")
        session.state = "awaiting_context"
        session.task_category = "data_analysis"
        session.current_draft = "You are a data analyst..."
        mock_load.return_value = session

        event = _make_event({
            "session_id": session.session_id,
            "message": "I'm using PostgreSQL with a customers table",
            "target_model": "claude",
        })
        resp = handle_message(event)
        body = json.loads(resp["body"])

        self.assertEqual(resp["statusCode"], 200)
        self.assertEqual(body["state"], "review")
        self.assertIn("refined prompt", body["message"].lower())

    def test_review_state(self, mock_invoke, mock_load, mock_save):
        from orchestrator.orchestrator import handle_message
        from orchestrator.session import create_session

        session = create_session("u", "claude", "Analyze data")
        session.state = "review"
        session.task_category = "data_analysis"
        session.current_draft = "You are a data analyst..."
        mock_load.return_value = session

        event = _make_event({
            "session_id": session.session_id,
            "message": "evaluate",
            "target_model": "claude",
        })
        resp = handle_message(event)
        body = json.loads(resp["body"])

        self.assertEqual(resp["statusCode"], 200)
        self.assertEqual(body["state"], "iterating")
        self.assertIn("Prompt Evaluation", body["message"])
        self.assertIsInstance(body["scores"], dict)
        self.assertIn("clarity", body["scores"])

    def test_review_accept_shortcut(self, mock_invoke, mock_load, mock_save):
        from orchestrator.orchestrator import handle_message
        from orchestrator.session import create_session

        session = create_session("u", "claude", "Analyze data")
        session.state = "review"
        session.task_category = "data_analysis"
        session.current_draft = "Final prompt"
        mock_load.return_value = session

        event = _make_event({
            "session_id": session.session_id,
            "message": "accept",
            "target_model": "claude",
        })
        resp = handle_message(event)
        body = json.loads(resp["body"])

        self.assertEqual(body["state"], "accepted")
        self.assertIn("finalised", body["message"])

    def test_iterating_state(self, mock_invoke, mock_load, mock_save):
        from orchestrator.orchestrator import handle_message
        from orchestrator.session import create_session

        session = create_session("u", "claude", "Analyze data")
        session.state = "iterating"
        session.task_category = "data_analysis"
        session.current_draft = "Old draft"
        mock_load.return_value = session

        event = _make_event({
            "session_id": session.session_id,
            "message": "Make it more concise",
            "target_model": "claude",
        })
        resp = handle_message(event)
        body = json.loads(resp["body"])

        self.assertEqual(resp["statusCode"], 200)
        self.assertEqual(body["state"], "review")

    def test_iterating_accept(self, mock_invoke, mock_load, mock_save):
        from orchestrator.orchestrator import handle_message
        from orchestrator.session import create_session

        session = create_session("u", "gpt", "Write code")
        session.state = "iterating"
        session.task_category = "code_generation"
        session.current_draft = "You are a coder..."
        session.scores = {"overall": 9}
        mock_load.return_value = session

        event = _make_event({
            "session_id": session.session_id,
            "message": "accept",
            "target_model": "gpt",
        })
        resp = handle_message(event)
        body = json.loads(resp["body"])

        self.assertEqual(body["state"], "accepted")

    def test_error_missing_message(self, mock_invoke, mock_load, mock_save):
        from orchestrator.orchestrator import handle_message
        event = _make_event({"target_model": "claude"})
        resp = handle_message(event)
        self.assertEqual(resp["statusCode"], 400)

    def test_error_invalid_model(self, mock_invoke, mock_load, mock_save):
        from orchestrator.orchestrator import handle_message
        event = _make_event({"message": "hi", "target_model": "llama"})
        resp = handle_message(event)
        self.assertEqual(resp["statusCode"], 400)


# ---------------------------------------------------------------------------
# 5. invoke_parallel ordering test
# ---------------------------------------------------------------------------

class TestInvokeParallel(unittest.TestCase):

    @patch("orchestrator.agent_router.invoke_agent")
    def test_preserves_order(self, mock_invoke):
        from orchestrator.agent_router import invoke_parallel

        def _side_effect(agent_name, messages, system, **kw):
            return f"response_from_{agent_name}"

        mock_invoke.side_effect = _side_effect

        configs = [
            {"agent_name": "agent_a", "messages": [], "system": "sys_a"},
            {"agent_name": "agent_b", "messages": [], "system": "sys_b"},
            {"agent_name": "agent_c", "messages": [], "system": "sys_c"},
        ]
        results = invoke_parallel(configs)
        self.assertEqual(results, [
            "response_from_agent_a",
            "response_from_agent_b",
            "response_from_agent_c",
        ])


if __name__ == "__main__":
    unittest.main()

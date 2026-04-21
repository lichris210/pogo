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
        s.subcategory = "haiku"
        s.current_draft = "You are a poet..."
        s.fewshot_examples = "Example 1 — haiku"
        s.user_context = {"style": "haiku"}
        s.clarification_answers = {"q1": "5-7-5 syllables"}
        s.scores = {"overall": 8}
        s.ingested = True
        s.add_message("user", "Hello")
        s.add_message("assistant", "Hi there")

        d = s.to_dict()
        # JSON-serialised fields are strings in the dict
        self.assertIsInstance(d["conversation_history"], str)
        self.assertIsInstance(d["user_context"], str)

        s2 = Session.from_dict(d)
        self.assertEqual(s2.session_id, s.session_id)
        self.assertEqual(s2.subcategory, "haiku")
        self.assertEqual(s2.current_draft, "You are a poet...")
        self.assertEqual(s2.fewshot_examples, "Example 1 — haiku")
        self.assertEqual(s2.user_context, {"style": "haiku"})
        self.assertEqual(s2.scores, {"overall": 8})
        self.assertTrue(s2.ingested)
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
            prompt_format="xml",
        )
        self.assertIn("initial prompt draft", result["message"])
        self.assertIn("data analyst", result["message"])
        self.assertIn("what you could provide", result["message"].lower())
        self.assertIn("Schema", result["message"])
        self.assertIn("questions to sharpen", result["message"].lower())
        self.assertIn("What format?", result["message"])
        self.assertEqual(result["render_blocks"][1]["type"], "prompt_draft")
        self.assertEqual(result["render_blocks"][1]["format"], "xml")
        self.assertEqual(result["render_blocks"][2]["type"], "context_checklist")
        self.assertEqual(result["render_blocks"][3]["type"], "clarifier_questions")

    def test_merge_draft_no_code_fence(self):
        """Draft without a code fence should still be included."""
        from orchestrator.response_merger import merge_draft_scout_clarifier
        result = merge_draft_scout_clarifier(
            draft_response="Plain text draft",
            scout_response="context stuff",
            clarifier_response="questions here",
            prompt_format="markdown",
        )
        self.assertIn("Plain text draft", result["message"])
        self.assertEqual(result["render_blocks"][1]["prompt"], "Plain text draft")

    def test_merge_refinement_passed(self):
        from orchestrator.response_merger import merge_refinement
        result = merge_refinement(
            refined_prompt="```\nRefined prompt text\n```",
            fewshot_response="Example 1: ...\nExample 2: ...",
            guardrail_result={"passed": True, "warnings": [], "errors": []},
            prompt_format="markdown",
        )
        self.assertIn("Refined prompt text", result["message"])
        self.assertIn("Few-shot examples", result["message"])
        self.assertIn("Guardrails passed", result["message"])
        self.assertEqual(result["render_blocks"][0]["type"], "prompt_draft")
        self.assertEqual(result["render_blocks"][1]["type"], "fewshot_examples")
        self.assertEqual(result["render_blocks"][2]["type"], "text")

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
            prompt_format="xml",
        )
        self.assertIn("Guardrail warnings", result["message"])
        self.assertIn("Vague instruction", result["message"])
        self.assertEqual(result["render_blocks"][0]["type"], "guardrail_banner")
        self.assertEqual(result["render_blocks"][0]["severity"], "warning")
        self.assertEqual(result["render_blocks"][1]["type"], "prompt_draft")

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
            prompt_format="xml",
        )
        self.assertIn("Guardrail errors", result["message"])
        self.assertNotIn("Guardrails passed", result["message"])
        self.assertEqual(result["render_blocks"][0]["severity"], "error")

    def test_merge_review(self):
        from orchestrator.response_merger import merge_review
        scores = {
            "clarity": 8, "specificity": 7, "completeness": 9,
            "constraint_coverage": 6, "hallucination_risk": 3, "overall": 8,
        }
        result = merge_review(
            critic_response=(
                "Good prompt overall.\n\n"
                "Suggestions:\n"
                "1. Add an explicit output schema.\n"
                "2. Clarify the failure fallback."
            ),
            scores=scores,
            suggestions=[
                "Add an explicit output schema.",
                "Clarify the failure fallback.",
            ],
            sample_input="Analyze 3 customer cohorts and summarize churn drivers.",
            sample_output={
                "output": "Here is my analysis...",
                "latency_ms": 420,
                "tokens_used": 182,
            },
        )
        self.assertIn("Prompt Evaluation", result["message"])
        self.assertIn("8/10", result["message"])
        self.assertIn("Specific improvements", result["message"])
        self.assertIn("Sample input", result["message"])
        self.assertIn("Sample output", result["message"])
        self.assertIn("accept", result["message"].lower())
        self.assertEqual(result["render_blocks"][0]["type"], "scorecard")
        self.assertEqual(result["render_blocks"][1]["type"], "text")
        self.assertEqual(result["render_blocks"][2]["type"], "suggestions_list")
        self.assertEqual(result["render_blocks"][3]["type"], "sample_input")
        self.assertEqual(result["render_blocks"][4]["type"], "sample_output")

    def test_critic_parse_suggestions(self):
        from agents.critic import parse_suggestions

        response = (
            "```json\n{\"overall\": 8}\n```\n"
            "Clarity is strong.\n"
            "Improvement suggestions:\n"
            "1. Add a JSON schema for the final answer.\n"
            "2. Specify what to do when evidence is missing.\n"
            "3. Tighten the requested answer length."
        )

        self.assertEqual(parse_suggestions(response), [
            "Add a JSON schema for the final answer.",
            "Specify what to do when evidence is missing.",
            "Tighten the requested answer length.",
        ])

    def test_format_accepted(self):
        from orchestrator.response_merger import format_accepted
        result = format_accepted("Final prompt text", ingested=True, prompt_format="xml", threshold=0.8)
        self.assertIn("finalised", result["message"])
        self.assertIn("Final prompt text", result["message"])
        self.assertIn("reference library", result["message"])
        self.assertEqual(result["render_blocks"][1]["type"], "final_prompt")
        self.assertTrue(result["render_blocks"][1]["ingested"])

    def test_format_accepted_not_ingested(self):
        from orchestrator.response_merger import format_accepted
        result = format_accepted("Final prompt text", ingested=False, prompt_format="markdown", threshold=0.8)
        self.assertIn("finalised", result["message"])
        self.assertIn("0.80", result["message"])
        self.assertEqual(result["render_blocks"][1]["format"], "markdown")


class TestAcceptedPromptHelpers(unittest.TestCase):

    def test_build_prompt_record_from_session(self):
        from orchestrator.orchestrator import _build_prompt_record_from_session
        from orchestrator.session import create_session

        session = create_session("u", "claude", "Analyze data")
        session.task_category = "data_analysis"
        session.subcategory = "churn_prediction"
        session.current_draft = (
            "System: You are a data analyst who explains churn drivers.\n"
            "User: Analyze {{DATASET}} and return 3 findings."
        )
        session.fewshot_examples = (
            "Example 1 — typical\n"
            "Input: customers.csv\n"
            "Output: Three churn drivers"
        )
        session.scores = {"techniques_identified": ["role_assignment", "structured_output"]}

        record = _build_prompt_record_from_session(session, 0.83)

        self.assertEqual(record.task_category, "data_analysis")
        self.assertEqual(record.subcategory, "churn_prediction")
        self.assertEqual(record.target_model, "claude")
        self.assertEqual(record.format, "xml")
        self.assertEqual(record.system_prompt, "You are a data analyst who explains churn drivers.")
        self.assertEqual(record.user_prompt_template, "Analyze {{DATASET}} and return 3 findings.")
        self.assertEqual(record.few_shot_examples[0]["input"], "customers.csv")
        self.assertEqual(record.few_shot_examples[0]["output"], "Three churn drivers")
        self.assertEqual(record.quality_score, 0.83)


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
            "Good prompt overall.\n"
            "Suggestions:\n"
            "1. Add an explicit output schema.\n"
            "2. Clarify the failure fallback."
        ),
        "live_test": "Here is a sample analysis of the data...",
    }
    return responses.get(agent_name, "Mock response")


def _mock_live_test(prompt, target_model):
    return {
        "output": "Here is a sample analysis of the data...",
        "latency_ms": 325,
        "tokens_used": 144,
        "sample_input": "Analyze 3 customer cohorts and summarize churn drivers.",
    }


@patch("orchestrator.orchestrator.save_session")
@patch("orchestrator.orchestrator.load_session", return_value=None)
@patch("orchestrator.orchestrator.invoke_agent", new=_mock_invoke_agent)
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
        self.assertEqual(body["render_blocks"][1]["type"], "prompt_draft")
        self.assertEqual(body["render_blocks"][1]["format"], "xml")
        self.assertEqual(body["render_blocks"][2]["type"], "context_checklist")
        self.assertEqual(body["render_blocks"][3]["type"], "clarifier_questions")

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
        self.assertEqual(body["render_blocks"][0]["type"], "prompt_draft")
        self.assertEqual(body["render_blocks"][0]["format"], "xml")

    def test_review_state(self, mock_invoke, mock_load, mock_save):
        from orchestrator.orchestrator import handle_message
        from orchestrator.session import create_session

        session = create_session("u", "claude", "Analyze data")
        session.state = "review"
        session.task_category = "data_analysis"
        session.current_draft = "You are a data analyst..."
        session.fewshot_examples = "Example 1 — typical\nInput: dataset\nOutput: summary"
        mock_load.return_value = session

        event = _make_event({
            "session_id": session.session_id,
            "message": "evaluate",
            "target_model": "claude",
        })
        with patch("orchestrator.orchestrator.run_live_test", side_effect=_mock_live_test) as mock_live_test:
            resp = handle_message(event)
        body = json.loads(resp["body"])

        self.assertEqual(resp["statusCode"], 200)
        self.assertEqual(body["state"], "iterating")
        self.assertIn("Prompt Evaluation", body["message"])
        self.assertIsInstance(body["scores"], dict)
        self.assertIn("clarity", body["scores"])
        self.assertEqual(body["suggestions"], [
            "Add an explicit output schema.",
            "Clarify the failure fallback.",
        ])
        self.assertEqual(
            body["sample_input"],
            "Analyze 3 customer cohorts and summarize churn drivers.",
        )
        self.assertEqual(body["sample_output"]["output"], "Here is a sample analysis of the data...")
        self.assertEqual(body["render_blocks"][0]["type"], "scorecard")
        self.assertEqual(body["render_blocks"][2]["type"], "suggestions_list")
        self.assertEqual(body["render_blocks"][3]["type"], "sample_input")
        self.assertEqual(body["render_blocks"][4]["type"], "sample_output")
        mock_live_test.assert_called_once()

    def test_review_state_without_auto_live_test(self, mock_invoke, mock_load, mock_save):
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
        with patch("orchestrator.orchestrator._live_testing_enabled", return_value=False), \
                patch("orchestrator.orchestrator.run_live_test", side_effect=_mock_live_test) as mock_live_test:
            resp = handle_message(event)
        body = json.loads(resp["body"])

        self.assertEqual(body["state"], "iterating")
        self.assertFalse(body["live_testing_enabled"])
        self.assertIsNone(body["sample_input"])
        self.assertIsNone(body["sample_output"])
        self.assertEqual(body["render_blocks"][2]["type"], "suggestions_list")
        mock_live_test.assert_not_called()

    def test_iterating_can_trigger_live_test(self, mock_invoke, mock_load, mock_save):
        from orchestrator.orchestrator import handle_message
        from orchestrator.session import create_session

        session = create_session("u", "claude", "Analyze data")
        session.state = "iterating"
        session.task_category = "data_analysis"
        session.current_draft = "You are a data analyst..."
        mock_load.return_value = session

        event = _make_event({
            "session_id": session.session_id,
            "message": "Run a live test for this prompt.",
            "target_model": "claude",
            "run_live_test": True,
        })
        with patch("orchestrator.orchestrator._live_testing_enabled", return_value=False), \
                patch("orchestrator.orchestrator.run_live_test", side_effect=_mock_live_test) as mock_live_test:
            resp = handle_message(event)
        body = json.loads(resp["body"])

        self.assertEqual(body["state"], "iterating")
        self.assertEqual(body["sample_output"]["tokens_used"], 144)
        mock_live_test.assert_called_once()

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
        self.assertEqual(body["render_blocks"][1]["type"], "final_prompt")

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
        self.assertEqual(body["render_blocks"][0]["type"], "prompt_draft")
        self.assertEqual(body["render_blocks"][0]["format"], "xml")

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
        self.assertEqual(body["render_blocks"][1]["type"], "final_prompt")

    def test_iterating_accept_sets_ingested_flag_when_saved(self, mock_invoke, mock_load, mock_save):
        from orchestrator.orchestrator import handle_message
        from orchestrator.session import create_session

        session = create_session("u", "claude", "Analyze data")
        session.state = "iterating"
        session.task_category = "data_analysis"
        session.subcategory = "data_analysis"
        session.current_draft = "System: You are a data analyst.\nUser: Analyze {{DATASET}}."
        session.fewshot_examples = "Example 1 — typical\nInput: data.csv\nOutput: summary"
        session.scores = {"overall": 8, "techniques_identified": ["role_assignment"]}
        mock_load.return_value = session

        event = _make_event({
            "session_id": session.session_id,
            "message": "accept",
            "target_model": "claude",
        })
        with patch("orchestrator.orchestrator._ingest_accepted_prompt", return_value=True) as mock_ingest:
            resp = handle_message(event)
        body = json.loads(resp["body"])

        self.assertTrue(body["ingested"])
        self.assertIn("reference library", body["message"])
        mock_ingest.assert_called_once()

    def test_accept_below_threshold_stays_saved_but_not_ingested(self, mock_invoke, mock_load, mock_save):
        from orchestrator.orchestrator import handle_message
        from orchestrator.session import create_session

        session = create_session("u", "gpt", "Write code")
        session.state = "iterating"
        session.task_category = "code_generation"
        session.current_draft = "Final prompt"
        session.scores = {"overall": 7}
        mock_load.return_value = session

        event = _make_event({
            "session_id": session.session_id,
            "message": "accept",
            "target_model": "gpt",
        })
        with patch("orchestrator.orchestrator._ingest_accepted_prompt", return_value=True) as mock_ingest:
            resp = handle_message(event)
        body = json.loads(resp["body"])

        self.assertFalse(body["ingested"])
        self.assertIn("saved", body["message"].lower())
        self.assertIn("0.80", body["message"])
        mock_ingest.assert_not_called()

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

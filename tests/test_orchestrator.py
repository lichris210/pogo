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


class TestCoTPreambleStripping(unittest.TestCase):
    """Regression coverage for bugs #8 and #9.

    The Clarifier and Context Scout occasionally emit chain-of-thought or a
    conversational lead-in before their numbered list. The merger must strip
    that preamble before it reaches the user-facing message or render blocks.
    """

    # --- Direct helper coverage ------------------------------------------------

    def test_strip_preamble_removes_cot_lead_in(self):
        from orchestrator.response_merger import _strip_preamble
        text = (
            "Let me think about what needs clarification here...\n"
            "Here are some questions:\n"
            "1. What format should the output take?\n"
            "2. How long should the response be?"
        )
        out = _strip_preamble(text)
        self.assertTrue(out.startswith("1."))
        self.assertNotIn("Let me think", out)
        self.assertNotIn("Here are some questions", out)
        self.assertIn("What format", out)

    def test_strip_preamble_no_list_returns_unchanged(self):
        from orchestrator.response_merger import _strip_preamble
        text = "No list here, just prose."
        self.assertEqual(_strip_preamble(text), text)

    def test_strip_preamble_already_clean(self):
        from orchestrator.response_merger import _strip_preamble
        text = "1. First item\n2. Second item"
        self.assertEqual(_strip_preamble(text), text)

    # --- End-to-end through merge_draft_scout_clarifier ------------------------

    def _draft(self) -> str:
        return "```\nYou are a data analyst...\n```\n\n**Techniques Used:**\nCoT"

    def test_clarifier_preamble_stripped_from_message(self):
        from orchestrator.response_merger import merge_draft_scout_clarifier
        clarifier_with_cot = (
            "Let me think about what needs clarification here. Looking at the "
            "draft, I see several ambiguities.\n\n"
            "1. What output format do you want?\n"
            "2. How many rows of data should be processed?\n"
            "3. Should errors be raised or logged?"
        )
        result = merge_draft_scout_clarifier(
            draft_response=self._draft(),
            scout_response="1. Schema\n2. Sample data",
            clarifier_response=clarifier_with_cot,
            prompt_format="xml",
        )
        self.assertNotIn("Let me think", result["message"])
        self.assertNotIn("ambiguities", result["message"])
        self.assertIn("What output format", result["message"])
        clarifier_block = result["render_blocks"][3]
        self.assertEqual(clarifier_block["type"], "clarifier_questions")
        for item in clarifier_block["items"]:
            self.assertNotIn("Let me think", item)
            self.assertNotIn("ambiguities", item)
        self.assertEqual(len(clarifier_block["items"]), 3)

    def test_context_scout_preamble_stripped_from_message(self):
        from orchestrator.response_merger import merge_draft_scout_clarifier
        scout_with_cot = (
            "To make this stronger, let me consider what context would help.\n"
            "Here's my analysis of what would improve the prompt:\n\n"
            "1. Database schema\n"
            "   Why: Grounds the analysis in real columns.\n"
            "2. Sample rows\n"
            "   Why: Anchors the model on representative data."
        )
        result = merge_draft_scout_clarifier(
            draft_response=self._draft(),
            scout_response=scout_with_cot,
            clarifier_response="1. What format?\n2. How many rows?",
            prompt_format="xml",
        )
        self.assertNotIn("To make this stronger, let me consider", result["message"])
        self.assertNotIn("Here's my analysis", result["message"])
        self.assertIn("Database schema", result["message"])
        scout_block = result["render_blocks"][2]
        self.assertEqual(scout_block["type"], "context_checklist")
        for item in scout_block["items"]:
            self.assertNotIn("let me consider", item.lower())
            self.assertNotIn("here's my analysis", item.lower())
        self.assertEqual(len(scout_block["items"]), 2)

    def test_both_agents_clean_when_no_preamble(self):
        """Preamble-free outputs must pass through unchanged."""
        from orchestrator.response_merger import merge_draft_scout_clarifier
        result = merge_draft_scout_clarifier(
            draft_response=self._draft(),
            scout_response="1. Schema\n2. Sample data\n3. Metrics definitions",
            clarifier_response="1. What format?\n2. How many rows?\n3. Edge cases?",
            prompt_format="xml",
        )
        self.assertEqual(len(result["render_blocks"][2]["items"]), 3)
        self.assertEqual(len(result["render_blocks"][3]["items"]), 3)
        self.assertIn("Schema", result["render_blocks"][2]["items"][0])
        self.assertIn("What format", result["render_blocks"][3]["items"][0])

    def test_clarifier_system_prompt_forbids_preamble(self):
        """Agent system prompt must explicitly forbid CoT preamble."""
        from agents.clarifier import SYSTEM_PROMPT
        lower = SYSTEM_PROMPT.lower()
        self.assertIn("no preamble", lower)
        self.assertIn("strict output rules", lower)

    def test_context_scout_system_prompt_forbids_preamble(self):
        from agents.context_scout import SYSTEM_PROMPT
        lower = SYSTEM_PROMPT.lower()
        self.assertIn("no preamble", lower)
        self.assertIn("strict output rules", lower)


class TestCriticParseScores(unittest.TestCase):
    """parse_scores has a JSON path, a regex fallback path, and a default.

    Covered by the orchestrator state tests via the JSON path only; this
    locks down the regex-fallback and missing-key default behaviour.
    """

    def test_parse_scores_json_block(self):
        from agents.critic import parse_scores
        response = (
            '```json\n{"clarity": 9, "specificity": 8, "completeness": 7, '
            '"constraint_coverage": 6, "hallucination_risk": 2, "overall": 8}\n```\n'
            "Solid prompt overall."
        )
        scores = parse_scores(response)
        self.assertEqual(scores["clarity"], 9)
        self.assertEqual(scores["overall"], 8)

    def test_parse_scores_regex_fallback_when_no_json(self):
        from agents.critic import parse_scores
        response = (
            "Clarity: 7\n"
            "Specificity: 8\n"
            "Completeness: 6\n"
            "Overall: 7\n"
        )
        scores = parse_scores(response)
        self.assertEqual(scores["clarity"], 7)
        self.assertEqual(scores["specificity"], 8)
        self.assertEqual(scores["overall"], 7)
        # Missing keys default to -1.
        self.assertEqual(scores["constraint_coverage"], -1)
        self.assertEqual(scores["hallucination_risk"], -1)

    def test_parse_scores_invalid_json_falls_back_to_regex(self):
        """Malformed JSON should not crash; regex pass picks up loose values."""
        from agents.critic import parse_scores
        response = (
            "```json\n{this is not valid json}\n```\n"
            "clarity: 5, specificity: 4, overall: 5"
        )
        scores = parse_scores(response)
        self.assertEqual(scores["clarity"], 5)
        self.assertEqual(scores["overall"], 5)

    def test_parse_scores_handles_legacy_techniques_field(self):
        """Older captures may include a ``techniques_identified`` field.

        B4B removed the field from the system-prompt schema. ``parse_scores``
        must tolerate its presence in legacy responses without crashing.
        Design choice: the field is silently dropped from the returned dict
        (not surfaced), since downstream consumers no longer read it.
        """
        from agents.critic import parse_scores
        response = (
            '```json\n{"clarity": 8, "specificity": 7, "completeness": 9, '
            '"constraint_coverage": 6, "hallucination_risk": 3, "overall": 8, '
            '"techniques_identified": ["role_assignment", "few_shot"]}\n```\n'
            "Legacy-format response from a captured run."
        )
        scores = parse_scores(response)
        self.assertEqual(scores["clarity"], 8)
        self.assertEqual(scores["overall"], 8)
        self.assertNotIn("techniques_identified", scores)


class TestCriticSystemPrompt(unittest.TestCase):
    """B4B contract: Critic system prompt no longer rewards self-advertised
    structure (techniques_identified removed), has an INPUT HYGIENE section
    that names pipeline-leakage defects, and a HARD RULE capping
    Completeness when required sections are missing."""

    def test_critic_system_prompt_omits_techniques_identified(self):
        from agents.critic import SYSTEM_PROMPT
        json_start = SYSTEM_PROMPT.find("```json")
        json_end = SYSTEM_PROMPT.find("```", json_start + len("```json"))
        self.assertGreater(json_start, -1, "JSON schema block missing")
        schema = SYSTEM_PROMPT[json_start:json_end]
        self.assertNotIn("techniques_identified", schema)

    def test_critic_system_prompt_has_input_hygiene_section(self):
        from agents.critic import SYSTEM_PROMPT
        self.assertIn("INPUT HYGIENE", SYSTEM_PROMPT)
        markers = [
            "chain-of-thought preamble",
            "<thinking>",
            "Techniques Used:",
            "raw input data",
        ]
        present = sum(1 for m in markers if m.lower() in SYSTEM_PROMPT.lower())
        self.assertGreaterEqual(present, 3, f"Only {present}/4 hygiene markers present")

    def test_critic_system_prompt_has_section_presence_hard_rule(self):
        from agents.critic import SYSTEM_PROMPT
        self.assertIn("HARD RULE", SYSTEM_PROMPT)
        self.assertIn("Completeness", SYSTEM_PROMPT)
        self.assertIn("3/10", SYSTEM_PROMPT)


class TestCriticReferenceFlag(unittest.TestCase):
    """B4B: ``ENABLE_CRITIC_REFERENCES`` gates ``fetch_reference_prompts``
    in the Critic call path. Default is off; flag-flipped tests cover the
    on-path."""

    def test_critic_references_disabled_by_default(self):
        from orchestrator import agent_router
        self.assertIs(agent_router.ENABLE_CRITIC_REFERENCES, False)

        critic_text = (
            '```json\n{"clarity": 5, "specificity": 5, "completeness": 5, '
            '"constraint_coverage": 5, "hallucination_risk": 5, "overall": 5}\n```'
        )
        with patch(
            "orchestrator.agent_router.fetch_reference_prompts",
            return_value=["[id=ref ...]"],
        ) as mock_refs, patch(
            "orchestrator.agent_router.invoke_agent",
            return_value=critic_text,
        ):
            from agents.format_profiles import FORMAT_PROFILES
            result = agent_router.run_critic_review(
                final_prompt="some prompt",
                task_category="code_generation",
                format_profile=FORMAT_PROFILES["claude"],
                target_model="claude",
            )

        mock_refs.assert_not_called()
        self.assertEqual(result["reference_prompts"], [])


# ---------------------------------------------------------------------------
# B4C — Bug #10 / #16 — preamble strip extended to Architect + Few-Shot
# ---------------------------------------------------------------------------

class TestExtendedPreambleStripping(unittest.TestCase):
    """B4C: ``_strip_preamble`` now covers Architect (#16) and Few-Shot (#10)
    in addition to the legacy Clarifier / Context Scout coverage from
    Phase A. The merger applies the dedicated strippers defensively before
    extracting the prompt body."""

    def test_strip_architect_preamble_before_code_fence(self):
        from orchestrator.response_merger import strip_architect_preamble
        text = (
            "I'll refine the prompt based on the new requirements. Here's an "
            "updated version:\n\n"
            "```\n"
            "<role>You are an analyst.</role>\n"
            "<context>Some context.</context>\n"
            "<task>Do the work.</task>\n"
            "<constraints>Stay concise.</constraints>\n"
            "```\n"
        )
        out = strip_architect_preamble(text)
        self.assertTrue(out.startswith("```"))
        self.assertNotIn("I'll refine the prompt", out)

    def test_strip_architect_preamble_before_xml_tag(self):
        from orchestrator.response_merger import strip_architect_preamble
        text = (
            "Let me think about the structure first...\n"
            "<role>You are an analyst.</role>\n"
            "<context>Some context.</context>\n"
        )
        out = strip_architect_preamble(text)
        self.assertTrue(out.startswith("<role>"))
        self.assertNotIn("Let me think", out)

    def test_strip_fewshot_preamble_before_example(self):
        from orchestrator.response_merger import strip_fewshot_preamble
        text = (
            "I'll generate three diverse examples covering the typical case, "
            "an edge case, and a boundary case.\n\n"
            "Example 1 — typical\n"
            "Input: foo\n"
            "Output: bar\n"
        )
        out = strip_fewshot_preamble(text)
        self.assertTrue(out.startswith("Example 1"))
        self.assertNotIn("I'll generate", out)

    def test_strip_fewshot_unchanged_when_no_marker(self):
        from orchestrator.response_merger import strip_fewshot_preamble
        text = "Just prose without any Example block."
        self.assertEqual(strip_fewshot_preamble(text), text)

    def test_merger_strips_architect_preamble_end_to_end(self):
        from orchestrator.response_merger import merge_draft_scout_clarifier
        draft_with_cot = (
            "I'll draft a prompt for this task. Here it is:\n\n"
            "```\n<role>analyst</role>\n```\n\n**Techniques Used:** CoT"
        )
        result = merge_draft_scout_clarifier(
            draft_response=draft_with_cot,
            scout_response="1. Schema",
            clarifier_response="1. Format?",
            prompt_format="xml",
        )
        self.assertNotIn("I'll draft a prompt", result["message"])
        prompt_block = result["render_blocks"][1]
        self.assertEqual(prompt_block["type"], "prompt_draft")
        self.assertNotIn("I'll draft a prompt", prompt_block["prompt"])

    def test_merger_strips_fewshot_preamble_end_to_end(self):
        from orchestrator.response_merger import merge_refinement
        refined = "```\n<role>r</role>\n<context>c</context>\n<task>t</task>\n<constraints>x</constraints>\n```"
        fewshot_with_cot = (
            "I'll create examples grounded in the schema.\n\n"
            "Example 1 — typical\nInput: x\nOutput: y"
        )
        result = merge_refinement(
            refined_prompt=refined,
            fewshot_response=fewshot_with_cot,
            guardrail_result={"passed": True, "warnings": [], "errors": []},
            prompt_format="xml",
        )
        self.assertNotIn("I'll create examples", result["message"])
        for block in result["render_blocks"]:
            if block.get("type") == "fewshot_examples":
                self.assertNotIn("I'll create examples", block["text"])

    def test_fewshot_system_prompt_forbids_preamble(self):
        from agents.fewshot_generator import SYSTEM_PROMPT
        lower = SYSTEM_PROMPT.lower()
        self.assertIn("strict output rules", lower)
        self.assertIn("no preamble", lower)
        self.assertIn("<thinking>", lower)

    def test_architect_system_prompt_forbids_preamble(self):
        from agents.prompt_architect import SYSTEM_PROMPT
        lower = SYSTEM_PROMPT.lower()
        self.assertIn("strict output rules", lower)
        self.assertIn("no preamble", lower)
        self.assertIn("<thinking>", lower)


# ---------------------------------------------------------------------------
# B4C — Bug #13 — Few-Shot placeholder enforcement (system-prompt contract)
# ---------------------------------------------------------------------------

class TestFewShotPlaceholderEnforcement(unittest.TestCase):
    """B4C: Few-Shot Generator system prompt must enforce placeholder tokens
    instead of inventing user-specific values. This is a contract test on the
    system prompt; behavioral validation requires a live model run."""

    def test_fewshot_system_prompt_has_placeholder_enforcement_section(self):
        from agents.fewshot_generator import SYSTEM_PROMPT
        self.assertIn("INPUT GROUNDING & PLACEHOLDERS", SYSTEM_PROMPT)

    def test_fewshot_system_prompt_lists_placeholder_tokens(self):
        from agents.fewshot_generator import SYSTEM_PROMPT
        # The prompt must teach the model to emit at least these placeholders.
        for token in ("{{USER_DATA}}", "{{FIELD_NAME}}", "{{COLUMN_NAME}}"):
            self.assertIn(token, SYSTEM_PROMPT)

    def test_fewshot_system_prompt_names_forbidden_hallucination_patterns(self):
        from agents.fewshot_generator import SYSTEM_PROMPT
        lower = SYSTEM_PROMPT.lower()
        self.assertIn("forbidden", lower)
        # Concrete callouts that proved problematic in B4A captures.
        self.assertIn("user_12345", SYSTEM_PROMPT)
        self.assertIn("employee_id", SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# B4C — Bug #12 — Architect output validation (retry-then-fail-loud)
# ---------------------------------------------------------------------------

class TestArchitectValidation(unittest.TestCase):
    """B4C: The orchestrator must detect Architect output missing required
    sections, retry the call once with a stricter addendum, and raise
    :class:`ArchitectOutputError` if the retry also fails. Silent drop
    (Bug #12) must NOT survive."""

    def test_missing_architect_sections_xml(self):
        from orchestrator.orchestrator import _missing_architect_sections
        text = "<role>r</role><task>t</task>"  # missing context and constraints
        missing = _missing_architect_sections(text, "claude")
        self.assertEqual(set(missing), {"context", "constraints"})

    def test_missing_architect_sections_markdown(self):
        from orchestrator.orchestrator import _missing_architect_sections
        text = "## Role\nrole\n## Task\ntask\n"
        missing = _missing_architect_sections(text, "gpt")
        self.assertEqual(set(missing), {"context", "constraints"})

    def test_missing_architect_sections_empty_text(self):
        from orchestrator.orchestrator import (
            _missing_architect_sections,
            REQUIRED_ARCHITECT_SECTIONS,
        )
        self.assertEqual(
            set(_missing_architect_sections("", "claude")),
            set(REQUIRED_ARCHITECT_SECTIONS),
        )

    def test_invoke_architect_retries_then_succeeds(self):
        from orchestrator import orchestrator as orch

        broken = "```\n<role>r</role>\n```\n**Techniques Used:**\nx"
        complete = (
            "```\n<role>r</role>\n<context>c</context>\n"
            "<task>t</task>\n<constraints>x</constraints>\n```"
        )
        with patch.object(
            orch, "invoke_agent", side_effect=[broken, complete],
        ) as mock_invoke:
            raw, body = orch._invoke_architect_with_validation(
                arch_msgs=[{"role": "user", "content": [{"type": "text", "text": "go"}]}],
                arch_sys="sys",
                target_model="claude",
            )
        self.assertEqual(mock_invoke.call_count, 2)
        self.assertIn("<context>", body)
        self.assertIn("<constraints>", body)
        # Retry must pass the addendum message, not just re-call.
        retry_msgs = mock_invoke.call_args_list[1].args[1]
        flat = json.dumps(retry_msgs)
        self.assertIn("missing required sections", flat)

    def test_invoke_architect_raises_when_retry_also_broken(self):
        from orchestrator import orchestrator as orch

        broken = "```\n<role>r</role>\n```\n**Techniques Used:**\nx"
        with patch.object(
            orch, "invoke_agent", side_effect=[broken, broken],
        ):
            with self.assertRaises(orch.ArchitectOutputError) as ctx:
                orch._invoke_architect_with_validation(
                    arch_msgs=[{"role": "user", "content": [{"type": "text", "text": "go"}]}],
                    arch_sys="sys",
                    target_model="claude",
                )
        self.assertIn("incomplete", str(ctx.exception).lower())
        self.assertIn("retry", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# B4C — Bug #11 — Few-Shot consumes refined Architect draft (not the stale one)
# ---------------------------------------------------------------------------

class TestFewShotReadsRefinedDraft(unittest.TestCase):
    """B4C: ``_handle_awaiting_context`` must run the Architect FIRST and feed
    the refined draft into the Few-Shot Generator. The previous parallel
    invocation gave Few-Shot the stale pre-refine draft (Bug #11)."""

    @patch("orchestrator.orchestrator.save_session")
    @patch("orchestrator.orchestrator.load_session")
    def test_fewshot_receives_refined_draft_not_original(self, mock_load, mock_save):
        from orchestrator.orchestrator import handle_message
        from orchestrator.session import create_session

        session = create_session("u", "claude", "Analyze data")
        session.state = "awaiting_context"
        session.task_category = "data_analysis"
        # The ORIGINAL (stale) draft has schema columns name/email/age.
        session.current_draft = (
            "<role>analyst</role>\n<context>name/email/age</context>\n"
            "<task>analyze</task>\n<constraints>none</constraints>"
        )
        mock_load.return_value = session

        # The Architect refines to use the corrected schema id/first_name/...
        refined_architect_response = (
            "```\n<role>analyst</role>\n"
            "<context>id, first_name, last_name, is_active</context>\n"
            "<task>analyze</task>\n<constraints>none</constraints>\n```\n"
            "**Techniques Used:** schema_grounding"
        )
        captured_fewshot_inputs: list[str] = []

        def fake_invoke(agent_name, messages, system, **kwargs):
            if agent_name == "prompt_architect":
                return refined_architect_response
            if agent_name == "fewshot_generator":
                # Capture the user-message text the Few-Shot saw.
                user_text = messages[0]["content"][0]["text"]
                captured_fewshot_inputs.append(user_text)
                return "Example 1 — typical\nInput: x\nOutput: y"
            if agent_name == "context_scout":
                return "1. Schema"
            if agent_name == "clarifier":
                return "1. Format?"
            return "stub"

        with patch(
            "orchestrator.orchestrator.invoke_agent", side_effect=fake_invoke,
        ), patch(
            "orchestrator.agent_router.invoke_agent", side_effect=fake_invoke,
        ):
            event = {
                "body": json.dumps({
                    "session_id": session.session_id,
                    "message": "Actually the schema is id/first_name/last_name/is_active",
                    "target_model": "claude",
                }),
                "rawPath": "/optimize",
            }
            resp = handle_message(event)

        self.assertEqual(resp["statusCode"], 200)
        self.assertEqual(len(captured_fewshot_inputs), 1)
        fs_input = captured_fewshot_inputs[0]
        # The Few-Shot must see the REFINED columns, not the stale ones.
        self.assertIn("id, first_name, last_name, is_active", fs_input)
        self.assertNotIn("name/email/age", fs_input)


# ---------------------------------------------------------------------------
# B4C — Encoding smoke test (eval harness handles Unicode round-trips)
# ---------------------------------------------------------------------------

class TestEncodingRoundTrip(unittest.TestCase):
    """B4C encoding audit: every file-IO site in eval/ uses encoding='utf-8'
    explicitly. Smoke-test that Unicode content round-trips cleanly through
    the run-file writer."""

    def test_unicode_round_trips_through_run_file(self):
        from datetime import datetime, timezone
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from eval import run_eval

        unicode_payload = "café — naïve — 日本語 — 🎉"
        cap = run_eval.CaptureResult(
            architect_draft=unicode_payload,
            critic_feedback=unicode_payload,
            final_output=unicode_payload,
        )
        results = [run_eval.build_entry_result({"id": "eval_u"}, cap)]
        with TemporaryDirectory() as tmp:
            out = run_eval.write_run_file(
                results,
                runs_dir=Path(tmp),
                label="unicode-smoke",
                timestamp=datetime(2026, 5, 18, tzinfo=timezone.utc),
                commit_sha="b4cunic1",
                branch="claude/b4c",
            )
            data = out.read_text(encoding="utf-8")
        self.assertIn("café", data)
        self.assertIn("日本語", data)
        self.assertIn("🎉", data)


class TestAgentRouterHelpers(unittest.TestCase):
    """Phase 6 introduced critic-with-references and live-test routing.

    Cover the helpers the orchestrator relies on for that wiring.
    """

    def test_resolve_target_model_id_defaults_to_architect(self):
        from orchestrator import agent_router
        # No env override → falls back to the architect model.
        with patch.dict(
            "os.environ",
            {k: "" for k in agent_router._TARGET_MODEL_ENV_KEYS.values()},
            clear=False,
        ):
            self.assertEqual(
                agent_router.resolve_target_model_id("claude"),
                agent_router.ARCHITECT_MODEL_ID,
            )

    def test_resolve_target_model_id_uses_env_override(self):
        from orchestrator import agent_router
        with patch.dict(
            "os.environ",
            {"POGO_TARGET_MODEL_ID_GPT": "openai.gpt-test"},
            clear=False,
        ):
            self.assertEqual(
                agent_router.resolve_target_model_id("gpt"),
                "openai.gpt-test",
            )

    def test_resolve_target_model_id_unknown_family_falls_back(self):
        from orchestrator import agent_router
        self.assertEqual(
            agent_router.resolve_target_model_id("llama"),
            agent_router.ARCHITECT_MODEL_ID,
        )

    def test_fetch_reference_prompts_degrades_on_retrieval_error(self):
        """Retrieval failures must return [] so the agent call still proceeds."""
        from orchestrator import agent_router
        with patch(
            "prompt_db.retrieve.retrieve_reference_prompts",
            side_effect=RuntimeError("vector store unavailable"),
        ):
            result = agent_router.fetch_reference_prompts("code_generation", "claude")
        self.assertEqual(result, [])

    def test_fetch_reference_prompts_formats_records(self):
        from orchestrator import agent_router
        from prompt_db.schema import PromptRecord

        record = PromptRecord(
            id="ref_1",
            task_category="code_generation",
            subcategory="api_endpoint",
            target_model="claude",
            format="xml",
            techniques=["role_assignment"],
            system_prompt="You are a developer.",
            user_prompt_template="Build {{X}}.",
            quality_score=0.91,
            source="curated",
        )
        with patch(
            "prompt_db.retrieve.retrieve_reference_prompts",
            return_value=[record],
        ):
            result = agent_router.fetch_reference_prompts("code_generation", "claude")

        self.assertEqual(len(result), 1)
        block = result[0]
        self.assertIn("id=ref_1", block)
        self.assertIn("category=code_generation/api_endpoint", block)
        self.assertIn("role_assignment", block)
        self.assertIn("You are a developer.", block)
        self.assertIn("Build {{X}}.", block)

    def test_run_critic_review_wires_critic_and_parses(self):
        """run_critic_review wires critic → parsed scores and suggestions.

        References are disabled by default in B4B (see
        ``ENABLE_CRITIC_REFERENCES``); this test flips the flag on to
        exercise the original wiring. Default-off behavior is covered in
        ``TestCriticReferenceFlag``.
        """
        from orchestrator import agent_router

        critic_text = (
            '```json\n{"clarity": 8, "specificity": 7, "completeness": 9, '
            '"constraint_coverage": 6, "hallucination_risk": 3, "overall": 8}\n```\n'
            "Good prompt overall.\n"
            "Suggestions:\n"
            "1. Add a JSON schema.\n"
            "2. Specify failure handling.\n"
        )
        with patch.object(agent_router, "ENABLE_CRITIC_REFERENCES", True), patch(
            "orchestrator.agent_router.fetch_reference_prompts",
            return_value=["[id=ref_1 ...]"],
        ) as mock_refs, patch(
            "orchestrator.agent_router.invoke_agent",
            return_value=critic_text,
        ) as mock_invoke:
            from agents.format_profiles import FORMAT_PROFILES
            result = agent_router.run_critic_review(
                final_prompt="You are a data analyst...",
                task_category="data_analysis",
                format_profile=FORMAT_PROFILES["claude"],
                target_model="claude",
            )

        mock_refs.assert_called_once_with("data_analysis", "claude", k=3)
        mock_invoke.assert_called_once()
        self.assertEqual(result["scores"]["overall"], 8)
        self.assertEqual(result["suggestions"], [
            "Add a JSON schema.",
            "Specify failure handling.",
        ])
        self.assertEqual(result["reference_prompts"], ["[id=ref_1 ...]"])


class TestSplitFinalDraft(unittest.TestCase):
    """``_split_final_draft`` underpins ingestion record construction.

    The orchestrator falls back to system-only when no marker is present;
    cover each supported marker plus the fallback.
    """

    def test_system_user_plain_marker(self):
        from orchestrator.orchestrator import _split_final_draft
        sys_p, user_p = _split_final_draft(
            "System: You are an analyst.\nUser: Analyze {{X}}."
        )
        self.assertEqual(sys_p, "You are an analyst.")
        self.assertEqual(user_p, "Analyze {{X}}.")

    def test_xml_markers(self):
        from orchestrator.orchestrator import _split_final_draft
        sys_p, user_p = _split_final_draft(
            "<system>You are a coder.</system><user_prompt>Write {{X}}.</user_prompt>"
        )
        self.assertEqual(sys_p, "You are a coder.")
        self.assertEqual(user_p, "Write {{X}}.")

    def test_no_marker_returns_whole_text_as_system(self):
        from orchestrator.orchestrator import _split_final_draft
        sys_p, user_p = _split_final_draft("Just a one-block prompt with no marker.")
        self.assertEqual(sys_p, "Just a one-block prompt with no marker.")
        self.assertEqual(user_p, "")

    def test_empty_input(self):
        from orchestrator.orchestrator import _split_final_draft
        self.assertEqual(_split_final_draft(""), ("", ""))
        self.assertEqual(_split_final_draft(None), ("", ""))


class TestParseFewshotExamples(unittest.TestCase):
    """``_parse_fewshot_examples`` feeds the ingested PromptRecord."""

    def test_parses_multiple_examples_with_input_output(self):
        from orchestrator.orchestrator import _parse_fewshot_examples
        text = (
            "Example 1 — typical\n"
            "Input: customers.csv\n"
            "Output: 3 churn drivers\n\n"
            "Example 2 — edge case\n"
            "Input: empty.csv\n"
            "Output: no signal, request more data"
        )
        examples = _parse_fewshot_examples(text)
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["input"], "customers.csv")
        self.assertEqual(examples[0]["output"], "3 churn drivers")
        self.assertEqual(examples[0]["label"], "typical")
        self.assertEqual(examples[1]["label"], "edge case")
        self.assertIn("no signal", examples[1]["output"])

    def test_empty_input_returns_empty_list(self):
        from orchestrator.orchestrator import _parse_fewshot_examples
        self.assertEqual(_parse_fewshot_examples(""), [])
        self.assertEqual(_parse_fewshot_examples("   \n\n  "), [])

    def test_block_without_input_output_kept_as_text(self):
        from orchestrator.orchestrator import _parse_fewshot_examples
        text = "Example 1 — illustrative\nFreeform prose without structured input/output."
        examples = _parse_fewshot_examples(text)
        self.assertEqual(len(examples), 1)
        self.assertIn("Freeform prose", examples[0]["text"])
        # The "Example N —" label header should be stripped from .text.
        self.assertNotIn("Example 1", examples[0]["text"])


class TestIngestAcceptedPrompt(unittest.TestCase):
    """``_ingest_accepted_prompt`` swallows ingestion errors so accept still wins."""

    def test_ingestion_failure_returns_false(self):
        from orchestrator.orchestrator import _ingest_accepted_prompt
        from orchestrator.session import create_session

        session = create_session("u", "claude", "Analyze data")
        session.task_category = "data_analysis"
        session.current_draft = "System: You are an analyst.\nUser: Analyze {{X}}."

        with patch(
            "prompt_db.ingest.ingest_single_prompt",
            side_effect=RuntimeError("DB unreachable"),
        ):
            ok = _ingest_accepted_prompt(session, 0.9)
        self.assertFalse(ok)

    def test_ingestion_success_returns_true(self):
        from orchestrator.orchestrator import _ingest_accepted_prompt
        from orchestrator.session import create_session

        session = create_session("u", "claude", "Analyze data")
        session.task_category = "data_analysis"
        session.current_draft = "System: You are an analyst.\nUser: Analyze {{X}}."

        with patch(
            "prompt_db.ingest.ingest_single_prompt",
            return_value=True,
        ) as mock_ingest:
            ok = _ingest_accepted_prompt(session, 0.9)

        self.assertTrue(ok)
        # The PromptRecord we passed should have come from session state.
        called_with = mock_ingest.call_args[0][0]
        self.assertEqual(called_with.task_category, "data_analysis")
        self.assertEqual(called_with.target_model, "claude")


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
            "```\n"
            "<role>You are a data analyst.</role>\n"
            "<context>The user supplied churn metrics.</context>\n"
            "<task>Analyze the provided data.</task>\n"
            "<constraints>Return at most 3 findings.</constraints>\n"
            "```\n\n"
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

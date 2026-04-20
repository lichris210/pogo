"""Tests for the guardrails module.

Covers every check (existing and new), backward compatibility of
check_prompt(), severity tagging, and the suggest_fixes() helper.
All tests are offline — no model calls.
"""

from __future__ import annotations

import unittest

from agents.guardrails import (
    MODEL_CONTEXT_LIMITS,
    _OUTPUT_BUFFER_TOKENS,
    _estimate_tokens,
    _has_duplicate_instructions,
    check_prompt,
    suggest_fixes,
)

# ---------------------------------------------------------------------------
# Helpers — reusable prompt fragments
# ---------------------------------------------------------------------------

# A "clean" prompt long enough to pass the too-short check, with role and
# output-format indicators so only the check under test fires.
_CLEAN = (
    "You are a senior backend engineer. "
    "Analyse the access-log CSV below and return a JSON object with keys: "
    "total_requests, error_rate, p99_latency."
)


def _checks(prompt: str, model: str = "claude") -> dict:
    return check_prompt(prompt, model)


def _finding_checks(prompt: str, model: str = "claude") -> list[str]:
    """Return the list of check identifiers from findings."""
    return [f["check"] for f in _checks(prompt, model)["findings"]]


# ===================================================================
# 1. Existing checks
# ===================================================================


class TestEmptyPrompt(unittest.TestCase):
    def test_empty_string(self):
        r = _checks("")
        self.assertFalse(r["passed"])
        self.assertIn("empty_prompt", _finding_checks(""))

    def test_whitespace_only(self):
        r = _checks("   \n\t  ")
        self.assertFalse(r["passed"])
        self.assertIn("empty_prompt", _finding_checks("   \n\t  "))

    def test_non_empty_does_not_trigger(self):
        self.assertNotIn("empty_prompt", _finding_checks(_CLEAN))


class TestTooShort(unittest.TestCase):
    def test_short_prompt_triggers(self):
        short = "Summarise this."  # 15 chars
        self.assertIn("too_short", _finding_checks(short))

    def test_adequate_length_does_not_trigger(self):
        self.assertNotIn("too_short", _finding_checks(_CLEAN))


class TestVagueLanguage(unittest.TestCase):
    def test_do_your_best(self):
        p = _CLEAN + " Do your best."
        self.assertIn("vague_instruction", _finding_checks(p))

    def test_be_creative(self):
        p = _CLEAN + " Be creative."
        self.assertIn("vague_instruction", _finding_checks(p))

    def test_be_creative_with_qualifier_does_not_trigger(self):
        p = _CLEAN + " Be creative within the existing brand guidelines."
        self.assertNotIn("vague_instruction", _finding_checks(p))

    def test_as_needed(self):
        p = _CLEAN + " Add examples as needed."
        self.assertIn("vague_instruction", _finding_checks(p))

    def test_feel_free(self):
        p = _CLEAN + " Feel free to add more."
        self.assertIn("vague_instruction", _finding_checks(p))

    def test_try_to(self):
        p = _CLEAN + " Try to keep it short."
        self.assertIn("vague_instruction", _finding_checks(p))

    def test_clean_prompt_no_vague(self):
        self.assertNotIn("vague_instruction", _finding_checks(_CLEAN))


class TestMissingOutputFormat(unittest.TestCase):
    def test_no_format_triggers(self):
        p = "You are a data analyst. Analyse the quarterly revenue numbers and explain the trend."
        self.assertIn("missing_output_format", _finding_checks(p))

    def test_json_keyword_does_not_trigger(self):
        p = "You are a data analyst. Return the result as JSON."
        self.assertNotIn("missing_output_format", _finding_checks(p))

    def test_respond_in_does_not_trigger(self):
        p = "You are a data analyst. Respond in a numbered list."
        self.assertNotIn("missing_output_format", _finding_checks(p))


class TestMissingRole(unittest.TestCase):
    def test_no_role_triggers(self):
        p = "Analyse the quarterly revenue numbers and return a JSON summary."
        self.assertIn("missing_role", _finding_checks(p))

    def test_you_are_does_not_trigger(self):
        p = "You are a data analyst. Analyse the quarterly revenue numbers."
        self.assertNotIn("missing_role", _finding_checks(p))

    def test_act_as_does_not_trigger(self):
        p = "Act as a data analyst. Analyse the quarterly revenue numbers."
        self.assertNotIn("missing_role", _finding_checks(p))


# ===================================================================
# 2. New checks (Phase 4A)
# ===================================================================


class TestContradictions(unittest.TestCase):
    def test_concise_vs_detailed(self):
        p = _CLEAN + " Be concise. Also provide detailed explanations."
        self.assertIn("contradiction", _finding_checks(p))

    def test_json_vs_markdown(self):
        p = _CLEAN + " Respond in JSON. Use markdown for headings."
        self.assertIn("contradiction", _finding_checks(p))

    def test_no_bullets_vs_list(self):
        p = _CLEAN + " Never use bullet points. List the items."
        self.assertIn("contradiction", _finding_checks(p))

    def test_no_examples_vs_provide_examples(self):
        p = _CLEAN + " Do not include examples. Provide examples of edge cases."
        self.assertIn("contradiction", _finding_checks(p))

    def test_keep_short_vs_comprehensive(self):
        p = _CLEAN + " Keep the answer short. Be comprehensive."
        self.assertIn("contradiction", _finding_checks(p))

    def test_no_formatting_vs_use_headers(self):
        p = _CLEAN + " Do not use any formatting. Use headers for each section."
        self.assertIn("contradiction", _finding_checks(p))

    def test_no_code_vs_write_function(self):
        p = _CLEAN + " No code allowed. Write a function that sorts the data."
        self.assertIn("contradiction", _finding_checks(p))

    def test_non_conflicting_does_not_trigger(self):
        p = _CLEAN + " Be concise. Use bullet points."
        self.assertNotIn("contradiction", _finding_checks(p))

    def test_single_side_does_not_trigger(self):
        p = _CLEAN + " Be concise."
        self.assertNotIn("contradiction", _finding_checks(p))


class TestTokenEstimation(unittest.TestCase):
    def test_estimate_tokens_basic(self):
        text = " ".join(["word"] * 100)
        self.assertEqual(_estimate_tokens(text), 130)

    def test_exceeds_gpt_but_not_claude(self):
        # 98,463 words → ~128,001 tokens — exceeds gpt (128k) but not claude (200k).
        huge = " ".join(["word"] * 98_463)
        r_gpt = _checks(huge, "gpt")
        r_claude = _checks(huge, "claude")
        self.assertIn("exceeds_context_window",
                       [f["check"] for f in r_gpt["findings"]])
        self.assertNotIn("exceeds_context_window",
                         [f["check"] for f in r_claude["findings"]])
        self.assertFalse(r_gpt["passed"])
        self.assertTrue(r_claude["passed"])

    def test_large_prompt_warning_threshold(self):
        # 48,463 words → ~63,001 tokens, just over 50% of gpt usable (126,000).
        big = " ".join(["word"] * 48_463)
        checks = _finding_checks(big, "gpt")
        self.assertIn("large_prompt", checks)
        self.assertNotIn("exceeds_context_window", checks)

    def test_normal_prompt_no_token_warning(self):
        self.assertNotIn("large_prompt", _finding_checks(_CLEAN, "claude"))
        self.assertNotIn("exceeds_context_window", _finding_checks(_CLEAN, "claude"))


class TestAmbiguousPronouns(unittest.TestCase):
    def test_it_should_triggers(self):
        p = _CLEAN + " It should be correct."
        self.assertIn("ambiguous_pronoun", _finding_checks(p))

    def test_this_must_triggers(self):
        p = _CLEAN + "\nThis must include all items."
        self.assertIn("ambiguous_pronoun", _finding_checks(p))

    def test_that_will_triggers(self):
        p = _CLEAN + ". That will be validated later."
        self.assertIn("ambiguous_pronoun", _finding_checks(p))

    def test_they_should_triggers(self):
        p = _CLEAN + "\nThey should follow the rules."
        self.assertIn("ambiguous_pronoun", _finding_checks(p))

    def test_explicit_subject_does_not_trigger(self):
        p = (
            "You are a data analyst. Analyse the CSV and return JSON. "
            "The output should be correct."
        )
        self.assertNotIn("ambiguous_pronoun", _finding_checks(p))

    def test_pronoun_mid_sentence_does_not_trigger(self):
        p = _CLEAN + " Make sure that it handles edge cases."
        self.assertNotIn("ambiguous_pronoun", _finding_checks(p))


class TestMissingConstraints(unittest.TestCase):
    def test_creative_without_constraints_triggers(self):
        p = "You are a poet. Write a poem about the ocean."
        self.assertIn("missing_constraints", _finding_checks(p))

    def test_generate_without_constraints_triggers(self):
        p = "You are a copywriter. Generate taglines for a coffee brand."
        self.assertIn("missing_constraints", _finding_checks(p))

    def test_creative_with_length_constraint_does_not_trigger(self):
        p = "You are a poet. Write a poem about the ocean, maximum 20 words."
        self.assertNotIn("missing_constraints", _finding_checks(p))

    def test_creative_with_format_constraint_does_not_trigger(self):
        p = "You are a copywriter. Draft an email in a formal tone for the CEO."
        self.assertNotIn("missing_constraints", _finding_checks(p))

    def test_non_creative_prompt_does_not_trigger(self):
        self.assertNotIn("missing_constraints", _finding_checks(_CLEAN))


class TestDuplicateInstructions(unittest.TestCase):
    def test_near_identical_sentences_trigger(self):
        p = (
            "You are a code reviewer. Return your feedback as JSON.\n"
            "Always check for proper error handling in the submitted code.\n"
            "Some unrelated middle text goes here for padding.\n"
            "Always check for proper error handling in the submitted code."
        )
        self.assertIn("duplicate_instruction", _finding_checks(p))

    def test_high_overlap_sentences_trigger(self):
        p = (
            "You are a code reviewer. Return your feedback as JSON.\n"
            "Ensure that all variables follow the camelCase naming convention.\n"
            "Some middle context.\n"
            "Ensure that all variables follow the camelCase naming standard."
        )
        self.assertIn("duplicate_instruction", _finding_checks(p))

    def test_distinct_sentences_do_not_trigger(self):
        p = (
            "You are a code reviewer. Return your feedback as JSON.\n"
            "Check for proper error handling in the submitted code.\n"
            "Verify that all public methods have docstrings attached."
        )
        self.assertNotIn("duplicate_instruction", _finding_checks(p))

    def test_short_repeated_phrases_do_not_trigger(self):
        p = _CLEAN + " Check errors. Check errors."
        self.assertNotIn("duplicate_instruction", _finding_checks(p))

    def test_has_duplicate_instructions_helper(self):
        self.assertTrue(_has_duplicate_instructions(
            "Always validate the user input before processing the request.\n"
            "Always validate the user input before processing the request."
        ))
        self.assertFalse(_has_duplicate_instructions(
            "Validate user input. Process the request."
        ))


# ===================================================================
# 3. suggest_fixes()
# ===================================================================


class TestSuggestFixes(unittest.TestCase):
    def test_returns_list_same_length_as_findings(self):
        r = _checks("")
        fixes = suggest_fixes(r["findings"], "")
        self.assertEqual(len(fixes), len(r["findings"]))

    def test_each_entry_is_a_nonempty_string(self):
        p = "Do your best. Feel free to be creative."
        r = _checks(p)
        fixes = suggest_fixes(r["findings"], p)
        for fix in fixes:
            self.assertIsInstance(fix, str)
            self.assertGreater(len(fix), 0)

    def test_vague_do_your_best_suggestion(self):
        p = _CLEAN + " Do your best."
        r = _checks(p)
        vague_findings = [f for f in r["findings"] if f["check"] == "vague_instruction"]
        fixes = suggest_fixes(vague_findings, p)
        self.assertTrue(any("do your best" in f.lower() for f in fixes))

    def test_vague_be_creative_suggestion(self):
        p = _CLEAN + " Be creative."
        r = _checks(p)
        vague_findings = [f for f in r["findings"] if f["check"] == "vague_instruction"]
        fixes = suggest_fixes(vague_findings, p)
        self.assertTrue(any("be creative" in f.lower() for f in fixes))

    def test_missing_role_suggestion(self):
        p = "Analyse the revenue data and return a JSON summary."
        r = _checks(p)
        role_findings = [f for f in r["findings"] if f["check"] == "missing_role"]
        fixes = suggest_fixes(role_findings, p)
        self.assertTrue(any("role" in f.lower() or "you are" in f.lower() for f in fixes))

    def test_missing_output_format_suggestion(self):
        p = "You are a data analyst. Explain the quarterly revenue trend."
        r = _checks(p)
        fmt_findings = [f for f in r["findings"] if f["check"] == "missing_output_format"]
        fixes = suggest_fixes(fmt_findings, p)
        self.assertTrue(any("format" in f.lower() for f in fixes))

    def test_contradiction_suggestion_quotes_phrases(self):
        p = _CLEAN + " Be concise. Provide detailed analysis."
        r = _checks(p)
        ctr_findings = [f for f in r["findings"] if f["check"] == "contradiction"]
        fixes = suggest_fixes(ctr_findings, p)
        self.assertTrue(any("conflicting" in f.lower() for f in fixes))

    def test_ambiguous_pronoun_suggestion(self):
        p = _CLEAN + " It should be accurate."
        r = _checks(p)
        pron_findings = [f for f in r["findings"] if f["check"] == "ambiguous_pronoun"]
        fixes = suggest_fixes(pron_findings, p)
        self.assertTrue(any("pronoun" in f.lower() or "refer" in f.lower() for f in fixes))

    def test_missing_constraints_suggestion(self):
        p = "You are a poet. Write a poem about the ocean."
        r = _checks(p)
        mc_findings = [f for f in r["findings"] if f["check"] == "missing_constraints"]
        fixes = suggest_fixes(mc_findings, p)
        self.assertTrue(any("constraint" in f.lower() for f in fixes))

    def test_duplicate_instruction_suggestion(self):
        p = (
            "You are a reviewer. Return your analysis as JSON.\n"
            "Always validate the user input before processing the data.\n"
            "Some filler.\n"
            "Always validate the user input before processing the data."
        )
        r = _checks(p)
        dup_findings = [f for f in r["findings"] if f["check"] == "duplicate_instruction"]
        fixes = suggest_fixes(dup_findings, p)
        self.assertTrue(any("merge" in f.lower() or "duplicate" in f.lower() for f in fixes))

    def test_empty_findings_returns_empty_list(self):
        self.assertEqual(suggest_fixes([], _CLEAN), [])

    def test_unknown_check_falls_back(self):
        fake = [{"severity": "info", "check": "unknown_xyz", "message": "Something odd."}]
        fixes = suggest_fixes(fake, _CLEAN)
        self.assertEqual(len(fixes), 1)
        self.assertIn("Something odd", fixes[0])


# ===================================================================
# 4. Backward compatibility
# ===================================================================


class TestBackwardCompatibility(unittest.TestCase):
    def test_return_keys(self):
        r = _checks(_CLEAN)
        self.assertIn("passed", r)
        self.assertIn("warnings", r)
        self.assertIn("errors", r)
        self.assertIn("findings", r)

    def test_passed_is_bool(self):
        self.assertIsInstance(_checks(_CLEAN)["passed"], bool)
        self.assertIsInstance(_checks("")["passed"], bool)

    def test_warnings_is_list_of_str(self):
        r = _checks(_CLEAN + " Do your best.")
        self.assertIsInstance(r["warnings"], list)
        for w in r["warnings"]:
            self.assertIsInstance(w, str)

    def test_errors_is_list_of_str(self):
        r = _checks("")
        self.assertIsInstance(r["errors"], list)
        for e in r["errors"]:
            self.assertIsInstance(e, str)

    def test_passed_true_means_no_errors(self):
        r = _checks(_CLEAN)
        if r["passed"]:
            self.assertEqual(r["errors"], [])

    def test_passed_false_means_has_errors(self):
        r = _checks("")
        self.assertFalse(r["passed"])
        self.assertGreater(len(r["errors"]), 0)

    def test_warnings_derived_from_findings(self):
        p = _CLEAN + " Do your best. Try to keep it short."
        r = _checks(p)
        warning_msgs = [f["message"] for f in r["findings"] if f["severity"] == "warning"]
        self.assertEqual(r["warnings"], warning_msgs)

    def test_errors_derived_from_findings(self):
        r = _checks("")
        error_msgs = [f["message"] for f in r["findings"] if f["severity"] == "error"]
        self.assertEqual(r["errors"], error_msgs)


# ===================================================================
# 5. Severity levels
# ===================================================================


class TestSeverityLevels(unittest.TestCase):
    VALID_SEVERITIES = {"info", "warning", "error"}

    def test_all_findings_have_valid_severity(self):
        prompts = [
            "",
            "short",
            _CLEAN,
            _CLEAN + " Do your best. Be creative. It should work.",
            _CLEAN + " Be concise. Provide detailed analysis.",
        ]
        for p in prompts:
            for f in _checks(p)["findings"]:
                self.assertIn(f["severity"], self.VALID_SEVERITIES,
                              f"Bad severity in finding: {f}")

    def test_all_findings_have_required_keys(self):
        r = _checks(_CLEAN + " Do your best. It should work.")
        for f in r["findings"]:
            self.assertIn("severity", f)
            self.assertIn("check", f)
            self.assertIn("message", f)

    def test_empty_prompt_is_error(self):
        findings = _checks("")["findings"]
        self.assertEqual(findings[0]["severity"], "error")
        self.assertEqual(findings[0]["check"], "empty_prompt")

    def test_too_short_is_error(self):
        findings = _checks("Hi")["findings"]
        sev = {f["check"]: f["severity"] for f in findings}
        self.assertEqual(sev["too_short"], "error")

    def test_vague_instruction_is_warning(self):
        p = _CLEAN + " Do your best."
        findings = _checks(p)["findings"]
        vague = [f for f in findings if f["check"] == "vague_instruction"]
        self.assertTrue(vague)
        for f in vague:
            self.assertEqual(f["severity"], "warning")

    def test_missing_output_format_is_info(self):
        p = "You are a data analyst. Explain the quarterly revenue trend."
        findings = _checks(p)["findings"]
        fmt = [f for f in findings if f["check"] == "missing_output_format"]
        self.assertTrue(fmt)
        self.assertEqual(fmt[0]["severity"], "info")

    def test_missing_role_is_info(self):
        p = "Analyse the quarterly revenue data and return a JSON summary."
        findings = _checks(p)["findings"]
        role = [f for f in findings if f["check"] == "missing_role"]
        self.assertTrue(role)
        self.assertEqual(role[0]["severity"], "info")

    def test_contradiction_is_warning(self):
        p = _CLEAN + " Be concise. Provide detailed analysis."
        findings = _checks(p)["findings"]
        ctr = [f for f in findings if f["check"] == "contradiction"]
        self.assertTrue(ctr)
        self.assertEqual(ctr[0]["severity"], "warning")

    def test_ambiguous_pronoun_is_info(self):
        p = _CLEAN + " It should be fast."
        findings = _checks(p)["findings"]
        pron = [f for f in findings if f["check"] == "ambiguous_pronoun"]
        self.assertTrue(pron)
        self.assertEqual(pron[0]["severity"], "info")

    def test_missing_constraints_is_info(self):
        p = "You are a poet. Write a poem about the ocean."
        findings = _checks(p)["findings"]
        mc = [f for f in findings if f["check"] == "missing_constraints"]
        self.assertTrue(mc)
        self.assertEqual(mc[0]["severity"], "info")

    def test_duplicate_instruction_is_warning(self):
        p = (
            "You are a reviewer. Return your analysis as JSON.\n"
            "Always validate the user input before processing the data.\n"
            "Some filler text.\n"
            "Always validate the user input before processing the data."
        )
        findings = _checks(p)["findings"]
        dup = [f for f in findings if f["check"] == "duplicate_instruction"]
        self.assertTrue(dup)
        self.assertEqual(dup[0]["severity"], "warning")

    def test_exceeds_context_window_is_error(self):
        huge = " ".join(["word"] * 98_463)
        findings = _checks(huge, "gpt")["findings"]
        ecw = [f for f in findings if f["check"] == "exceeds_context_window"]
        self.assertTrue(ecw)
        self.assertEqual(ecw[0]["severity"], "error")

    def test_large_prompt_is_warning(self):
        big = " ".join(["word"] * 48_463)
        findings = _checks(big, "gpt")["findings"]
        lp = [f for f in findings if f["check"] == "large_prompt"]
        self.assertTrue(lp)
        self.assertEqual(lp[0]["severity"], "warning")


if __name__ == "__main__":
    unittest.main()

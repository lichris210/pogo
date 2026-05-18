# POGO v2.1 Redesign — Implementation Plan

This is the living implementation plan for the v2.1 workflow redesign. Each redesign phase has its own Claude Code prompt block below. After completing a phase, update the **Status** checklist and **Changelog** so any agent picking up the next phase has full context.

## Reference docs

Read these before starting any redesign phase:

- `WORKFLOW_REDESIGN.md` — the target state for POGO v2.1. The final product we're working toward.
- `ARCHITECTURE.md` — current production architecture (v2). Accurate until Redesign Phase E lands.
- `prompt_engineering_principles.md` — foundational principles, still valid.
- `PLAN.md` — historical record of Phases 1–7 (the original build). Reference only; not modified by the redesign.

## Final product (TL;DR)

POGO v2.1 adds a path divergence to the pipeline. Prompts that benefit from sequential phases follow a **chained-generation path** with a Decomposer agent producing N copy-pasteable phase prompts (each with embedded test specs and a recommended model tier). Prompts that don't benefit follow the existing **one-shot path**. Context gathering expands with a renamed Research agent that finds similar projects from public sources (user-approved) and accepts reference uploads. Per-phase RAG retrieval and per-phase ingestion to the vector DB compound quality over time.

Full spec in `WORKFLOW_REDESIGN.md`.

## Status

- [x] Design doc committed (`WORKFLOW_REDESIGN.md`)
- [x] Implementation plan committed (`PLAN_REDESIGN.md`)
- [x] **Phase A.** Bug fixes (#1, #8, #9) — completed 2026-05-12, commit `7314966`
- [x] **Phase B1.** Test backfill for original Phases 6–7 — completed 2026-05-12, commit `5a573ce`, branch `claude/pogo-v2-phase-b1-k3205`
- [x] **Phase B2.** Build eval harness — harness built 2026-05-13, commit `4201783`, branch `claude/pogo-v2-phase-b2-2SIti`. Baseline capture deferred to post-B3 (AWS credentials gap in Claude Code sandbox).
- [x] **Phase B3.** Migrate Bedrock → Anthropic API + bump models to Sonnet 4.6 / Haiku 4.5 — completed 2026-05-13, commit `a704ae5`, branch `claude/migrate-anthropic-api-AYZnc` (consolidated onto `redesign/v2.1`). Smoke test ran 2026-05-14 (commit `391a012`). Full v2 baseline captured and rated 2026-05-14 (commit `1681158`).
- [ ] **Phase B4.** Baseline Restoration — prompt DB seeding, fix bugs #10–14, Critic decontamination + calibration test, B3 closure verification, re-baseline. **Next to draft.**
- [ ] **Phase C.** Research agent expansion (autonomous discovery, references, summarization, conditional triggering)
- [ ] **Phase D.** Decomposer agent + per-phase model recommendation + tier maps for each frontier family
- [ ] **Phase E.** Per-phase RAG retrieval + per-phase ingestion + format profile inner-only scoping + phase plan assembly
- [ ] **Phase F.** UI updates (manual overrides, per-phase model display, aggregate cost estimate)
- [ ] **Phase G.** E2E smoke test extensions covering both paths

## Changelog

*(Preserve Claude Code's exact wording from its in-flight updates; the entries below are summary reconstructions from Phase completion reports.)*

**2026-05-12 — Phase A complete (commit `7314966`)**

Fixed three bugs:
- Bug #1: `deploy.sh` line 248 changed from bare `sed -i` to cross-platform `sed -i.bak ... && rm` pattern.
- Bug #8 (Clarifier CoT leak) and Bug #9 (Context Scout CoT leak) fixed via defense in depth:
  - Added `=== STRICT OUTPUT RULES ===` section to `agents/clarifier.py` and `agents/context_scout.py` system prompts forbidding preamble.
  - Added `_strip_preamble()` helper in `orchestrator/response_merger.py`, applied to both responses in `merge_draft_scout_clarifier()`.
- Rejected JSON migration on the grounds that it would force coordinated schema changes downstream and Phase C is already renaming Context Scout.
- Test count: 138 → 145 (added 7 tests in `TestCoTPreambleStripping`).

**Notes for subsequent phases:**
- If Phase C migrates Context Scout to structured JSON output, drop the `_strip_preamble` call for that agent in the merger.
- Latent bug in `_extract_list_items()`: treats leading non-numbered lines as bullet items, which was rendering preamble as fake checklist items. The `_strip_preamble` fix neutralizes the symptom but the underlying bug remains. Worth folding into Phase E when the renderer is touched.

**2026-05-12 — Phase B1 complete (commit `5a573ce`, branch `claude/pogo-v2-phase-b1-k3205`)**

Backfilled unit test coverage for code introduced in original PLAN.md Phases 6 (Critic + Live Testing) and 7 (Prompt Ingestion Loop). All Bedrock and vector-store calls remain mocked so the suite runs offline.

Stage 1 inventory (module → gap → tests added):

- `orchestrator/live_test.py` → `_clean_generated_input`, `_fallback_test_input`, and the empty-generator-fallback branch of `run_live_test` were untested. **+7 tests** in `tests/test_live_test.py` (`TestCleanGeneratedInput`, `TestFallbackTestInput`, `TestRunLiveTestFallback`).
- `agents/critic.parse_scores` → only the JSON happy path was exercised indirectly through orchestrator state tests; the regex fallback and missing-key defaults were untested. **+3 tests** in `tests/test_orchestrator.py::TestCriticParseScores`.
- `orchestrator/agent_router.py` → `resolve_target_model_id`, `fetch_reference_prompts` (formatting + retrieval-error fallback), and `run_critic_review` (the end-to-end critic-with-references wiring introduced in Phase 6) had no direct tests. **+6 tests** in `tests/test_orchestrator.py::TestAgentRouterHelpers`.
- `orchestrator/orchestrator._split_final_draft` → only one marker style was covered via `_build_prompt_record_from_session`; XML markers, the no-marker fallback, and empty input were untested. **+4 tests** in `tests/test_orchestrator.py::TestSplitFinalDraft`.
- `orchestrator/orchestrator._parse_fewshot_examples` → multi-example parsing, empty input, and the "block without Input/Output" branch were untested. **+3 tests** in `tests/test_orchestrator.py::TestParseFewshotExamples`.
- `orchestrator/orchestrator._ingest_accepted_prompt` → the swallow-on-failure contract relied on by `_handle_accepted` was untested. **+2 tests** in `tests/test_orchestrator.py::TestIngestAcceptedPrompt`.
- `prompt_db/ingest.py` seed-normalisation helpers (`_normalise_target_model`, `_split_system_user`, `_seed_to_record`) → exercised indirectly via the seed-ingest integration test but with no unit-level assertions on mapping correctness. **+6 tests** in `tests/test_prompt_db.py::TestSeedNormalisation`.
- `prompt_db/admin.py` → `update_score` bounds-check and the "ID not found" branches of `remove_prompt`/`update_score` were untested. **+3 tests** in `tests/test_prompt_db.py::TestAdminValidation`.

Test count: 145 → **179** (added 34 tests). All tests pass.

Bugs surfaced during Stage 3: **none**. The new tests confirmed existing Phase 6–7 behaviour rather than uncovering regressions.

**2026-05-13 — Phase B2 complete (commit `4201783`, branch `claude/pogo-v2-phase-b2-2SIti`)**

Built the eval harness. Baseline capture deferred — Claude Code sandbox had no AWS credentials, so all 18 entries skipped with `NoCredentialsError`. Placeholder file at `eval/runs/2026-05-13_4201783_v2-baseline.json`. Real baseline captured after Phase B3 migrated off Bedrock (see 2026-05-14 entry below).

Artifacts:
- `eval/inputs.json` — 18 entries spanning all 11 task categories (code_generation ×3; analysis ×2; agentic_workflow ×2; creative_writing ×2; summarization ×2; reasoning ×2; data_transformation, classification, extraction, translation, multimodal ×1 each).
- Target families: claude ×7, gpt ×6, gemini ×5.
- Paths: one_shot ×11, chained ×7 (roughly 60/40; close to the 50/50 target).
- `eval/run_eval.py` (programmatic orchestrator driver)
- `eval/rate.py` (CLI rating tool)
- `eval/README.md` (workflow docs)
- Tests: B2 branch baseline was 145 → 155 (+10 in `tests/test_eval_harness.py`) because B2 branched before B1 landed. After branch consolidation in Phase B3 Stage 0, the combined suite is 145 + 34 (B1) + 10 (B2) = 189 tests.

**Orchestrator quirks discovered driving it programmatically:**

1. `_handle_initial` / `_handle_awaiting_context` are tightly coupled to the response merger and the Lambda response shape. The runner deliberately reproduces the agent-call sequence (architect draft → refine + few-shot in parallel → critic) directly via `agent_router`, rather than invoking the state-machine handlers. That decoupling means future state-machine changes (Phase D's Decomposer in particular) will need a corresponding runner update.
2. `agent_router.invoke_agent` strips the usage metadata that `invoke_agent_raw` returns. Token counting required monkey-patching `invoke_agent_raw` for the run's duration.
3. The keyword-based `classify_task` in `agent_router.py` only knows 6 buckets (data_analysis, code_generation, writing, creative, web_development, research, general) — it cannot produce the 11 canonical categories used in `seed_prompts.json` or in this eval set. Eval `task_category` is therefore metadata for human use; the orchestrator classifies independently. Phase C/D should reconcile this.
4. `fetch_reference_prompts` and `fetch_fewshot_examples` are called eagerly and silently swallow exceptions, but the prompt_db S3 fetch happens before that catch in some code paths and raises `NoCredentialsError` up the stack. The runner treats this as a per-entry skip, not a hard failure.
5. **Sandbox limitation:** the environment where Phase B2 was executed had no AWS credentials. As a result, the committed baseline run file had all 18 entries marked `skipped: NoCredentialsError`. Re-run produced the real baseline numbers on 2026-05-14 after B3 migrated off Bedrock.

**Branch divergence note:** Phases A, B1, B2 each landed on separate branches. Phase B3 Stage 0 consolidated all three into a single long-lived `redesign/v2.1` branch and reconciled this file.

**2026-05-13 — Phase B3 complete (commit `a704ae5`, branch `claude/migrate-anthropic-api-AYZnc`, branched from `redesign/v2.1`)**

Migrated all agent generation calls from AWS Bedrock to the Anthropic API directly and bumped the default agent model to Claude Haiku 4.5.

Files migrated:
- `orchestrator/anthropic_client.py` — **new**. Thin wrapper around `anthropic.Anthropic().messages.create()`. Reads `ANTHROPIC_API_KEY` from env, caches the SDK client, normalises response shape to `{"text", "usage": {input_tokens, output_tokens, total_tokens}}`. Raises `AnthropicConfigError` on missing key.
- `orchestrator/agent_router.py` — dropped `_get_bedrock()`/`boto3` import; `invoke_agent_raw` now delegates to `anthropic_client.create_message`. `ARCHITECT_MODEL_ID` default bumped from `us.anthropic.claude-3-5-haiku-20241022-v1:0` to `claude-haiku-4-5-20251001`.
- `lambda/handler.py` — `generate_prompt()` (v1 `/generate` path) routed through the new wrapper. `GEN_MODEL_ID` bumped to `claude-haiku-4-5-20251001`. `get_bedrock()` retained for Titan embeddings only.
- `requirements.txt` — added `anthropic`. Lambda packaging (`deploy.sh`) already runs `pip install -r requirements.txt -t /tmp/pogo-package/` so the SDK is bundled automatically.
- `deploy.sh` — fails fast if `ANTHROPIC_API_KEY` is not exported in the caller's shell; after `update-function-code` runs `update-function-configuration --environment "Variables={ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY}"` to inject the key into the Lambda env. The key is **never** hardcoded in the script. Bedrock IAM permissions on the Lambda role are intentionally left in place (Titan embeddings still need them); future cleanup.
- `ARCHITECTURE.md` — §8 retitled "Model Calls (Anthropic API + Bedrock Titan)" with the wrapper documented, cost-profile note updated for Anthropic API list pricing, and a Local Development section noting the `ANTHROPIC_API_KEY` requirement.
- `eval/README.md` — cost-estimate updated to Anthropic API rates (~$1.20 per 18-entry run on Haiku 4.5; ~$0.20/entry on Sonnet 4.6 when target-family overrides route there). Setup section now says the runner needs `ANTHROPIC_API_KEY`, not Bedrock credentials.
- `eval/run_eval.py` — docstring/comment refresh (Bedrock → Anthropic API). No behavioural changes; the runner patches `agent_router.invoke_agent_raw` which now goes through the wrapper.
- `tests/test_eval_harness.py` — renamed `test_captures_pipeline_outputs_with_stubbed_bedrock` → `..._stubbed_anthropic`.
- `tests/test_live_test.py` — flipped a `bedrock timed out` error string to `anthropic call timed out` for accuracy.

Model ID changes:
- Default agent / Architect / Critic / Few-Shot / Clarifier / Context Scout / live test light model: `us.anthropic.claude-3-5-haiku-20241022-v1:0` → `claude-haiku-4-5-20251001`.
- No Sonnet model was wired into the codebase before B3, so the "Sonnet → `claude-sonnet-4-6`" half of the mapping is a no-op for now. Phase D will route specific phases to Sonnet 4.6 when the per-phase tier map lands.

Test count: 189 → **195** (added 6 tests in `tests/test_anthropic_client.py`: happy path, missing-env-var error, `anthropic.APIError` propagation, client caching, empty `content` fallback, and a sanity test that `agent_router.invoke_agent_raw` now routes through `create_message`). Mock pattern wraps the SDK at the module boundary (`sys.modules["anthropic"]`) so no live API access is needed.

Smoke test in sandbox: skipped. `ANTHROPIC_API_KEY` was not present in the Claude Code sandbox. Local smoke test ran 2026-05-14 (see entry below).

**Bedrock cleanups deferred to a later phase:**
- `prompt_db/embeddings.py` and `lambda/handler.py:embed_query()` still call Bedrock Titan for embeddings. Anthropic does not currently expose a comparable embedding model in the Python SDK; revisit if/when one ships, or evaluate whether to switch to a different embedding provider.
- `pogo/scripts/build_index_titan.py` (offline indexer) is untouched — same Titan dependency, not Lambda runtime.
- `_normalise_usage` in `orchestrator/agent_router.py` is now defensive overlap with the wrapper's own normalisation. Could be pruned in Phase E once the agent code is touched again.
- `AmazonBedrockFullAccess` IAM policy attachment in `deploy.sh` is still in place because embeddings need it. Scope it to embeddings-only (or migrate embeddings off Bedrock) before doing the full Bedrock decommission.
- v1 `/generate` path still loads `boto3` at import time for the Titan client. Consider lazy-importing it inside `embed_query` only.

**2026-05-14 — Phase B3 local smoke test (commit `391a012`, branch `claude/migrate-anthropic-api-AYZnc`)**

Ran the smoke test locally with `ANTHROPIC_API_KEY` set. One eval entry completed end-to-end through the migrated Anthropic SDK path. Elapsed ~57s, ~16,640 tokens, critic score 0.70. Migration validated. Capture artifact at `eval/runs/2026-05-14_391a012_v2-baseline-anthropic.json` (1 entry, no rating).

Three pre-existing pipeline bugs surfaced by inspecting the captured smoke-test output. These are not migration bugs; they existed in production before B3 but had never been caught because no one was reading full captured outputs before. Details in the Known Issues section below.

Local dev requirement noted: `requirements.txt` was missing `boto3` (AWS Lambda runtime provides it but local dev needs explicit install). Fix committed during this session.

**2026-05-14 — v2 baseline captured and rated (commit `1681158`, branch `redesign/v2.1`)**

Full 18-entry v2 baseline run captured at `eval/runs/2026-05-14_1681158_v2-baseline.json`. All 18 entries completed successfully (zero skips). Rating session followed.

**Baseline numbers:**
- Mean human score: **2.28 / 5** (1–5 rating scale; 3 = baseline expectation)
- Score distribution: 1 ×3, 2 ×9, 3 ×4, 4 ×2, 5 ×0
- Clean rate (no edits needed): 6% (1 entry out of 18)
- 12 of 18 entries below baseline expectation (score < 3)

**Top failure modes (manual triage):**
- Truncation: 50% of evals (output cut off mid-section)
- Schema/example contradictions: 39% of evals
- Architect output missing entirely: 28% of evals

**Critic-vs-human delta:**
The Critic's bias has a specific shape, not uniform generosity. It agrees with humans on the worst outputs (eval_001/006/013: critic 0.1–0.3, human 1) and on the best (eval_004/016: critic 0.9, human 4). The miscalibration concentrates in the middle: when output has any structure at all, the Critic rates 0.7–0.9 regardless of usability. 5 of 18 entries (28%) cross the 0.8 auto-ingest threshold while humans rate them 2 or below. The auto-ingest flywheel would poison the prompt DB if turned on against the current Critic.

Bugs surfaced or sharpened by the rating pass: bugs #11, #12, #13 (already known from smoke test), plus the new Critic miscalibration finding (logged as bug #14). Details in the Known Issues section below.

---

## Known Issues (to address in Phase B4)

These bugs and operational gaps surfaced during the 2026-05-14 smoke test and baseline rating. They are not blockers for Phase B3 completion (the migration shipped cleanly), but they make the v2 baseline an unreliable measurement and would poison the auto-ingest flywheel if it were turned on. Phase B4 owns the fixes.

**Bug #10 — Few-Shot Generator CoT preamble leak.**
- *Symptom:* Few-Shot Generator output contains a `<thinking>...</thinking>` block exposing chain-of-thought reasoning.
- *Same shape as bugs #8/#9* fixed in Phase A for Clarifier and Context Scout. Phase A's `_strip_preamble` helper in `orchestrator/response_merger.py` does not cover the Few-Shot Generator.
- *Fix:* extend `_strip_preamble` coverage to Few-Shot Generator output. Add a "STRICT OUTPUT RULES" section to the Few-Shot Generator system prompt forbidding preamble, matching the Phase A pattern.
- *Severity:* high. Visible in user-facing output and contributes to bug #14 (Critic miscalibration).

**Bug #11 — Few-Shot Generator uses stale Architect draft.**
- *Symptom:* When the Architect refines its draft based on Critic feedback, the Few-Shot Generator continues to use the original (unrefined) draft.
- *Concrete example:* In eval_001, the Architect's initial draft hallucinated schema fields (`id`/`first_name`/`last_name`/`is_active`). The Critic flagged the mismatch. The Architect's refined output correctly used the schema the user provided (`name`/`email`/`age`). The Few-Shot Generator's three examples still used the wrong (hallucinated) schema, producing an internally-contradictory final output.
- *Likely cause:* The Few-Shot Generator is invoked in parallel with Architect refinement and reads from a stale state field rather than the refined draft.
- *Severity:* high. Produces user-facing output that contradicts itself.

**Bug #12 — Architect output drops entirely from final output.**
- *Symptom:* Final output contains only Few-Shot examples plus a stray curl command, with no `<role>`, `<context>`, `<task>`, or `<constraints>` sections.
- *Reproducible:* eval_001 in the v2-baseline run (critic score 0.10, human score 1). The same input produced a complete output in the 2026-05-14 smoke test (critic 0.70), so this is intermittent.
- *Affects 28% of baseline entries* (5 of 18 entries had partial or missing Architect output).
- *Likely cause:* `_split_final_draft` parsing edge case when the Architect output is shaped unusually, or model nondeterminism producing unparseable structures. Root cause unknown.
- *Severity:* critical.

**Bug #13 — Architect / Few-Shot hallucinate user-specific data instead of using placeholders.**
- *Symptom:* Both agents invent user-specific values (e.g., specific user IDs, real-looking names, fabricated table schemas) rather than using placeholders the downstream user would substitute.
- *Likely related to:* the prompt DB never having been seeded (see "Operational gap" below). With no real exemplars to ground on, the model invents.
- *Severity:* medium-high. May partially resolve once the DB is seeded; verify before deeper fix.

**Bug #14 — Critic score miscalibration.**
- *Symptom:* Critic gives 0.7–0.9 scores to outputs humans rate 2/5.
- *Shape (from baseline rating):* agreement on the extremes (worst and best outputs), miscalibration in the middle. When output has any structure at all (including leaked CoT or self-contradictory examples), the Critic rates 0.7+.
- *Concrete cost:* 5 of 18 baseline entries (28%) cross the 0.8 auto-ingest threshold while humans rate them ≤ 2. Seeding the DB and enabling auto-ingest before fixing this would poison the DB.
- *Likely cause:* The Critic system prompt (`agents/critic.py`) asks for `techniques_identified` and instructs the model to "cite the exact part of the prompt that justifies each rating." Leaked CoT, "Techniques Used:" sections, and other meta-commentary from upstream agents are read as evidence of quality. The Critic has no defense against pipeline leakage.
- *Severity:* critical for the auto-ingest flywheel.

**Bug #15 — Few-Shot Generator retrieval blocked by empty `few_shot_examples` field in seed data.**
- *Symptom:* `retrieve_few_shot_examples` filters to records where `few_shot_examples` is non-empty. All 139 seed records have an empty `few_shot_examples` field (both in source `seed_prompts.json` and in `s3://pogo-knowledge-base/prompt_db/prompts.json`). Retrieval always returns 0.
- *Effect:* Few-Shot Generator falls back to its hardcoded template path regardless of whether the DB is seeded. Seeding does not improve the Few-Shot side of the pipeline.
- *Implication for Bug #13:* the "hallucinate user-specific data" symptom is not caused by an unseeded DB. Root cause is elsewhere — likely the Few-Shot Generator's prompt instructs the model to fabricate concrete values rather than emit placeholders. Address in the Few-Shot system prompt during Phase B4B, not via data fixes.
- *Possible fixes:* (a) populate `few_shot_examples` for each seed record (manual or semi-automated — substantial work, probably a Phase E task), (b) modify the retriever to synthesize example pairs from `user_prompt_template` plus a hypothetical output, or (c) repurpose `retrieve_reference_prompts` output as few-shot context.
- *Severity:* high. Invalidates one of B4A's premises.

**Bug #16 — Architect CoT preamble leak.**
- *Symptom:* Architect output begins with chain-of-thought meta-commentary (e.g., "I'll refine the prompt based on the new requirements. Here's an updated version:") before the structured `<role>`/`<context>`/etc. sections.
- *Same shape as bugs #8 (Clarifier), #9 (Context Scout), #10 (Few-Shot).* The Architect was not covered by Phase A's `_strip_preamble` helper.
- *Visible in:* `eval/runs/2026-05-18_3454e3d_b4a-critic-refs-only.json`, eval_007 first 200 chars.
- *Fix direction:* extend `_strip_preamble` coverage in `orchestrator/response_merger.py` to Architect output, and add a STRICT OUTPUT RULES section to the Architect system prompt forbidding preamble.
- *Severity:* high — contributes to Critic miscalibration since the preamble reads as "content" rather than a defect.

**Operational gap — Prompt DB never seeded.**
- *Symptom:* `bash scripts/seed_prompt_db.sh` has never been run.
- *Effect:* The Few-Shot Generator runs on hardcoded fallback templates rather than the 139 seeded exemplars. The Critic's `reference_prompts` field is empty, removing the comparison anchor it was designed to use. RAG retrieval returns nothing.
- *Implication:* the entire v2 baseline (mean 2.28) was measured on a degraded fallback path, not the architecture as designed. Some fraction of the failure modes above may resolve once the DB is seeded.
- *Fix:* run the seed script. First step of Phase B4.

**Eval harness vs. UI parity (verification needed).**
- *Question:* Do the eval harness captures match what `pogo.html` renders in the live UI?
- *Why it matters:* If the harness diverges from the UI, the 2.28 baseline rating is rating a different artifact than what users see. Some or all of the rating may need to be redone.
- *Verification step:* run eval_001 through the live UI, diff against the JSON capture. Listed as B3 closure step 4 in user notes but never formally verified.
- *Action:* fold into Phase B4 Stage 0.

---

## Phase B1 — Test Backfill for Original Phases 6–7

### Goal

The original PLAN.md's Phases 6 and 7 (built via Codex) shipped without unit test instructions, leaving a coverage gap in production code. Identify the gap, fill it with focused unit tests, and bring those modules up to the same coverage standard as Phases 1–5.

### Why this before the eval harness

The eval harness in Phase B2 will run the current pipeline end-to-end and capture outputs as a v2 baseline. If Phases 6–7 code has bugs that test backfill would have caught, the v2 baseline locks in those bugs as "expected behavior" and every subsequent redesign phase will look like a regression when it shouldn't. Fix what you have before measuring it.

### Claude Code prompt (paste below)

```
You are working on POGO v2.1. We are executing Phase B1 of the redesign.

Before doing anything, read these files in this order to load context:
1. WORKFLOW_REDESIGN.md — the target state for v2.1.
2. PLAN_REDESIGN.md — this implementation plan. Note the Changelog; Phase A is complete.
3. PLAN.md — the historical plan for the original build. Phases 6 and 7 are your scope here.
4. ARCHITECTURE.md — current production architecture.

Your task is to backfill unit test coverage for code introduced in Phases 6 and 7 of the original PLAN.md. Approach it in three stages.

==========
Stage 1 — Identify the coverage gap
==========

1. Read PLAN.md and list every feature, module, or code path introduced in Phases 6 and 7. Be concrete: file paths and function names where possible.
2. For each item from step 1, find the corresponding source code in the repo.
3. For each source module, check the existing tests/ directory for coverage. Identify which features have no tests, which have shallow coverage (only happy path), and which are well-covered.
4. Produce a short inventory in your response before you write any tests. Format: "Module → existing tests → gap." This is your work plan for Stage 2.

==========
Stage 2 — Write the tests
==========

Follow the existing test conventions (pytest, class-based organization, the patterns established in Phase A's TestCoTPreambleStripping). Prefer unit tests where the code is unit-testable. Use integration tests only where unit tests would be contrived.

For each gap from Stage 1:
- Cover the happy path
- Cover at least one error or edge case
- Cover any non-obvious behavior (defaults, fallbacks, retry logic, etc.)

Do not pad the test count. Five tests that cover real behavior beat fifteen that cover variations of the same thing.

Add tests to the existing test files where the file already covers the module. Create new test files only when no existing file fits.

==========
Stage 3 — Validate
==========

1. Run the full test suite. All 145 existing tests must still pass. Your new tests must also pass.
2. If any new test reveals a real bug in Phase 6–7 code (not a test setup issue), STOP. Do not fix the bug in this phase. Report it clearly so it can be triaged into the changelog as a known issue, and either fixed in Phase E or as a fast-follow Phase B1.5.
3. Note the new test count.

==========
When Phase B1 is complete
==========

1. Update PLAN_REDESIGN.md:
   - Check off Phase B1 in the Status checklist.
   - Add a Changelog entry with today's date, commit SHA, the Stage 1 inventory, the count of tests added per module, the new total test count, and any bugs surfaced during Stage 3.
2. Commit with the message: "Phase B1 complete: test backfill for original Phases 6–7"
3. Push.
4. Stop. Do not proceed to Phase B2 without explicit instruction.

In your final response to me, include:
- The Stage 1 inventory (module → gap)
- Tests added per module with brief description of what each covers
- Total test count before and after
- Any bugs surfaced (with enough detail to triage)
- Confirmation all tests pass
```

### Acceptance criteria for Phase B1

- Every meaningful behavior from original Phases 6–7 has at least one corresponding unit or integration test
- All existing tests still pass
- New tests pass
- Any bugs surfaced are documented in the Changelog, not silently fixed
- `PLAN_REDESIGN.md` updated with checked-off status and detailed Changelog entry

---

## Phase B2 — Build the Eval Harness (Manual Ratings, v2 Baseline)

### Goal

Build an eval harness that captures POGO's outputs across a fixed set of inputs and stores them for human rating. Run it against current production code to lock in a v2 baseline. Future redesign phases will run the same eval against their changes; comparing ratings reveals real quality improvements versus regressions.

### Why this before Phase C

Every redesign phase from C onward changes pipeline behavior (new Research agent scope, new Decomposer, new per-phase RAG). Without a baseline measurement, "did this help?" is unanswerable. The harness only earns its keep if the v2 baseline is captured before any behavior changes ship.

### Scope (Option A — manual ratings only)

This phase builds the minimum useful eval harness:
- A curated eval input set (~15–20 prompts)
- A runner script that drives POGO programmatically and captures full pipeline output
- A simple CLI rating tool
- A v2 baseline run with ratings captured

Out of scope (deferred for later phases): LLM-as-judge scoring, downstream verification (does the output actually produce working code), web UI for rating.

### Claude Code prompt (paste below)

```
You are working on POGO v2.1. We are executing Phase B2 of the redesign.

Before doing anything, read these files in this order to load context:
1. WORKFLOW_REDESIGN.md — the target state for v2.1.
2. PLAN_REDESIGN.md — this implementation plan. Note the Changelog; Phases A and B1 are complete.
3. ARCHITECTURE.md — current production architecture.
4. orchestrator/agent_router.py, orchestrator/orchestrator.py, and orchestrator/live_test.py — these are how you'll drive the pipeline programmatically.
5. seed_prompts.json — reference for the 11 task categories.

Your task is to build the eval harness in four stages. The harness is for manual ratings only in this phase; LLM-as-judge and downstream verification will be added later.

==========
Stage 1 — Curate the eval input set
==========

Build a fresh eval set at eval/inputs.json. Curate 15–20 prompts spanning the 11 task categories (analysis, agentic_workflow, code_generation, creative_writing, data_transformation, classification, extraction, summarization, reasoning, multimodal, translation).

Do NOT draw from seed_prompts.json. The seed prompts are training data for the vector DB; using them as eval inputs creates a measurement-contamination problem.

Each eval entry has this shape:

{
  "id": "eval_001",
  "task_category": "code_generation",
  "target_model_family": "claude",
  "expected_path": "one_shot" | "chained",
  "user_prompt": "the original prompt as a user would submit it",
  "pre_baked_context": "optional context to skip Clarifier interaction",
  "notes": "what makes this prompt interesting for eval"
}

Mix the set roughly 50/50 between one-shot-suitable and chain-suitable prompts. Spread target_model_family across claude/gpt/gemini. Include realistic prompts of varying complexity. Avoid trivial prompts and avoid prompts that overlap with seed_prompts.json content.

==========
Stage 2 — Build the eval runner
==========

Create eval/run_eval.py. The runner:

1. Loads eval/inputs.json.
2. For each entry, drives the POGO orchestrator programmatically (use the agent_router and orchestrator entry points; do not call the Lambda API). Use pre_baked_context to skip Clarifier interaction where possible. If a prompt cannot complete without Clarifier interaction, log it and skip that entry rather than failing the whole run.
3. Captures, per entry: the Architect draft, the Critic score and feedback, the final output, and the elapsed time and token count.
4. Writes results to eval/runs/<YYYY-MM-DD>_<commit_sha_short>_<branch>.json.

Output schema per entry:

{
  "eval_id": "eval_001",
  "input": { ...the original eval entry... },
  "captured": {
    "architect_draft": "...",
    "critic_score": 0.0,
    "critic_feedback": "...",
    "final_output": "...",
    "elapsed_seconds": 0.0,
    "token_count": 0
  },
  "rating": {
    "score": null,
    "notes": null,
    "rated_at": null
  }
}

The "rating" block stays null on first capture. The rating CLI in Stage 3 fills it in.

==========
Stage 3 — Build the rating CLI
==========

Create eval/rate.py. The rate CLI:

1. Takes a run file path as argument.
2. Walks through each entry where rating.score is null.
3. Prints the eval input, the captured output, and the Critic score.
4. Prompts for: score (1–5 integer) and optional notes.
5. Writes back to the same file with rating.score and rating.rated_at populated.
6. Allows quitting partway — partial ratings are preserved.

Rating scale documented in eval/README.md (Stage 4):
- 1: Output is unusable or wrong
- 2: Output is below baseline quality
- 3: Output is acceptable, equivalent to baseline expectation
- 4: Output is above baseline expectation
- 5: Output is significantly better than baseline expectation

==========
Stage 4 — Capture the v2 baseline + documentation
==========

1. Write eval/README.md explaining:
   - What the eval harness is for
   - How to add new eval entries to inputs.json
   - How to run a full eval (`python eval/run_eval.py`)
   - How to rate outputs (`python eval/rate.py eval/runs/<file>.json`)
   - Rating scale criteria
   - How to compare across runs (manual diff for now; tooling deferred to later phase)
   - Cost expectation per run (estimate based on Bedrock pricing × eval set size)

2. Run the eval against current production code: `python eval/run_eval.py`. This produces eval/runs/<today>_<sha>_v2-baseline.json.

3. Do NOT rate the baseline yourself. The user will rate it separately. Leave all rating blocks null.

4. Add eval/ to .gitignore for the runs/ subdirectory? No — runs ARE the historical record and should be committed. Only generated cache files should be gitignored if any are created.

==========
Validation
==========

- All 179 existing tests still pass.
- Add at least 3 new unit tests for the eval harness (eval/run_eval.py and eval/rate.py): one for the runner's per-entry capture logic, one for the rate CLI's file-update logic, one for the rating-scale validation.
- A baseline run file exists at eval/runs/<today>_<sha>_v2-baseline.json with all entries captured (or logged-and-skipped) and ratings null.

==========
When Phase B2 is complete
==========

1. Update PLAN_REDESIGN.md:
   - Check off Phase B2 in the Status checklist.
   - Add a Changelog entry with today's date, commit SHA, eval set size and category distribution, baseline run file path, number of entries successfully captured vs skipped, and any orchestrator quirks discovered when driving it programmatically.
2. Commit with the message: "Phase B2 complete: eval harness + v2 baseline captured"
3. Push.
4. Stop. Do not proceed to Phase C without explicit instruction.

In your final response to me, include:
- The eval set composition (counts per category, per model family, one_shot vs chained)
- Any prompts that had to be skipped because Clarifier interaction couldn't be bypassed
- Total cost of the baseline run (Bedrock token usage × rates)
- Confirmation all tests pass
- The baseline run file path
```

### Acceptance criteria for Phase B2

- `eval/inputs.json` exists with 15–20 curated entries spanning all 11 task categories
- `eval/run_eval.py` drives the orchestrator programmatically and captures outputs
- `eval/rate.py` provides a working CLI for adding manual ratings
- `eval/README.md` documents the workflow
- Baseline run captured at `eval/runs/<date>_<sha>_v2-baseline.json` with ratings null
- All existing tests pass; new harness tests pass
- `PLAN_REDESIGN.md` updated with checked-off status and detailed Changelog entry

### Status note (post-completion)

Harness built and tested cleanly. Baseline capture in the Claude Code sandbox failed due to missing AWS credentials; the resulting placeholder file at `eval/runs/2026-05-13_4201783_v2-baseline.json` has all 18 entries marked skipped. The real baseline was captured on 2026-05-14 after B3 migrated off Bedrock, at `eval/runs/2026-05-14_1681158_v2-baseline.json`. Both files are retained in the repo as historical record.

---

## Phase B3 — Migrate Bedrock → Anthropic API + bump models

### Goal

Replace AWS Bedrock SDK calls with the Anthropic Python SDK across all agent and orchestrator code. Bump internal agent models to current versions (`claude-sonnet-4-6` and `claude-haiku-4-5-20251001`). Keep all other AWS infrastructure (Lambda, API Gateway, DynamoDB, S3) unchanged.

### Why this before Phase C

Three reasons:
1. Every redesign phase from C onward adds new agent code. Writing it against the Anthropic SDK from the start is cheaper than porting it later.
2. The v2 baseline measurement must be captured on the stack that subsequent phases run on. Capturing on Bedrock then migrating would conflate "SDK changed" with "redesign changed."
3. Removes AWS-credentials friction from local dev and eval-harness runs.

### Claude Code prompt (paste below)

```
You are working on POGO v2.1. We are executing Phase B3 of the redesign: migrating from AWS Bedrock to the Anthropic API directly for internal agent calls, and bumping internal agent models to current versions.

Before doing anything, read these files in this order to load context:
1. WORKFLOW_REDESIGN.md — the target state for v2.1.
2. PLAN_REDESIGN.md — the implementation plan. Note the Changelog; Phases A, B1, B2 are complete.
3. ARCHITECTURE.md — current production architecture (will need updating).
4. The agent and orchestrator source files that make Bedrock calls.

==========
Stage 0 — Branch consolidation (do this FIRST)
==========

Phase A landed on `claude/pogo-v2-workflow-redesign-K5tfm`. Phase B1 landed on `claude/pogo-v2-phase-b1-k3205`. Phase B2 landed on `claude/pogo-v2-phase-b2-2SIti`. They are divergent branches.

Consolidate to a single long-lived feature branch `redesign/v2.1`:

1. Check out main (or whatever your trunk is).
2. Create `redesign/v2.1` from main.
3. Merge Phase A's branch into `redesign/v2.1`.
4. Merge Phase B1's branch in. Resolve any PLAN_REDESIGN.md conflicts by keeping ALL changelog entries in chronological order and ALL checkboxes ticked for completed phases.
5. Merge Phase B2's branch in. Same conflict resolution rule.
6. Run the full test suite. All 179 tests (B1's count) + 10 (B2's count) = 189 should pass. If anything fails, the merges introduced a real conflict — stop and report.
7. Push `redesign/v2.1`.

Branch from `redesign/v2.1` for Phase B3 work. Name it `claude/pogo-v2-phase-b3-<random>`.

==========
Stage 1 — Inventory + design
==========

1. Grep the codebase for every Bedrock call site. Look for: `bedrock`, `bedrock-runtime`, `invoke_model`, `boto3.client('bedrock`. Produce a list of files and call sites.
2. For each call site, note: which agent uses it, which Bedrock model ID is currently passed, what the system/user prompt structure looks like, and how the response is parsed.
3. Identify any shared wrapper or factory that all agents go through (likely in `agents/` or `orchestrator/`). If one exists, the migration centralizes there. If agents call Bedrock directly with no abstraction, this phase should ALSO add a thin client wrapper to centralize the new Anthropic calls.
4. Produce the inventory in your response before writing any code.

Model ID mapping (use these exact strings):
- Wherever a Sonnet model was used via Bedrock → `claude-sonnet-4-6`
- Wherever a Haiku model was used via Bedrock → `claude-haiku-4-5-20251001`
- Do not introduce Opus calls in this phase. Stick with what was there.

==========
Stage 2 — Migration
==========

1. Add `anthropic` to the Lambda dependencies. Lambda packaging in `deploy.sh` must include the new SDK.
2. Build (or extend, if one exists) a thin client wrapper that wraps `anthropic.Anthropic().messages.create(...)`. The wrapper should:
   - Read `ANTHROPIC_API_KEY` from env
   - Accept the same logical params every agent currently passes (model, messages, system prompt, max_tokens, etc.)
   - Return a normalized response object that has the same shape as what agents currently expect from the Bedrock-parsing code, so agent code doesn't have to change downstream of the call site
3. Replace every Bedrock call site to use the new wrapper.
4. Anthropic SDK differences to handle:
   - System prompts are top-level `system=` param, not in `messages` array (unlike some Bedrock patterns)
   - Response is `Message` object; text is `.content[0].text`
   - Token usage is `.usage.input_tokens` and `.usage.output_tokens`
   - Errors are `anthropic.APIError` subclasses, not `botocore.exceptions.ClientError`. Any retry/error handling needs updating.
5. Update Lambda environment configuration to inject `ANTHROPIC_API_KEY`. Use a Lambda env var directly (no Secrets Manager). Update `deploy.sh` to set the env var on `update-function-configuration`. The API key should NOT be hardcoded in the script — it should be read from the caller's local shell environment.
6. Leave existing Bedrock IAM permissions on the Lambda role in place for now. Do not remove them. (Future cleanup.)

==========
Stage 3 — Tests + docs + smoke test
==========

1. Update tests:
   - Replace any Bedrock mocks (likely `moto` or `botocore.stub`) with Anthropic SDK mocks. The cleanest pattern is to mock the wrapper from Stage 2, not the underlying SDK.
   - All 189 existing tests must still pass.
   - Add at least 4 new tests covering the new wrapper: happy path, API error, retry behavior (if any), env-var missing → clear error message.

2. Update documentation:
   - `ARCHITECTURE.md`: change "Bedrock (Claude Sonnet + Haiku)" to "Anthropic API (Claude Sonnet 4.6 + Haiku 4.5)" in the infrastructure section. Update the cost-profile note since per-call rates have changed.
   - Add a line in the Local Development section (create one if missing): "Set ANTHROPIC_API_KEY in your shell to run agents locally or run the eval harness without AWS model permissions."
   - `eval/README.md`: update the cost-estimate section with new per-session expectations.

3. Smoke test:
   - With your `ANTHROPIC_API_KEY` env var set in the sandbox (assume the user will provide it if not already available — if not, log clearly that the smoke test was skipped and explain how the user runs it manually after merging), run one happy-path session end-to-end against a simple prompt. Confirm the orchestrator produces a final accepted prompt.
   - If smoke test ran: note token counts and approximate cost.
   - If smoke test skipped: provide the exact command the user should run to validate.

==========
When Phase B3 is complete
==========

1. Update PLAN_REDESIGN.md:
   - Check off Phase B3 in the Status checklist.
   - Add a Changelog entry with today's date, commit SHA, the list of files migrated, the model ID changes, the new test count, smoke test outcome, and any unexpected behavior differences from the model bump.
2. Commit with the message: "Phase B3 complete: Bedrock → Anthropic API migration + model bumps"
3. Push.
4. Stop. Do not proceed to Phase C without explicit instruction.

In your final response to me, include:
- The Stage 1 inventory
- Confirmation that branch consolidation succeeded and all 189 prior tests pass on `redesign/v2.1` before B3 work began
- New test count after B3
- Smoke test outcome (or skip reason)
- Any model-output differences noticed during smoke test (Sonnet 4.6 may behave differently than the previous Bedrock model in subtle ways — flag anything that looks like a regression)
- Any Bedrock-specific code paths that should be cleaned up later (retry logic, error handling, model-ID utilities, etc.)
```

### Acceptance criteria for Phase B3

- `redesign/v2.1` exists with Phase A, B1, B2 merged in and 189 tests passing as a starting point
- No remaining Bedrock SDK calls in the codebase (grep returns nothing meaningful)
- Anthropic SDK is in deps and Lambda package
- All agent calls go through the new wrapper using `claude-sonnet-4-6` and `claude-haiku-4-5-20251001`
- `ANTHROPIC_API_KEY` is required env var; missing key produces a clear error
- All existing tests pass; new wrapper tests pass
- ARCHITECTURE.md and eval/README.md updated
- Smoke test produces a working session end-to-end (or skip reason clearly documented)
- `PLAN_REDESIGN.md` updated with checked-off status and detailed Changelog entry

---

## Phase B4 — Baseline Restoration (next to draft)

### Goal

The v2 baseline scored 2.28 / 5 with a 6% clean rate. Several findings from the rating pass make the baseline itself an unreliable measurement: the prompt DB has never been seeded (Few-Shot Generator ran on hardcoded fallback templates), the Critic miscalibrates on mid-quality outputs (poisoning the auto-ingest signal), and four pipeline bugs (#10–13) produce visible quality regressions. Phase B4 closes these gaps and re-runs the baseline so subsequent phases measure against a system running as designed.

### Why this before Phase C

Phase C adds a Research agent to the pipeline. Layering new agents on a substrate where 94% of outputs need editing inherits every existing failure mode and adds new surfaces. The current baseline cannot tell us whether Phase C's additions help, hurt, or do nothing, because the signal is dominated by upstream bugs. Fix the substrate before adding to it.

### Scope

- Seed the prompt DB (`bash scripts/seed_prompt_db.sh`) and verify Few-Shot retrieval is actually pulling from it.
- Verify eval-harness captures match what `pogo.html` renders (B3 closure step 4). If they diverge, reconcile before re-rating.
- Fix bugs #10, #11, #12, #13.
- Decontaminate the Critic system prompt; add a Critic calibration test against 5 known-good and 5 known-bad prompts; gate auto-ingest on calibrated scores plus a second signal.
- Re-run the 18-entry baseline and rate it.

Out of scope: any Phase C / D / E feature work. Bedrock cleanups (still deferred per B3 entry).

### Claude Code prompt (paste below)

*To be drafted.*

### Acceptance criteria for Phase B4

*To be specified when the prompt is drafted. Will include: prompt DB seeded and verified, all four bugs fixed with regression tests, Critic calibration test passing, harness/UI parity confirmed, re-baseline captured and rated, mean score improvement documented.*

---

## Phase C onward

To be drafted after Phase B4 lands and the v2.1-substrate baseline is captured + rated. The Phase C scope (Research agent expansion) is defined in `WORKFLOW_REDESIGN.md` but the Claude Code prompt itself depends on what Phase B4 surfaces.

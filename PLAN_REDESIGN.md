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
- [x] **Phase B2.** Build eval harness — harness built 2026-05-13, commit `4201783`, branch `claude/pogo-v2-phase-b2-2SIti`. Baseline capture deferred during B2 (AWS credentials gap in Claude Code sandbox); captured locally after B3.
- [x] **Phase B3.** Migrate Bedrock → Anthropic API + bump models to Sonnet 4.6 / Haiku 4.5 — code migration completed 2026-05-13, commits `a704ae5` and `9ca3650`, branch `claude/migrate-anthropic-api-AYZnc`. Smoke test passed locally 2026-05-14.
- [ ] **Phase B3 closure.** Outstanding pre-Phase-C work: capture full v2 baseline run locally, rate baseline (manual, 30–45 min), merge `claude/migrate-anthropic-api-AYZnc` into `redesign/v2.1`, deploy to Lambda.
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

Test backfill for original PLAN.md Phases 6–7. Test count: 145 → 179 (+34 tests). Zero bugs surfaced.

Modules covered:
- `orchestrator/live_test.py` (+7 tests: TestCleanGeneratedInput, TestFallbackTestInput, TestRunLiveTestFallback)
- `agents/critic.parse_scores` (+3 tests: TestCriticParseScores covering JSON path, regex fallback, malformed JSON)
- `orchestrator/agent_router.py` (+6 tests: TestAgentRouterHelpers covering resolve_target_model_id variants, fetch_reference_prompts fallback, run_critic_review wiring)
- `orchestrator/orchestrator._split_final_draft` (+4 tests)
- `orchestrator/orchestrator._parse_fewshot_examples` (+3 tests)
- `orchestrator/orchestrator._ingest_accepted_prompt` (+2 tests)
- `prompt_db/ingest.py` seed helpers (+6 tests: TestSeedNormalisation)
- `prompt_db/admin.py` (+3 tests: TestAdminValidation)

**2026-05-13 — Phase B2 complete (commit `4201783`, branch `claude/pogo-v2-phase-b2-2SIti`)**

Eval harness built. Baseline capture deferred during this phase — Claude Code sandbox had no AWS credentials, so all 18 entries skipped with `NoCredentialsError`. Placeholder file at `eval/runs/2026-05-13_4201783_v2-baseline.json`. Real baseline captured locally after B3 migration.

Artifacts:
- `eval/inputs.json` (18 entries spanning all 11 task categories, ~60/40 one-shot/chained, Claude ×7 / GPT ×6 / Gemini ×5)
- `eval/run_eval.py` (programmatic orchestrator driver)
- `eval/rate.py` (CLI rating tool)
- `eval/README.md` (workflow docs)
- Tests: B2 branch baseline was 145 → 155 (+10) because B2 branched before B1 landed. Both sets present after B3 Stage 0 branch consolidation.

**Branch divergence note:** Phases A, B1, B2 each landed on separate branches. Phase B3 Stage 0 consolidated all three into the long-lived `redesign/v2.1` branch.

**2026-05-13 — Phase B3 code migration complete (commits `a704ae5` and `9ca3650`, branch `claude/migrate-anthropic-api-AYZnc`)**

Migrated AWS Bedrock SDK calls to the Anthropic Python SDK across all agent and orchestrator completion paths. Bumped internal agent models to current versions. Test count: 189 → 195 (+6 tests in `tests/test_anthropic_client.py`).

Branch consolidation:
- Created `redesign/v2.1` from main (which already contained Phase A).
- Merged `origin/claude/pogo-v2-phase-b1-k3205` and `origin/claude/pogo-v2-phase-b2-2SIti` into `redesign/v2.1`.
- Resolved `PLAN_REDESIGN.md` conflicts by keeping every changelog entry chronological with all completed-phase checkboxes ticked.
- 189 tests pass on `redesign/v2.1` before any B3 code changes. Pushed to origin.
- B3 work cut as `claude/migrate-anthropic-api-AYZnc` from `redesign/v2.1`.

Migration scope:
- Central wrapper at `orchestrator/agent_router.invoke_agent_raw` (used by every v2 agent) migrated to use a new thin client layer at `orchestrator/anthropic_client.py`. Agent code downstream needed zero changes.
- Legacy v1 `/generate` path in `lambda/handler.py:generate_prompt()` also migrated.
- Model IDs bumped: `us.anthropic.claude-3-5-haiku-20241022-v1:0` → `claude-haiku-4-5-20251001` in both `agent_router.ARCHITECT_MODEL_ID` and `lambda/handler.GEN_MODEL_ID`. No Sonnet model was configured in code, so the Sonnet mapping was a no-op for this codebase.
- `anthropic` added to `requirements.txt`. `deploy.sh` packages it and fails fast if `ANTHROPIC_API_KEY` is not exported in the caller's shell.
- `ANTHROPIC_API_KEY` plumbed end-to-end: read by `orchestrator/anthropic_client._get_client()` (raises `AnthropicConfigError` if missing); injected into Lambda env via `update-function-configuration --environment "Variables={ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY}"`.

Out of scope (intentionally deferred):
- `prompt_db/embeddings.py` and `lambda/handler.py:embed_query()` still call Bedrock for Titan embeddings (Anthropic SDK has no comparable model). Future cleanup if/when embeddings move to a different vendor (Voyage AI, OpenAI ada, or local sentence-transformers).
- `pogo/scripts/build_index_titan.py` offline indexer untouched.
- Other AWS services unchanged: `orchestrator/session.py` (DynamoDB), `prompt_db/store.py` (S3), `scripts/ingest.py` (S3).
- AmazonBedrockFullAccess IAM attachment in `deploy.sh` left in place; scope down once embeddings move off Bedrock.
- `_normalise_usage` in `agent_router.py` is now defensive overlap with the wrapper's own normalisation — prune in Phase E.
- v1 `/generate` path still imports `boto3` eagerly; could lazy-import inside `embed_query`.

Smoke test deferred to local environment because Claude Code sandbox had no `ANTHROPIC_API_KEY`.

**2026-05-14 — Phase B3 smoke test passed locally**

Smoke test completed end-to-end on the `claude/migrate-anthropic-api-AYZnc` branch: 1 prompt through the migrated Anthropic SDK path, 57.1s elapsed, 16,640 tokens (10,054 input + 6,586 output), critic score 0.70. Migration validated against `eval_001` (the chained-path Flask CSV upload prompt). Captured run file: `eval/runs/2026-05-14_391a012_v2-baseline-anthropic.json`.

**Local-dev requirement discovered:** `requirements.txt` was missing `boto3`. AWS Lambda runtime provides it implicitly, so production worked, but local dev (and the Claude Code sandbox) needs an explicit install. Added `boto3` to `requirements.txt` and committed.

**Three pre-existing pipeline bugs surfaced by inspecting the captured smoke-test output and the full v2 baseline run. These are NOT migration bugs — they exist in production before B3, but the eval surfaced them for the first time because no one was reading full captured outputs before.**

- **Bug #10: Few-Shot Generator CoT preamble leak.** Output contains a `<thinking>...</thinking>` block exposing chain-of-thought reasoning. Same shape as bugs #8 and #9 fixed in Phase A for Clarifier and Context Scout, but the Few-Shot Generator wasn't covered by Phase A's `_strip_preamble()` helper. Fix: extend `_strip_preamble` coverage to Few-Shot Generator output in `orchestrator/response_merger.py`. Defer to Phase E (renderer touch).
- **Bug #11: Few-Shot Generator uses stale Architect draft instead of refined output.** When the Architect refines its draft based on Critic feedback (e.g., correcting a hallucinated schema), the Few-Shot Generator continues to use the original draft's content. Visible in eval_001: Architect draft hallucinated schema fields (`id`/`first_name`/`last_name`/`is_active`), Critic flagged the mismatch, Architect's refined output used the correct schema (`name`/`email`/`age`), but Few-Shot Generator's examples still used the wrong schema. Causes internally-contradictory final outputs. Defer to Phase E, or earlier if scope allows.
- **Bug #12: Architect output occasionally drops entirely from final output.** Reproducible on `eval_001` in the v2-baseline run (critic score 0.10) — the final_output contains only Few-Shot examples plus a stray curl command, with no `<role>`, `<context>`, `<task>`, or `<constraints>` sections. Same prompt produced a complete output in the 2026-05-14 smoke test (critic 0.70). Root cause unknown; possibly an `_split_final_draft` parsing edge case when the Architect output is shaped unusually, or model nondeterminism producing unparseable structures. Investigate in Phase E.

**Outstanding B3 closure work (before Phase C can start):**
1. Run full baseline locally (`python eval/run_eval.py --label v2-baseline`) — captures all 18 prompts.
2. Rate the baseline (`python eval/rate.py eval/runs/<file>.json`) — manual, 30–45 minutes.
3. Commit the rated baseline.
4. Merge `claude/migrate-anthropic-api-AYZnc` into `redesign/v2.1`.
5. Deploy to Lambda from `redesign/v2.1` (`./deploy.sh` with `ANTHROPIC_API_KEY` set in shell).
6. Verify prod still works against the live URL.

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

The harness build is complete but the baseline capture failed during B2 due to a credentials gap in the Claude Code sandbox. Real baseline captured locally after Phase B3 lands.

---

## Phase B3 — Migrate Bedrock → Anthropic API + bump models

### Goal

Replace AWS Bedrock SDK calls with the Anthropic Python SDK across all agent and orchestrator code. Bump internal agent models to current versions (`claude-sonnet-4-6` and `claude-haiku-4-5-20251001`). Keep all other AWS infrastructure (Lambda, API Gateway, DynamoDB, S3) unchanged.

### Why this before Phase C

Three reasons:
1. Every redesign phase from C onward adds new agent code. Writing it against the Anthropic SDK from the start is cheaper than porting it later.
2. The v2 baseline measurement must be captured on the stack that subsequent phases run on. Capturing on Bedrock then migrating would conflate "SDK changed" with "redesign changed."
3. Removes AWS-credentials friction from local dev and eval-harness runs (partially — Titan embeddings still on Bedrock).

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

- `redesign/v2.1` exists with Phase A, B1, B2 merged in and 189 tests passing as a starting point ✅
- No remaining Bedrock SDK calls for agent/completion code (Titan embeddings deferred as out-of-scope) ✅
- Anthropic SDK is in deps and Lambda package ✅
- All agent calls go through the new wrapper using `claude-sonnet-4-6` and `claude-haiku-4-5-20251001` ✅
- `ANTHROPIC_API_KEY` is required env var; missing key produces a clear error ✅
- All existing tests pass; new wrapper tests pass (195 passing) ✅
- ARCHITECTURE.md and eval/README.md updated ✅
- Smoke test produces a working session end-to-end (passed locally 2026-05-14) ✅
- `PLAN_REDESIGN.md` updated with checked-off status and detailed Changelog entry ✅

---

## Phase C onward

To be drafted after Phase B3 closure work is done (full baseline captured, baseline rated, B3 merged into `redesign/v2.1`, deployed to Lambda, prod verified).

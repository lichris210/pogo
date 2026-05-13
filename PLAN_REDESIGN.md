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
- [ ] **Phase B1.** Test backfill for original Phases 6–7
- [ ] **Phase B2.** Build eval harness (manual ratings, v2 baseline capture)
- [ ] **Phase C.** Research agent expansion (autonomous discovery, references, summarization, conditional triggering)
- [ ] **Phase D.** Decomposer agent + per-phase model recommendation + tier maps for each frontier family
- [ ] **Phase E.** Per-phase RAG retrieval + per-phase ingestion + format profile inner-only scoping + phase plan assembly
- [ ] **Phase F.** UI updates (manual overrides, per-phase model display, aggregate cost estimate)
- [ ] **Phase G.** E2E smoke test extensions covering both paths

## Changelog

*(Preserve Claude Code's exact wording from its in-flight updates; the entry below is a summary reconstruction.)*

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

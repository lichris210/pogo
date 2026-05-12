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
- [ ] **Phase A.** Bug fixes (#1, #8, #9)
- [ ] **Phase B.** Test backfill (Phases 6–7) + build eval harness
- [ ] **Phase C.** Research agent expansion (autonomous discovery, references, summarization, conditional triggering)
- [ ] **Phase D.** Decomposer agent + per-phase model recommendation + tier maps for each frontier family
- [ ] **Phase E.** Per-phase RAG retrieval + per-phase ingestion + format profile inner-only scoping + phase plan assembly
- [ ] **Phase F.** UI updates (manual overrides, per-phase model display, aggregate cost estimate)
- [ ] **Phase G.** E2E smoke test extensions covering both paths

## Changelog

Add a dated entry here after every completed phase. Note anything subsequent phases need to know about (renamed files, changed contracts, deferred items).

*No entries yet.*

---

## Phase A — Bug Fixes

### Goal

Fix three bugs that block clean downstream redesign work:
1. `deploy.sh` macOS sed-i incompatibility (bug #1)
2. Clarifier agent leaks CoT preamble into "Things to sharpen" output (bug #8)
3. Context Scout agent leaks CoT preamble into "To make this stronger" output (bug #9)

### Why these first

Phases C and D add new agents (Research agent, Decomposer) on top of the existing Clarifier and Context Scout. Adding new agents on top of two that already leak chain-of-thought compounds the problem. Fix the leaks before adding more agents. Bug #1 blocks clean deploys for everyone on macOS.

### Claude Code prompt (paste below)

```
You are working on POGO v2.1, a multi-agent prompt optimization system.

Before doing anything, read these files in this order to load context:
1. WORKFLOW_REDESIGN.md — the target state for v2.1, the final product we're working toward.
2. PLAN_REDESIGN.md — the implementation plan. We are executing Phase A.
3. ARCHITECTURE.md — current production architecture.

Then locate and read the source files for the Clarifier agent, the Context Scout agent, and deploy.sh. Do not modify anything yet.

Your task for Phase A is to fix three bugs. Address them in this order.

==========
BUG #1 — deploy.sh macOS sed incompatibility
==========

Current behavior: deploy.sh uses `sed -i 's/.../.../' file` which is GNU sed syntax. macOS ships BSD sed, which requires `sed -i '' 's/.../.../'`. The result is that deploys fail on macOS unless the developer manually patches the sed command. This patch was applied once during the initial deploy but never baked into the script.

Fix: Use the cross-platform pattern `sed -i.bak 's/.../.../' file && rm file.bak`, which works on both GNU and BSD sed. Apply this to every `sed -i` invocation in deploy.sh.

Verify the fix by reading deploy.sh after the change and confirming no bare `sed -i` invocations remain. If you can run it locally (or simulate the relevant section in isolation), do so.

==========
BUG #8 — Clarifier CoT preamble leak
==========

Current behavior: The Clarifier's "Things to sharpen" output includes a chain-of-thought preamble (e.g., "Let me think about what needs clarification here...") before the actual list of items. This preamble leaks to the user.

Investigate first:
1. Read the Clarifier agent's source file.
2. Inspect its system prompt.
3. Inspect how its output is parsed and returned to the user.
4. Determine the leak's origin: is the system prompt allowing preamble, or is the output not being parsed/stripped?

Pick the cleanest fix given the code you find:
- Tighten the system prompt to forbid preamble explicitly ("Output only the clarification items as a JSON array. No reasoning, no preamble, no explanation.").
- Switch to structured output (JSON) so preamble can't sneak in.
- Post-process the output to strip any text before the first item marker.

State which approach you chose and why in your final response.

Verify by running the Clarifier with at least 3 test prompts and confirming clean output. If existing tests don't cover this, add or extend tests in the relevant test file.

==========
BUG #9 — Context Scout CoT preamble leak
==========

Same shape as bug #8, applied to the Context Scout's "To make this stronger" output. Apply the same fix pattern you chose for bug #8 unless something in the Context Scout's code makes a different approach better. State your reasoning.

Note: Context Scout will be renamed to "Research agent" with expanded scope in Phase C. The CoT fix here still applies after the rename. Don't rename the file in Phase A; just fix the bug.

Verify with at least 3 test prompts and add/extend tests.

==========
Test verification
==========

All existing tests must still pass. New or extended tests for bugs #8 and #9 must pass. If you can test deploy.sh in isolation, do so.

==========
When Phase A is complete
==========

1. Update PLAN_REDESIGN.md:
   - Check off Phase A in the Status checklist.
   - Add a Changelog entry with today's date, a one-line summary of what was fixed, the approach taken for bugs #8 and #9, and any notes subsequent phases need (e.g., if you changed Context Scout's output schema, Phase C needs to know).
2. Commit all changes with the message: "Phase A complete: fix bugs #1, #8, #9"
3. Push.
4. Stop. Do not proceed to Phase B without explicit instruction.

In your final response to me, summarize:
- Which fix approach you chose for bugs #8 and #9 and why
- Any files that changed beyond the obvious targets
- Any surprises or new bugs surfaced during the work
- Confirmation that all tests pass
```

### Acceptance criteria for Phase A

- `deploy.sh` runs on macOS without manual sed patching
- Clarifier output contains no CoT preamble across 3+ test runs
- Context Scout output contains no CoT preamble across 3+ test runs
- All existing tests pass
- New tests for bug #8 and #9 added and passing
- `PLAN_REDESIGN.md` updated with checked-off status and Changelog entry

---

## Phase B onward

To be drafted after Phase A lands. Each subsequent phase will follow the same structure: goal, why-this-now, Claude Code prompt, acceptance criteria.

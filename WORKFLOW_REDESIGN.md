# POGO v2.1 Workflow Redesign

## Overview

POGO v2 currently runs a fixed pipeline: Prompt Architect, Context Scout, Clarifier, Few-Shot Generator, Critic, Guardrails. Every prompt flows through the same path. This redesign adds a path divergence: prompts that benefit from being broken into sequential phases follow a chained-generation path, while one-shot prompts follow the existing single-prompt path.

The redesign also folds in three new capabilities: per-phase model tier recommendation, reference-based context gathering, and per-phase ingestion to the vector DB.

## The four-step workflow

### Step 1: Classification

The user selects the target frontier model family (Claude, GPT, or Gemini) and submits the original prompt.

POGO then:
1. Determines the output format based on the target family (loads the relevant format profile).
2. Classifies whether the prompt is best handled one-shot or as a chained sequence of sub-prompts with test gates.

The chain-vs-one-shot classification is **tentative**. POGO can revise it after Step 2 surfaces full context. The user can also override the classifier at any point through a UI toggle.

### Step 2: Context and references

Regardless of which path was tentatively assigned, POGO gathers context. Three input streams:

**Clarifier questions.** The Clarifier asks targeted questions when the prompt has gaps the Architect would otherwise have to guess at.

**Research agent (autonomous discovery).** The Research agent searches public sources (GitHub repos, product pages, design galleries) for similar projects or code that could inform the task. It surfaces 3 to 5 candidates with brief descriptions. The user approves which candidates to include before they enter the prompt.

**User-uploaded references and data.** The user can upload:
- *Conceptual references*: URLs, product names, or screenshots that capture aesthetic or functional intent
- *Structural references*: code files, repo snapshots, or pasted snippets that capture patterns and conventions
- *Task data*: actual data the prompt will operate on

All three input types are optional. The user can skip any of them.

The reference-gathering step is **task-category-conditional**. It fires for code_generation, creative_writing, agentic_workflow, and analysis. POGO skips it for translation, classification, summarization, and extraction.

For chained outputs, every reference and data input carries a **destination-phase tag** indicating which phase consumes it. A UI mockup goes to phase 1 (design); a code structure goes to phase 3 (implementation).

The Research agent **summarizes** references rather than dumping raw content. A full repo blows context windows; a distilled description ("uses Repository pattern, Tailwind for styling, file-per-component convention") does not.

At the end of Step 2, POGO re-checks the tentative classification from Step 1 against the now-complete context. If it should flip, it flips.

### Step 3: Generation

The pipeline diverges based on the now-confirmed classification.

**One-shot path:**
1. Prompt Architect drafts the canonical prompt.
2. RAG retrieval pulls similar high-scoring prompts from the vector DB.
3. Few-Shot Generator injects examples where useful.
4. Critic evaluates and assigns a quality score.

Output: one optimized prompt formatted for the target model.

**Chained path:**
1. Decomposer produces a phase plan: an ordered list of sub-tasks with dependencies, destination-tagged data and references, and a recommended model tier per phase (see "Per-phase model recommendation" below).
2. For each phase, in order:
   - Architect drafts the phase prompt.
   - RAG retrieves examples scoped to that phase's narrower role.
   - Few-Shot Generator enriches with relevant examples.
   - Critic generates test specifications and embeds them in the phase prompt. The downstream coding agent will author and run the actual tests at runtime.
3. POGO assembles the phases into the final phase plan artifact.

Output: a markdown phase plan with N copy-pasteable phase prompts. The wrapper is human-readable markdown. Each inner phase prompt follows the target model's format profile.

### Step 4: Critique and iteration

The Critic evaluates the final output (single prompt or full phase plan) and assigns a quality score.

The user reviews the output. If they request iteration, POGO re-runs the whole plan rather than individual phases. This keeps the iteration model simple and matches how the user consumes the output.

When the user accepts the output, it enters the ingestion flywheel.

## Agent inventory

| Agent | Role | When it fires |
|---|---|---|
| Guardrails (input) | Reject malformed, abusive, or out-of-scope prompts | Early Step 1 |
| Classifier | Determine target model + one-shot/chained path | Step 1 |
| Clarifier | Ask targeted clarifying questions | Step 2 |
| Research agent | Find similar projects/code from public sources (renamed Context Scout, expanded scope) | Step 2, conditional on task category |
| Decomposer (new) | Produce phase plan with per-phase model recommendations | Step 3, chained path only |
| Prompt Architect | Draft the canonical prompt (or per-phase prompts in chained mode) | Step 3 |
| Few-Shot Generator | Inject examples from RAG retrieval | Step 3 |
| Critic | Evaluate quality; generate test specs in chained mode | Step 3 (per-phase in chained mode; once at end in one-shot mode) |
| Guardrails (output) | Final safety check before delivery | Post-Step 3 |

## State machine

The current states stay intact:

```
initial → awaiting_context → review → iterating → accepted
```

Step 1 (classification), Step 2 (context and references), and any re-classification all fold into `awaiting_context`. Step 4 (review and iteration) covers `review` and `iterating`. This keeps the state machine simple and avoids architectural churn. If debugging becomes painful later, splitting out `classifying` and `gathering_references` states is a future option.

## Per-phase model recommendation

In the chained path, the Decomposer tags each phase with a recommended model tier from the user's chosen frontier family. The user can override any recommendation.

Classification rubric, applied per phase:

1. **Task complexity.** Retrieval and reformatting goes to the cheapest tier. Single-step reasoning goes to the mid tier. Multi-step or novel reasoning goes to the top tier.
2. **Position in chain.** Earlier phases bias up a tier because errors compound downstream. A sloppy phase 1 schema corrupts every phase that follows.
3. **Stakes.** Phases whose output gates the rest of the chain bias up.
4. **Output length.** Long-form outputs constrain the choice to tiers that support them.
5. **Modality.** Vision-heavy phases require a vision-capable tier.
6. **Latency.** Rarely matters for chained workflows since the user pastes between phases.

Default when uncertain: the family's mid tier. Do not upsell the top tier based on how important the task sounds.

Each frontier family needs a tier map stored alongside the format profiles. For Claude: Haiku 4.5, Sonnet 4.6, Opus 4.6, Opus 4.7. For GPT and Gemini, the equivalent ladders. The tier map should be config-driven so it updates when new models ship.

The phase plan output displays each phase's recommendation inline (e.g., "Phase 2: Schema Design • Recommended: Sonnet 4.6") with a short rationale the user can collapse. An aggregate cost estimate for running the full plan ("approximately $0.40 end-to-end") appears at the top of the plan.

## Format profiles

In the one-shot path, the target model's format profile applies to the single output prompt.

In the chained path:
- The **wrapper** (the overall phase plan structure that the user reads) uses clean markdown for human readability. No model-specific formatting.
- The **inner phase prompts** follow the target model's format profile, since those prompts get pasted into the downstream model.

## Ingestion flywheel

POGO auto-ingests outputs scoring ≥0.8 from the Critic into the vector DB. The 0.95 cosine similarity dedup threshold stays in place.

For one-shot outputs, ingestion is unchanged: one entry per accepted prompt.

For chained outputs, POGO ingests **each phase separately** with a role tag (planning, schema_design, implementation, testing, validation, etc.). This makes per-phase RAG retrieval sharp over time, which is where the leverage lives for chained generation quality. Whole-plan ingestion as a separate entry is a nice-to-have but not required.

## Guardrails

Guardrails fire twice:
- **Input guardrails** at early Step 1, before any agent runs. They reject malformed prompts, abusive content, and out-of-scope requests.
- **Output guardrails** post-Step 3, before delivery. They check the final output (single prompt or phase plan) for leaked chain-of-thought, hallucinated tool references, and policy violations.

## Open questions and deferred decisions

- **State splitting.** The redesign folds new logic into existing states. Splitting out `classifying` and `gathering_references` is a future option if debugging gets painful.
- **Phase plan partial re-runs.** Iteration re-runs the whole plan. If users repeatedly fix the same phase, per-phase iteration is worth revisiting.
- **Cross-family routing.** Today the user picks one frontier family at session start and all phases stay within it. Allowing mixed-family chains (Opus for planning, GPT-5 for implementation) is technically possible but adds significant UX and cost-tracking complexity. Deferred.
- **Reference auto-fetch from a URL.** The Research agent surfaces candidates the user approves. Fetching and parsing a user-supplied URL automatically would be useful but adds a scraping surface. Deferred.

## Implementation notes

Address the existing bug list before starting the redesign. Specifically:

1. **CoT preamble leaks in Clarifier and Context Scout (bugs #8, #9).** Adding a Decomposer on top of two agents that already leak preamble compounds the problem. Fix these first.
2. **Phases 6–7 unit test gap.** Adding what amounts to a Phase 8 (the Decomposer and per-phase routing) without test coverage on existing phases will regress.
3. **`deploy.sh` sed fix (bug #1).** Bake the `sed -i` macOS compatibility into the deploy script before the next round of Lambda updates.

Suggested phasing for implementing the redesign itself:

- **Redesign Phase A.** Bug fixes #1, #8, #9. Test coverage backfill for Phases 6–7.
- **Redesign Phase B.** Research agent expansion (autonomous discovery + reference uploads). Reference summarization. Task-category-conditional triggering.
- **Redesign Phase C.** Decomposer agent. Per-phase model recommendation. Tier maps for each family.
- **Redesign Phase D.** Per-phase RAG retrieval and per-phase ingestion. Phase plan assembly. Format profile scoping (inner-only).
- **Redesign Phase E.** UI updates for the new flows (manual overrides, per-phase model display, aggregate cost estimate).
- **Redesign Phase F.** End-to-end smoke tests covering both paths and the override behaviors.

Each redesign phase should ship with its own Claude Code prompt block and explicit test count, matching the existing PLAN.md convention.

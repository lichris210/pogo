# POGO Eval Harness

A minimal, manual-rating eval harness for measuring POGO's output quality across redesign phases.

## Why this exists

Every redesign phase from Phase C onward changes pipeline behavior (Research agent expansion, Decomposer, per-phase RAG). Without a fixed input set and an honest measurement, "did this help?" is unanswerable. This harness:

1. Drives the orchestrator programmatically across a curated set of inputs.
2. Captures Architect draft, Critic score, final output, elapsed time, and token counts.
3. Lets a human operator rate each output 1–5.
4. Stores the run file as the historical record.

LLM-as-judge scoring, downstream verification (does the generated code actually compile and pass tests), and a web rating UI are deferred to later phases.

## Files

| Path | Purpose |
|---|---|
| `eval/inputs.json` | Curated eval input set (~18 prompts spanning all 11 task categories). |
| `eval/run_eval.py` | Runner. Drives the orchestrator, writes a versioned run file. |
| `eval/rate.py` | Rating CLI. Walks unrated entries and prompts for a 1–5 score. |
| `eval/runs/` | Versioned run files. Committed — these *are* the historical record. |

## Run a full eval

```bash
python eval/run_eval.py
```

This produces `eval/runs/<YYYY-MM-DD>_<short_sha>_<branch>.json`. The runner requires AWS credentials configured for Bedrock in `us-east-1` (same configuration the deployed Lambda uses). Useful flags:

- `--label v2-baseline` — override the branch suffix on the output filename.
- `--limit N` — only run the first N entries (smoke test).
- `--dry-run` — stub all Bedrock calls with deterministic placeholders. Use only for shape-checking the harness itself; the output is not a real measurement.

Entries are processed one at a time. If an entry cannot be captured (missing `pre_baked_context`, an unknown model family, a Bedrock failure, etc.) the runner logs it, marks it skipped in the output file, and moves on. The run never fails as a whole.

## Rate outputs

```bash
python eval/rate.py eval/runs/<file>.json
```

The CLI walks each entry where `rating.score` is null. For each one it prints the input prompt, the captured final output, and the Critic score, then asks for:

1. A score (1–5 integer).
2. Optional free-text notes.

Ratings are written back to the same file after each entry. Type `q` at any prompt to quit; entries already rated this session are preserved.

### Rating scale

| Score | Meaning |
|---|---|
| 1 | Output is unusable or wrong. |
| 2 | Output is below baseline quality. |
| 3 | Output is acceptable, equivalent to baseline expectation. |
| 4 | Output is above baseline expectation. |
| 5 | Output is significantly better than baseline expectation. |

"Baseline expectation" for a given run = the corresponding v2 baseline output for the same eval_id. When rating the v2 baseline itself, use score 3 as the anchor: it is, definitionally, the baseline.

## Adding new eval entries

Append to `eval/inputs.json`. Each entry needs:

```json
{
  "id": "eval_NNN",
  "task_category": "one of the 11 canonical task categories",
  "target_model_family": "claude | gpt | gemini",
  "expected_path": "one_shot | chained",
  "user_prompt": "the original prompt as a user would submit it",
  "pre_baked_context": "context that substitutes for the Clarifier reply",
  "notes": "why this prompt matters for eval"
}
```

Guidelines:

- **Do not** copy prompts from `seed_prompts.json`. Those feed the vector DB; using them as eval inputs creates measurement contamination.
- Include realistic complexity — avoid trivial prompts.
- `pre_baked_context` must be specific enough that the pipeline can produce a useful refined prompt without a Clarifier round-trip. The runner skips entries with an empty `pre_baked_context`.
- Spread across families and across one-shot vs chained. Keep the set in the 15–25 range; bigger sets cost more per run and dilute rating attention.

## Comparing across runs

For this phase, comparison is manual:

```bash
diff <(jq '.results[] | {eval_id, score: .rating.score}' eval/runs/<run_a>.json) \
     <(jq '.results[] | {eval_id, score: .rating.score}' eval/runs/<run_b>.json)
```

A proper comparison tool (delta tables, regression flags, statistical significance) is deferred to a later phase once we have at least two rated runs to compare.

## Cost expectation per run

Per entry the pipeline issues roughly six Bedrock calls (Architect draft, Context Scout, Clarifier, Architect refine, Few-Shot Generator, Critic) at ~2k–4k total tokens each. Using Claude 3.5 Haiku as the default agent model (current production setting, $0.80 / M input, $4.00 / M output):

- ~15k input + ~10k output tokens per entry ≈ **$0.05** per entry.
- 18-entry set ≈ **$0.90** per full run.

The cost roughly doubles if `POGO_TARGET_MODEL_ID_*` is set to Sonnet or higher for any family, since the live test step uses the target model. The runner does not currently fire the live-test path, so target-family cost is only paid via the Critic when reference retrieval is heavy. Budget $1–$2 per full run with default settings; expect this to grow when the Decomposer arrives in Phase D.

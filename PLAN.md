# POGO v2 — Multi-Agent Prompt Optimization Pipeline

## Architecture Overview

POGO evolves from a single-model prompt optimizer into a multi-agent pipeline where specialized agents handle discrete stages of the prompt engineering process. The user interacts through a chat-like back-and-forth interface while an orchestrator routes their input through the appropriate agents.

### Agent Pipeline

```
User Intent + Model Selection
    → Prompt Architect (drafts structured prompt)
    → Context Scout + Clarifier (parallel — suggest context & ask questions)
    → User provides answers/context
    → Prompt Architect (refine) + Few-Shot Generator (parallel)
    → Guardrail check
    → Critic Agent + Live Test (parallel)
    → User reviews output, iterates or accepts
    → Accepted prompts with high scores → ingested into prompt database
```

### Key Design Decisions

- **Format is global config, not a post-processing step.** When the user selects a target model (Claude, GPT, Gemini), every agent inherits a format profile (XML, markdown, etc.) and outputs in that format natively.
- **Research papers are distilled into agent system prompts (static).** No RAG on academic papers.
- **RAG is used for a database of proven, battle-tested prompts.** The orchestrator queries by task category on behalf of agents, not from raw user input.
- **Conversation state is managed via DynamoDB**, keyed by session ID, storing conversation history, pipeline stage, accumulated context, model selection, and current prompt draft.

### Agents

| Agent | Purpose | Model Size | Uses RAG? |
|-------|---------|-----------|-----------|
| Prompt Architect | Drafts and refines structured prompts | Larger (e.g., Claude Sonnet) | Yes — retrieves reference prompts |
| Context Scout | Suggests supporting context the user should provide | Lighter/cheaper | No |
| Clarifier | Generates clarifying questions to surface unstated assumptions | Medium | No |
| Few-Shot Generator | Creates example input/output pairs for the prompt | Medium | Yes — pulls example patterns |
| Critic | Scores prompt quality and runs live test against target model | Larger | Yes — compares against high-scoring prompts |
| Guardrails | Rule-based + lightweight model check for anti-patterns | Mostly rule-based | No |

### Prompt Database Schema

```json
{
    "id": "prompt_042",
    "task_category": "data_analysis",
    "subcategory": "churn_prediction",
    "target_model": "claude",
    "format": "xml",
    "techniques": ["chain_of_thought", "role_assignment", "structured_output"],
    "system_prompt": "...",
    "user_prompt_template": "...",
    "few_shot_examples": [],
    "quality_score": 0.89,
    "source": "curated",
    "embedding": []
}
```

### Orchestrator State Machine

```
"initial"          → Prompt Architect → Context Scout + Clarifier → return merged response
"awaiting_context" → user replied → Prompt Architect (refine) + Few-Shot Generator → return refined prompt
"review"           → Guardrails → Critic + live test → return evaluation
"iterating"        → user requested changes → Prompt Architect (edit) → back to review
"accepted"         → store final prompt, optionally ingest into prompt DB
```

---

## Phase 1 — Agent System Prompts and Format Profiles

### Goal
Define all agent system prompts and the format profile system. No infrastructure changes yet — just the agent definitions as Python modules that can be tested in isolation.

### Files Created
- `agents/prompt_architect.py`
- `agents/context_scout.py`
- `agents/clarifier.py`
- `agents/critic.py`
- `agents/fewshot_generator.py`
- `agents/guardrails.py`
- `agents/format_profiles.py`
- `agents/__init__.py`

### Claude Code Prompt

```
Read the following files first:
- ARCHITECTURE.md (repo structure and current implementation)
- PLAN.md (new multi-agent architecture)
- prompt_engineering_principles.md (distilled research findings)

Then create an `agents/` directory with the following modules:

1. `format_profiles.py`
   - Define a FORMAT_PROFILES dictionary with entries for "claude", "gpt", and "gemini"
   - Each profile specifies: wrapper format (xml/markdown), section tags/headers, example formatting style, and any model-specific best practices
   - Include a `get_format_instructions(target_model: str) -> str` function that returns a formatted string of instructions that can be injected into any agent's system prompt
   - Include a `format_section(content: str, section_name: str, target_model: str) -> str` utility that wraps content in the appropriate format for the target model

2. `prompt_architect.py`
   - Define SYSTEM_PROMPT as a string containing:
     - The agent's role: an expert prompt engineer that creates structured, high-quality prompts
     - Relevant principles from prompt_engineering_principles.md (structural best practices, chain-of-thought, role assignment, output format specification)
     - A placeholder `{format_instructions}` that gets filled by the format profile at runtime
     - A placeholder `{reference_prompts}` for RAG-retrieved examples (empty string when none available)
     - Instructions for two modes: "draft" (create initial prompt from user intent) and "refine" (improve existing prompt with new context and clarified requirements)
   - Define a `build_messages(user_intent: str, mode: str, context: dict, format_profile: dict, reference_prompts: list = None) -> list` function that assembles the messages array for a Bedrock API call

3. `context_scout.py`
   - Define SYSTEM_PROMPT containing:
     - Role: analyzes draft prompts and identifies what supporting context would make them stronger
     - A taxonomy of context types mapped to task categories (e.g., data tasks → schema, sample data, business definitions; web tasks → example sites, brand guidelines, wireframes; writing tasks → audience info, tone examples, style guides)
     - Instructions to output suggestions as a prioritized list with brief explanations of WHY each piece of context would help
     - `{format_instructions}` placeholder
   - Define `build_messages(draft_prompt: str, task_category: str, format_profile: dict) -> list`

4. `clarifier.py`
   - Define SYSTEM_PROMPT containing:
     - Role: surfaces unstated assumptions and unresolved decisions in a draft prompt
     - Principles about what makes good clarifying questions (high-impact, non-obvious, focused on decisions not information)
     - Instructions to output 3-5 questions ranked by impact on prompt quality
     - Instructions to frame questions conversationally, not as a formal questionnaire
     - `{format_instructions}` placeholder
   - Define `build_messages(draft_prompt: str, user_intent: str, format_profile: dict) -> list`

5. `fewshot_generator.py`
   - Define SYSTEM_PROMPT containing:
     - Role: creates high-quality few-shot examples that demonstrate the expected behavior of the prompt
     - Principles from research on few-shot example selection (diversity, edge cases, format consistency)
     - Instructions to generate 2-3 example input/output pairs
     - A placeholder `{reference_examples}` for RAG-retrieved example patterns
     - `{format_instructions}` placeholder
   - Define `build_messages(refined_prompt: str, task_category: str, format_profile: dict, reference_examples: list = None) -> list`

6. `critic.py`
   - Define SYSTEM_PROMPT containing:
     - Role: evaluates prompt quality across specific dimensions
     - Scoring dimensions: clarity (0-10), specificity (0-10), completeness (0-10), constraint coverage (0-10), hallucination risk (0-10, lower is better), overall (0-10)
     - Instructions to provide a brief explanation for each score and 2-3 specific improvement suggestions
     - A placeholder `{reference_prompts}` for comparing against known good prompts
     - `{format_instructions}` placeholder
   - Define `build_messages(final_prompt: str, task_category: str, format_profile: dict, reference_prompts: list = None) -> list`
   - Define `parse_scores(response: str) -> dict` to extract scores from the model's response

7. `guardrails.py`
   - This is primarily rule-based, not a model agent
   - Define `check_prompt(prompt: str, target_model: str) -> dict` that returns:
     - `passed: bool`
     - `warnings: list[str]` (non-blocking issues)
     - `errors: list[str]` (blocking issues)
   - Rules to check:
     - Vague instructions (regex for "do your best", "be creative" without constraints, "as needed")
     - Missing output format specification
     - Contradictory instructions (basic heuristic check)
     - Token length estimate vs target model's context window
     - Missing role/persona definition
     - Prompt is empty or too short

8. `__init__.py`
   - Export all agents and the format_profiles module

Each agent module should be independently testable — I should be able to import it, call build_messages() with test data, and get a valid messages array that I can send to Bedrock.

Use Python type hints throughout. Add docstrings to each function. Keep the system prompts well-organized with clear section headers.
```

### Files to Upload to Claude Code
- `ARCHITECTURE.md`
- `PLAN.md`
- `prompt_engineering_principles.md`

---

## Phase 2 — Conversation Orchestrator

### Goal
Build the stateful conversation manager that replaces the current single-call Lambda. It manages session state, decides which agents to invoke, and merges their responses.

### Files Created/Modified
- `orchestrator/orchestrator.py`
- `orchestrator/session.py`
- `orchestrator/agent_router.py`
- `orchestrator/response_merger.py`
- DynamoDB table definition (or CloudFormation/SAM template update)
- Modified API Gateway integration

### Claude Code Prompt

```
Read the following files first:
- ARCHITECTURE.md
- PLAN.md
- agents/ directory (all files from Phase 1)
- The existing Lambda function(s) that currently handle prompt optimization requests
- Any existing SAM/CloudFormation templates

Then build the conversation orchestrator:

1. `orchestrator/session.py`
   - Define a Session dataclass/model with fields:
     - session_id: str
     - user_id: str (from auth context)
     - target_model: str (claude/gpt/gemini)
     - state: str (initial/awaiting_context/review/iterating/accepted)
     - user_intent: str (original raw input)
     - task_category: str (classified from user intent)
     - conversation_history: list[dict] (full chat history for frontend)
     - current_draft: str (latest prompt draft)
     - user_context: dict (context provided by user)
     - clarification_answers: dict (answers to clarifier questions)
     - scores: dict (critic scores, if evaluated)
     - created_at: str
     - updated_at: str
   - Define `save_session(session)` and `load_session(session_id)` using DynamoDB
   - Define `create_session(user_id, target_model, user_intent) -> Session`
   - Use the existing DynamoDB setup if there is one, otherwise create a new table definition

2. `orchestrator/agent_router.py`
   - Define `classify_task(user_intent: str) -> str` that categorizes the user's intent into task types (data_analysis, code_generation, writing, web_development, creative, research, general). Start with keyword-based classification — can upgrade to a model call later.
   - Define `invoke_agent(agent_name: str, messages: list, model_id: str) -> str` that calls Bedrock with the appropriate model. Use the existing Bedrock client/setup from the current codebase.
   - Define `invoke_parallel(agent_configs: list[dict]) -> list[str]` that runs multiple agent calls concurrently using asyncio or threading (whichever fits the existing Lambda pattern)

3. `orchestrator/response_merger.py`
   - Define `merge_scout_and_clarifier(scout_response: str, clarifier_response: str) -> str` that combines both responses into a single natural chat message. The merged response should feel like one assistant talking, not two separate bots. Structure it as:
     - The draft prompt (shown as a formatted code block or distinct section)
     - "To make this even stronger, here's what you could provide:" (from Context Scout)
     - "And a few questions to sharpen things:" (from Clarifier)
   - Define `merge_refinement(refined_prompt: str, fewshot_response: str) -> str` that combines the refined prompt with generated few-shot examples

4. `orchestrator/orchestrator.py`
   - This is the main Lambda handler (or integrates with the existing one)
   - Define `handle_message(event) -> dict` that:
     - Extracts session_id and user message from the request
     - Loads or creates the session
     - Routes based on session state:

       STATE "initial" (first message):
         1. Create session with user intent and target model
         2. Classify task category
         3. Call Prompt Architect in "draft" mode
         4. Call Context Scout + Clarifier in parallel
         5. Merge all responses
         6. Update session state to "awaiting_context"
         7. Save draft to session
         8. Return merged response

       STATE "awaiting_context" (user provided answers/context):
         1. Parse user's message — store context and answers in session
         2. Call Prompt Architect in "refine" mode with all accumulated context
         3. Call Few-Shot Generator in parallel
         4. Merge refined prompt + few-shot examples
         5. Run Guardrails check
         6. If guardrails pass: update state to "review"
         7. If guardrails fail: include warnings in response, stay in "awaiting_context"
         8. Return refined prompt with guardrail results

       STATE "review":
         1. Call Critic Agent
         2. Optionally run live test (call target model via Bedrock with the prompt)
         3. Return scores, improvement suggestions, and sample output
         4. Update state to "iterating" (user can edit) or "accepted" (user approves)

       STATE "iterating" (user wants changes):
         1. Call Prompt Architect in "refine" mode with edit instructions
         2. Run Guardrails
         3. Update state back to "review"
         4. Return updated prompt

       STATE "accepted":
         1. Return final prompt in clean format
         2. If critic score > threshold, flag for prompt DB ingestion

   - The response format for every state should include:
     - `session_id`
     - `state` (current state)
     - `message` (the chat message to display)
     - `prompt_draft` (current prompt, if any)
     - `scores` (critic scores, if in review state)
     - `sample_output` (live test result, if available)

   - Wire this into the existing API Gateway setup. The frontend should POST to the same endpoint with { session_id, message, target_model } and get back the response.

5. Update any SAM/CloudFormation templates to:
   - Add the DynamoDB sessions table if it doesn't exist
   - Update Lambda permissions for the new table
   - Ensure the Lambda timeout is sufficient for parallel agent calls (suggest 60s)

Keep all existing functionality working — the new orchestrator should be additive. If there's an existing endpoint, add the orchestrator as a new route rather than replacing the old one, so we can test side by side.

Write unit tests for:
- Session create/save/load
- Task classification
- State transitions (each state handler)
- Response merging
```

### Files to Upload to Claude Code
- `ARCHITECTURE.md`
- `PLAN.md`
- All files from `agents/` directory
- Existing Lambda function(s)
- Existing SAM/CloudFormation templates
- Any existing Bedrock client utilities

---

## Phase 3 — Prompt Database RAG Swap

### Goal
Replace the research paper vector store with a database of proven prompts. Change retrieval to be agent-side (orchestrator queries on behalf of agents) using task categories instead of raw user input.

### Files Created/Modified
- `prompt_db/ingest.py`
- `prompt_db/retrieve.py`
- `prompt_db/schema.py`
- Modified orchestrator to integrate retrieval
- Seed data ingestion script

### Claude Code Prompt

```
Read the following files first:
- ARCHITECTURE.md
- PLAN.md
- seed_prompts.json (the curated prompt database seed file)
- The existing vector store / RAG implementation (embeddings, retrieval, any Bedrock Titan embedding calls)
- orchestrator/ directory (all files from Phase 2)
- agents/ directory (all files from Phase 1)

The current RAG implementation uses research papers. We're replacing this with a database of proven, high-quality prompts. The key change: the orchestrator queries on behalf of agents using task categories, not from the user's raw input.

1. `prompt_db/schema.py`
   - Define a PromptRecord dataclass with fields:
     - id: str
     - task_category: str (data_analysis, code_generation, writing, web_development, creative, research, general)
     - subcategory: str (more specific, e.g., "churn_prediction", "api_endpoint", "blog_post")
     - target_model: str (claude/gpt/gemini)
     - format: str (xml/markdown)
     - techniques: list[str] (chain_of_thought, role_assignment, few_shot, structured_output, etc.)
     - system_prompt: str
     - user_prompt_template: str
     - few_shot_examples: list[dict] (optional)
     - quality_score: float (0.0-1.0)
     - source: str (curated/user_generated)
     - created_at: str
   - Define `to_embedding_text(record: PromptRecord) -> str` that creates the text string to embed — combine task_category, subcategory, techniques, and a summary of the prompt content. This is what gets embedded, NOT the raw prompt text.

2. `prompt_db/ingest.py`
   - Define `ingest_seed_data(seed_file_path: str)` that:
     - Reads seed_prompts.json
     - For each prompt record:
       - Validates it against the PromptRecord schema
       - Generates the embedding text via to_embedding_text()
       - Calls Bedrock Titan Embeddings to get the vector (reuse the existing embedding setup)
       - Stores the record + embedding in the vector store
     - Use the existing vector store infrastructure — just change what's being stored
   - Define `ingest_single_prompt(record: PromptRecord)` for adding individual prompts later (used by the acceptance pipeline)
   - Make this runnable as a standalone script: `python -m prompt_db.ingest --seed-file seed_prompts.json`

3. `prompt_db/retrieve.py`
   - Define `retrieve_reference_prompts(task_category: str, target_model: str, k: int = 3) -> list[PromptRecord]` that:
     - Constructs a query string from the task category (e.g., "best practices for data_analysis prompts structure format")
     - Calls the embedding model to vectorize the query
     - Queries the vector store with the embedding
     - Filters results by target_model (post-retrieval filter if your vector store doesn't support metadata filtering, pre-filter if it does)
     - Returns the top k matching PromptRecord objects
   - Define `retrieve_few_shot_examples(task_category: str, target_model: str, k: int = 2) -> list[dict]` that specifically retrieves prompts with non-empty few_shot_examples fields and returns just the examples

4. Modify `orchestrator/agent_router.py`:
   - Import retrieve_reference_prompts and retrieve_few_shot_examples
   - Before calling the Prompt Architect, retrieve reference prompts for the task category and inject them into the agent's context
   - Before calling the Few-Shot Generator, retrieve few-shot examples and inject them
   - Before calling the Critic, retrieve high-scoring reference prompts for comparison

5. Modify `orchestrator/orchestrator.py`:
   - In the "accepted" state handler, if the critic score is above 0.8 (configurable threshold), call ingest_single_prompt() to add the new prompt to the database
   - Add the quality_score from the critic into the record

Do NOT delete the existing research paper embeddings or infrastructure. Just stop querying them — route all retrieval through the new prompt_db module. We want to be able to fall back if needed.

Write a test script that:
- Ingests the seed data
- Runs a few test retrievals for different task categories
- Verifies the results are relevant and properly filtered by target model
```

### Files to Upload to Claude Code
- `ARCHITECTURE.md`
- `PLAN.md`
- `seed_prompts.json`
- All existing RAG/vector store code
- `orchestrator/` directory
- `agents/` directory

---

## Phase 4 — Guardrails Enhancement

### Goal
Expand the rule-based guardrails from Phase 1 into a more robust checking system. This is a quick win that adds polish.

> **Note:** This phase is split into 3 sub-prompts (4A, 4B, 4C) to keep each generation small enough to avoid Claude Code stream idle timeouts. Run them in order.

### Phase 4A — Severity Levels and New Check Functions

```
Read agents/guardrails.py and agents/format_profiles.py.

Update guardrails.py with these changes only (do NOT write tests yet):

1. Add a severity field to each finding: "info", "warning", or "error"
   - info: suggestions for improvement, not blocking
   - warning: likely issues that should be addressed
   - error: critical problems that will cause poor results

2. Change the internal representation to use a findings list:
   {"passed": bool, "findings": [{"severity": str, "check": str, "message": str}]}

3. Keep all existing checks but tag them with appropriate severity levels

4. Add these NEW checks:
   - Contradictory instruction detection: look for pairs like "be concise" + "provide detailed explanations", "respond in JSON" + "use markdown formatting", "never use bullet points" + "list the items"
   - Improved token estimation: count approximate tokens (words * 1.3 as rough estimate) and warn if the prompt + expected output might exceed the target model's context window. Use these limits: claude = 200k, gpt = 128k, gemini = 1M
   - Ambiguous pronoun detection: flag cases where "it", "this", "that" are used without clear antecedents in instructions
   - Missing constraint detection: if the prompt asks for creative/generative output but has no constraints on length, format, or scope, flag it
   - Duplicate instruction detection: flag near-identical instructions that appear in multiple sections

5. Keep backward compatibility — check_prompt() should still return {"passed", "warnings", "errors"} by deriving them from the new findings list.

Keep the module self-contained with no model calls — all checks are regex, heuristic, or rule-based.
```

### Phase 4B — suggest_fixes() Function

```
Read agents/guardrails.py (updated in Phase 4A).

Add a suggest_fixes(findings: list, prompt: str) -> list[str] function that generates specific, actionable fix suggestions for each finding. Examples of the kind of suggestions it should produce:

- Vague language → "Your prompt says 'be creative' without constraints. Consider adding: 'Generate exactly 3 options, each under 100 words.'"
- Missing role → "Add a role definition like: 'You are a senior data analyst specializing in...'"
- Contradictions → "Your prompt contains conflicting instructions: 'be concise' and 'provide detailed explanations'. Remove one or clarify when each applies."
- Missing output format → "Specify the expected output format, e.g.: 'Return your analysis as a JSON object with keys: summary, findings, recommendations.'"
- Ambiguous pronouns → "Replace 'it' with the specific noun it refers to for clarity."

Keep it rule-based, no model calls. Each finding in the list should map to one concrete suggestion string.
```

### Phase 4C — Comprehensive Tests

```
Read agents/guardrails.py (updated in Phases 4A and 4B).

Write tests/test_guardrails.py with comprehensive tests covering:

1. Each existing check (vague language, missing format, missing role, too short, empty prompt)
   - At least one prompt that SHOULD trigger each check
   - At least one prompt that should NOT trigger each check

2. Each new check from Phase 4A:
   - Contradictory instructions: test with conflicting pairs and with non-conflicting instructions
   - Token estimation: test with a prompt that exceeds GPT's 128k limit but not Claude's 200k
   - Ambiguous pronouns: test with unclear "it"/"this" and with properly referenced pronouns
   - Missing constraints: test creative prompt without length/format limits vs one with constraints
   - Duplicate instructions: test with near-identical repeated sentences vs legitimately similar but distinct instructions

3. suggest_fixes(): verify it returns a list of actionable strings, one per finding

4. Backward compatibility: verify check_prompt() still returns {"passed": bool, "warnings": list, "errors": list}

5. Severity levels: verify findings have correct severity tags ("info", "warning", "error")
```

### Files to Upload to Claude Code
- `agents/guardrails.py`
- `agents/format_profiles.py`
- `PLAN.md`

---

## Phase 5 — Frontend Chat Interface

### Goal
Evolve the frontend from a single input/output flow to a multi-turn chat interface that supports the full agent pipeline.

> **Note:** This phase is split into 3 sub-prompts (5A, 5B, 5C) to keep each generation small enough to avoid Claude Code stream idle timeouts. Run them in order.

### Phase 5A — Chat Interface and API Integration

```
Read the following files:
- ARCHITECTURE.md
- PLAN.md
- The entire existing frontend directory (all HTML, CSS, JS files)
- orchestrator/orchestrator.py (to understand the response format)

The backend orchestrator returns responses with this structure:
{
    "session_id": "...",
    "state": "initial|awaiting_context|review|iterating|accepted",
    "message": "...",         // chat message to display
    "prompt_draft": "...",    // current prompt (if any)
    "scores": {},             // critic scores (if in review state)
    "sample_output": "..."   // live test result (if available)
}

Build the core chat interface:

1. Chat Interface
   - Replace the current single input/output with a scrolling chat window
   - User messages appear on the right, assistant messages on the left
   - The model selection (Claude/GPT/Gemini) should be selectable at the start and locked once the conversation begins
   - Input area at the bottom with send button
   - Session persists via session_id stored in the frontend state

2. API Integration
   - All messages go to the same endpoint: POST /optimize (or whatever the current endpoint is)
   - Request body: { session_id, message, target_model }
   - First message creates a new session (no session_id sent), subsequent messages include the session_id from the first response
   - Handle loading states — show a typing indicator while waiting for the orchestrator response

3. Action Buttons
   - After the prompt draft is shown: "Looks good, evaluate it" (triggers review state) and "I want to make changes" (triggers iterating state)
   - After critic evaluation: "Accept this prompt" and "Refine further"
   - "Accept this prompt" should display the final prompt in a clean, copyable format and offer a "Copy to clipboard" button
   - "Start over" button always visible to reset the session

Keep the existing styling approach and design language. This should feel like an evolution of the current UI, not a complete redesign. If the current frontend uses a framework (React, Vue, etc.), continue using it. If it's vanilla JS/HTML, keep it vanilla.
```

### Phase 5B — Rich Message Rendering

```
Read the following files:
- The frontend files updated in Phase 5A
- orchestrator/orchestrator.py (for response format reference)

Add rich rendering for different message types in the chat interface:

1. Regular chat messages render as normal text bubbles
2. Prompt drafts render in a distinct card/code block with syntax highlighting appropriate to the format (XML highlighting for Claude prompts, markdown rendering for GPT prompts)
3. Context Scout suggestions render as a checklist the user can reference
4. Clarifier questions render as clearly numbered items
5. Critic scores render as a visual scorecard (small bar charts or colored indicators for each dimension)
6. Sample output (from live test) renders in a collapsible section with a distinct background
7. Guardrail warnings render as colored banners (yellow for warnings, red for errors) above the prompt draft

Keep the existing styling approach. Each message type should be visually distinct but feel cohesive within the chat UI.
```

### Phase 5C — Diff View

```
Read the frontend files updated in Phases 5A and 5B.

Add a prompt diff view:

1. When the prompt is refined (version 2+), show a toggle button: "Show changes"
2. When toggled, display a side-by-side or inline diff highlighting additions (green) and removals (red)
3. Use a lightweight JS diff library — jsdiff (npm: diff) is a good option
4. Store prompt versions in the frontend state for comparison

This should integrate naturally with the existing prompt draft cards from Phase 5B.
```

### Files to Upload to Claude Code
- `ARCHITECTURE.md`
- `PLAN.md`
- Entire frontend directory
- `orchestrator/orchestrator.py` (for response format reference)

---

## Phase 6 — Critic Agent with Live Testing

### Goal
Implement the evaluation pipeline that scores prompts and runs them against the target model to show sample output.

### Claude Code Prompt

```
Read the following files:
- ARCHITECTURE.md
- PLAN.md
- agents/critic.py (system prompt and message builder from Phase 1)
- orchestrator/orchestrator.py
- orchestrator/agent_router.py
- prompt_db/retrieve.py
- The existing Bedrock client/utility code

Build the live testing pipeline:

1. Create `orchestrator/live_test.py`
   - Define `run_live_test(prompt: str, target_model: str) -> dict` that:
     - Takes the final assembled prompt (system prompt + user prompt + few-shot examples)
     - Calls the target model via Bedrock with the prompt
     - Uses a generic test input appropriate to the task (e.g., for a data analysis prompt, send a brief sample dataset description; for a code prompt, send a brief code request)
     - Returns { "output": str, "latency_ms": int, "tokens_used": int }
   - The test input should be generated by a brief model call: "Given this prompt, generate a realistic but brief test input that would exercise its main functionality. Keep it under 200 words."
   - Include error handling — if the live test fails (timeout, model error), return a graceful failure message rather than crashing

2. Modify `orchestrator/orchestrator.py` in the "review" state:
   - Call the Critic Agent to get scores
   - Call run_live_test() in parallel with the Critic
   - Merge results: scores + improvement suggestions + sample output
   - Include in the response:
     - scores: { clarity, specificity, completeness, constraint_coverage, hallucination_risk, overall }
     - suggestions: list of specific improvements
     - sample_output: the live test result
     - sample_input: what was sent to the model (so the user understands the test)

3. Modify the Critic Agent call in agent_router.py:
   - Before calling the Critic, retrieve 2-3 high-scoring reference prompts for the same task category from the prompt DB
   - Include them in the Critic's context so it can compare

4. Add a configurable flag to enable/disable live testing (some users may not want to spend the tokens). Default to enabled.
   - If disabled, only return critic scores without sample output
   - The frontend should show a "Test this prompt" button that explicitly triggers the test if auto-testing is off

Ensure the Lambda timeout is sufficient for the live test (it runs a full model inference). If the existing timeout is under 60 seconds, note that it needs to be increased.
```

### Files to Upload to Claude Code
- `ARCHITECTURE.md`
- `PLAN.md`
- `agents/critic.py`
- `orchestrator/` directory
- `prompt_db/retrieve.py`
- Existing Bedrock client code

---

## Phase 7 — Prompt Ingestion Loop

### Goal
Close the flywheel: accepted prompts with high quality scores are automatically ingested into the prompt database, growing it over time.

### Claude Code Prompt

```
Read the following files:
- PLAN.md
- orchestrator/orchestrator.py
- prompt_db/ingest.py
- prompt_db/schema.py
- prompt_db/retrieve.py
- agents/critic.py

Build the automatic prompt ingestion pipeline:

1. Modify `orchestrator/orchestrator.py` in the "accepted" state:
   - When the user accepts a prompt AND the critic's overall score is >= 0.8 (make this threshold configurable via environment variable INGEST_THRESHOLD, default 0.8):
     - Construct a PromptRecord from the session data:
       - task_category and subcategory from the session
       - target_model from the session
       - format from the format profile
       - techniques: have the critic identify which techniques are used (add this to the critic's output parsing in Phase 1's critic.py)
       - system_prompt and user_prompt_template from the final draft
       - few_shot_examples from the Few-Shot Generator output (if any)
       - quality_score from the critic's overall score
       - source: "user_generated"
     - Call ingest_single_prompt() to add it to the vector store
     - Add a flag to the session: `ingested: true`
   - If the score is below threshold, still accept the prompt but don't ingest. Optionally note to the user: "Your prompt has been saved. Prompts scoring above X are added to our reference library."

2. Add a deduplication check in `prompt_db/ingest.py`:
   - Before ingesting, retrieve the top 3 most similar prompts from the DB
   - If any have a cosine similarity > 0.95, skip ingestion (it's essentially a duplicate)
   - Log skipped ingestions for monitoring

3. Add a simple admin utility `prompt_db/admin.py`:
   - `list_prompts(task_category: str = None, source: str = None) -> list[PromptRecord]` to browse the database
   - `remove_prompt(prompt_id: str)` to delete bad prompts
   - `update_score(prompt_id: str, new_score: float)` to manually adjust scores
   - Make this runnable as a CLI: `python -m prompt_db.admin --list --category data_analysis`

This is the growth flywheel — as more users accept high-quality prompts, the reference library improves, which makes future prompts better.
```

### Files to Upload to Claude Code
- `PLAN.md`
- `orchestrator/orchestrator.py`
- `prompt_db/` directory
- `agents/critic.py`

---

## Implementation Order Summary

| Phase | What | Depends On | Estimated Effort | Status |
|-------|------|-----------|-----------------|--------|
| 1 | Agent system prompts + format profiles | Phase 0 (prep) | Light — mostly writing prompts | ✅ Complete |
| 2 | Conversation orchestrator | Phase 1 | Heavy — core architecture change | ✅ Complete |
| 3 | Prompt database RAG swap | Phase 1, 2 | Medium — adapting existing infra | ✅ Complete |
| 4A | Guardrails — severity levels + new checks | Phase 1 | Light | ✅ Complete |
| 4B | Guardrails — suggest_fixes() | Phase 4A | Light | ✅ Complete |
| 4C | Guardrails — comprehensive tests | Phase 4B | Light | ✅ Complete |
| 5A | Frontend — chat interface + API + action buttons | Phase 2 | Medium | Not started |
| 5B | Frontend — rich message rendering | Phase 5A | Medium | Not started |
| 5C | Frontend — diff view | Phase 5B | Light | Not started |
| 6 | Critic + live testing | Phase 2, 3 | Medium | Not started |
| 7 | Prompt ingestion loop | Phase 3, 6 | Light | Not started |

Phases 4 and 5 can run in parallel with Phase 3. Phase 6 needs both 2 and 3 complete. Phase 7 is a quick add-on after 6.

> **Why sub-phases?** Phases 4 and 5 were split into smaller prompts (4A/4B/4C, 5A/5B/5C) because the original single-prompt versions caused Claude Code stream idle timeouts — the output was too large for a single generation. Each sub-prompt should be run sequentially within its phase.

---

## POGO ideas/possible additions

- optmize LLM usage rates
    - token usage
    - model usage
- expand rag db of prompts to include:
    - advanced one-shot examples
    - multi-shot examples

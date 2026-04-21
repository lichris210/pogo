"""Agent router — task classification, Bedrock invocation, and parallel execution.

Reuses the cached Bedrock client pattern from the existing ``lambda/handler.py``.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Model IDs (same as existing handler for consistency)
ARCHITECT_MODEL_ID = os.environ.get(
    "POGO_AGENT_MODEL_ID",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
)
LIGHT_MODEL_ID = os.environ.get("POGO_LIGHT_MODEL_ID", ARCHITECT_MODEL_ID)
_TARGET_MODEL_ENV_KEYS = {
    "claude": "POGO_TARGET_MODEL_ID_CLAUDE",
    "gpt": "POGO_TARGET_MODEL_ID_GPT",
    "gemini": "POGO_TARGET_MODEL_ID_GEMINI",
}

_bedrock = None


def _get_bedrock():
    """Return a cached ``bedrock-runtime`` client."""
    global _bedrock
    if _bedrock is None:
        import boto3

        _bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    return _bedrock


# ---------------------------------------------------------------------------
# Task classification (keyword-based, upgradeable to model call later)
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "data_analysis": [
        "data", "csv", "sql", "database", "analytics", "dashboard", "metrics",
        "statistics", "trend", "churn", "forecast", "report", "visualization",
        "chart", "graph", "pandas", "dataframe", "query", "aggregate",
    ],
    "code_generation": [
        "code", "function", "script", "program", "implement", "debug", "api",
        "endpoint", "class", "algorithm", "refactor", "test", "unit test",
        "python", "javascript", "typescript", "rust", "java", "golang",
    ],
    "writing": [
        "essay", "blog", "article", "copy", "email", "letter",
        "documentation", "readme", "technical writing", "content",
    ],
    "creative": [
        "story", "poem", "creative", "fiction", "narrative", "dialogue",
        "screenplay", "song", "lyrics",
    ],
    "web_development": [
        "website", "frontend", "backend", "html", "css", "react", "vue",
        "angular", "next.js", "web app", "landing page", "ui", "ux",
    ],
    "research": [
        "research", "literature review", "survey", "compare", "evaluate",
        "pros and cons", "analysis of", "investigate", "study",
    ],
}


def classify_task(user_intent: str) -> str:
    """Categorise *user_intent* into a task type using keyword matching.

    Args:
        user_intent: The user's raw task description.

    Returns:
        One of ``data_analysis``, ``code_generation``, ``writing``,
        ``creative``, ``web_development``, ``research``, or ``general``.
    """
    text = user_intent.lower()
    scores: dict[str, int] = {}
    for category, keywords in _CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", text))
        if score:
            scores[category] = score
    if not scores:
        return "general"
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Bedrock invocation
# ---------------------------------------------------------------------------

def resolve_target_model_id(target_model: str) -> str:
    """Return the Bedrock model ID configured for a target-model family.

    Falls back to the orchestrator's default Anthropic model so local
    development still works when family-specific overrides are unset.
    """
    env_key = _TARGET_MODEL_ENV_KEYS.get(target_model)
    if not env_key:
        return ARCHITECT_MODEL_ID
    return os.environ.get(env_key, "").strip() or ARCHITECT_MODEL_ID


def invoke_agent_raw(
    agent_name: str,
    messages: list[dict],
    system: str,
    model_id: str | None = None,
    max_tokens: int = 2000,
) -> dict:
    """Call Bedrock and return assistant text plus usage metadata."""
    bedrock = _get_bedrock()
    mid = model_id or ARCHITECT_MODEL_ID

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }

    response = bedrock.invoke_model(
        modelId=mid,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    usage = _normalise_usage(result.get("usage"))
    return {
        "agent_name": agent_name,
        "model_id": mid,
        "text": result["content"][0]["text"],
        "usage": usage,
    }


def invoke_agent(
    agent_name: str,
    messages: list[dict],
    system: str,
    model_id: str | None = None,
    max_tokens: int = 2000,
) -> str:
    """Call Bedrock with a messages array and return the assistant's text.

    Args:
        agent_name: Label for logging (e.g. ``"prompt_architect"``).
        messages: Messages list from an agent's ``build_messages()``.
        system: The system prompt string.
        model_id: Bedrock model ID. Defaults to :data:`ARCHITECT_MODEL_ID`.
        max_tokens: Response token limit.

    Returns:
        The model's text response.
    """
    return invoke_agent_raw(
        agent_name=agent_name,
        messages=messages,
        system=system,
        model_id=model_id,
        max_tokens=max_tokens,
    )["text"]


def invoke_parallel(agent_configs: list[dict]) -> list[str]:
    """Run multiple agent calls concurrently using threads.

    Args:
        agent_configs: A list of dicts, each with keys:
            - ``agent_name`` (str)
            - ``messages`` (list[dict])
            - ``system`` (str)
            - ``model_id`` (str, optional)
            - ``max_tokens`` (int, optional)

    Returns:
        A list of response strings **in the same order** as *agent_configs*.
    """
    results: dict[int, str] = {}

    def _call(idx: int, cfg: dict) -> tuple[int, str]:
        text = invoke_agent(
            agent_name=cfg["agent_name"],
            messages=cfg["messages"],
            system=cfg["system"],
            model_id=cfg.get("model_id"),
            max_tokens=cfg.get("max_tokens", 2000),
        )
        return idx, text

    with ThreadPoolExecutor(max_workers=len(agent_configs)) as pool:
        futures = {
            pool.submit(_call, i, cfg): i
            for i, cfg in enumerate(agent_configs)
        }
        for future in as_completed(futures):
            idx, text = future.result()
            results[idx] = text

    return [results[i] for i in range(len(agent_configs))]


# ---------------------------------------------------------------------------
# Prompt-DB retrieval helpers
# ---------------------------------------------------------------------------
#
# The orchestrator queries the prompt DB on behalf of each agent using the
# classified task category — never the user's raw input.  These helpers wrap
# the prompt_db module so the orchestrator has one place to pull references
# and so failures (e.g. empty DB in local tests) degrade gracefully.

def fetch_reference_prompts(
    task_category: str,
    target_model: str,
    k: int = 3,
) -> list[str]:
    """Return up to *k* reference-prompt strings for injection into an agent.

    Each string combines the record's system prompt and user template so it
    can be dropped straight into the Prompt Architect / Critic
    ``reference_prompts`` parameter. Returns ``[]`` on any failure.
    """
    try:
        from prompt_db.retrieve import retrieve_reference_prompts
    except Exception as e:  # pragma: no cover — import failures
        print(f"agent_router.fetch_reference_prompts: import failed ({e})")
        return []

    try:
        records = retrieve_reference_prompts(task_category, target_model, k=k)
    except Exception as e:
        print(f"agent_router.fetch_reference_prompts: retrieval failed ({e})")
        return []

    out: list[str] = []
    for r in records:
        parts = []
        header = (
            f"[id={r.id} category={r.task_category}/{r.subcategory} "
            f"techniques={','.join(r.techniques) or 'none'} "
            f"score={r.quality_score:.2f}]"
        )
        parts.append(header)
        if r.system_prompt:
            parts.append(f"System:\n{r.system_prompt}")
        if r.user_prompt_template:
            parts.append(f"User:\n{r.user_prompt_template}")
        out.append("\n".join(parts))
    return out


def fetch_fewshot_examples(
    task_category: str,
    target_model: str,
    k: int = 2,
) -> list[dict]:
    """Return up to *k* few-shot example dicts for the given task."""
    try:
        from prompt_db.retrieve import retrieve_few_shot_examples
    except Exception as e:  # pragma: no cover
        print(f"agent_router.fetch_fewshot_examples: import failed ({e})")
        return []

    try:
        return retrieve_few_shot_examples(task_category, target_model, k=k)
    except Exception as e:
        print(f"agent_router.fetch_fewshot_examples: retrieval failed ({e})")
        return []


def run_critic_review(
    final_prompt: str,
    task_category: str,
    format_profile: dict,
    target_model: str,
    *,
    k: int = 3,
) -> dict:
    """Invoke the Critic with retrieved reference prompts for comparison."""
    from agents import critic

    reference_prompts = fetch_reference_prompts(task_category, target_model, k=k)
    messages, system = critic.build_messages(
        final_prompt=final_prompt,
        task_category=task_category,
        format_profile=format_profile,
        reference_prompts=reference_prompts,
    )
    response = invoke_agent("critic", messages, system)
    return {
        "response": response,
        "scores": critic.parse_scores(response),
        "suggestions": critic.parse_suggestions(response),
        "reference_prompts": reference_prompts,
    }


def _normalise_usage(raw_usage: dict | None) -> dict:
    """Return a consistent usage dict even when the provider omits fields."""
    if not isinstance(raw_usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    input_tokens = int(raw_usage.get("input_tokens", 0) or 0)
    output_tokens = int(raw_usage.get("output_tokens", 0) or 0)
    total_tokens = int(raw_usage.get("total_tokens", 0) or 0)
    if not total_tokens:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

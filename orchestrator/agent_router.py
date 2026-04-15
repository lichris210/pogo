"""Agent router — task classification, Bedrock invocation, and parallel execution.

Reuses the cached Bedrock client pattern from the existing ``lambda/handler.py``.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

# Model IDs (same as existing handler for consistency)
ARCHITECT_MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
LIGHT_MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"

_bedrock = None


def _get_bedrock():
    """Return a cached ``bedrock-runtime`` client."""
    global _bedrock
    if _bedrock is None:
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
    return result["content"][0]["text"]


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

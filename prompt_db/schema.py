"""Prompt database record schema.

A :class:`PromptRecord` is the canonical unit stored in the v2 prompt DB.
Each record carries the prompt text plus metadata used for retrieval and
filtering (task category, target model, format, techniques, quality).

The text that is **embedded** is *not* the raw prompt — that would cause
retrieval to match on surface wording. Instead we embed a distilled
descriptor built by :func:`to_embedding_text` that combines the task
category, subcategory, techniques, and a short content summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Canonical vocabularies
# ---------------------------------------------------------------------------

VALID_TASK_CATEGORIES: tuple[str, ...] = (
    "data_analysis",
    "code_generation",
    "writing",
    "web_development",
    "creative",
    "research",
    "general",
)

VALID_TARGET_MODELS: tuple[str, ...] = ("claude", "gpt", "gemini")

VALID_FORMATS: tuple[str, ...] = ("xml", "markdown")

VALID_SOURCES: tuple[str, ...] = ("curated", "user_generated")


# ---------------------------------------------------------------------------
# PromptRecord
# ---------------------------------------------------------------------------

@dataclass
class PromptRecord:
    """A single prompt entry in the prompt DB."""

    id: str
    task_category: str
    subcategory: str
    target_model: str
    format: str
    techniques: list[str]
    system_prompt: str
    user_prompt_template: str
    few_shot_examples: list[dict] = field(default_factory=list)
    quality_score: float = 0.0
    source: str = "curated"
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`ValueError` if any field is invalid."""
        if not self.id:
            raise ValueError("PromptRecord.id is required")
        if self.task_category not in VALID_TASK_CATEGORIES:
            raise ValueError(
                f"task_category must be one of {VALID_TASK_CATEGORIES}, "
                f"got {self.task_category!r}"
            )
        if self.target_model not in VALID_TARGET_MODELS:
            raise ValueError(
                f"target_model must be one of {VALID_TARGET_MODELS}, "
                f"got {self.target_model!r}"
            )
        if self.format not in VALID_FORMATS:
            raise ValueError(
                f"format must be one of {VALID_FORMATS}, got {self.format!r}"
            )
        if self.source not in VALID_SOURCES:
            raise ValueError(
                f"source must be one of {VALID_SOURCES}, got {self.source!r}"
            )
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError(
                f"quality_score must be in [0.0, 1.0], got {self.quality_score}"
            )
        if not isinstance(self.techniques, list):
            raise ValueError("techniques must be a list[str]")
        if not isinstance(self.few_shot_examples, list):
            raise ValueError("few_shot_examples must be a list[dict]")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PromptRecord":
        """Construct a :class:`PromptRecord` from a dict."""
        return cls(
            id=d["id"],
            task_category=d["task_category"],
            subcategory=d.get("subcategory", ""),
            target_model=d["target_model"],
            format=d.get("format", "markdown"),
            techniques=list(d.get("techniques", [])),
            system_prompt=d.get("system_prompt", ""),
            user_prompt_template=d.get("user_prompt_template", ""),
            few_shot_examples=list(d.get("few_shot_examples", [])),
            quality_score=float(d.get("quality_score", 0.0)),
            source=d.get("source", "curated"),
            created_at=d.get("created_at", ""),
        )


# ---------------------------------------------------------------------------
# Embedding text
# ---------------------------------------------------------------------------

def to_embedding_text(record: PromptRecord) -> str:
    """Build the text to embed for *record*.

    The embedding text summarises *what* the prompt is for, not *how*
    it is worded. This makes retrieval match on task intent rather than
    on shared vocabulary between the query and the prompt body.

    Components:
        - Task category and subcategory
        - Target model family
        - Techniques used
        - A short content summary (first sentence of the system prompt
          plus the first line of the user prompt template)
    """
    summary = _summarise_prompt_content(
        record.system_prompt, record.user_prompt_template
    )

    techniques = ", ".join(record.techniques) if record.techniques else "(none)"
    subcat = record.subcategory or record.task_category

    return (
        f"Task category: {record.task_category}. "
        f"Subcategory: {subcat}. "
        f"Target model: {record.target_model}. "
        f"Techniques: {techniques}. "
        f"Summary: {summary}"
    ).strip()


def _summarise_prompt_content(system_prompt: str, user_template: str) -> str:
    """Short, lossy summary of the prompt body for embedding."""
    sp = (system_prompt or "").strip()
    ut = (user_template or "").strip()

    # First sentence of the system prompt (up to 200 chars).
    sys_summary = ""
    if sp:
        sentence_end = min(
            (sp.find(m) for m in (". ", "\n", "?") if sp.find(m) != -1),
            default=len(sp),
        )
        sys_summary = sp[: max(sentence_end, 1)].strip()[:200]

    # First non-empty line of the user template (up to 200 chars).
    user_summary = ""
    if ut:
        for line in ut.splitlines():
            line = line.strip()
            if line:
                user_summary = line[:200]
                break

    parts = [p for p in (sys_summary, user_summary) if p]
    return " | ".join(parts) if parts else "(empty)"

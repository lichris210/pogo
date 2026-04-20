"""POGO v2 prompt database.

Stores curated and user-generated prompts as vector records so the
orchestrator can retrieve reference prompts and few-shot examples by
task category on behalf of the agent pipeline.

Entry points::

    from prompt_db.ingest import ingest_seed_data, ingest_single_prompt
    from prompt_db.retrieve import (
        retrieve_reference_prompts,
        retrieve_few_shot_examples,
    )
    from prompt_db.schema import PromptRecord, to_embedding_text
"""

from prompt_db.schema import PromptRecord, to_embedding_text  # noqa: F401

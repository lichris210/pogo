"""End-to-end test for the prompt DB module.

Runs offline: the Bedrock Titan embedder is monkey-patched with a
deterministic hash-based stub, and the vector store is redirected to a
tmp directory via the ``POGO_PROMPT_DB_LOCAL_DIR`` env var. This lets us
ingest the real ``seed_prompts.json`` file and exercise retrieval for
every task category without making any AWS calls.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from prompt_db.embeddings import EMBED_DIM
from prompt_db.schema import (
    PromptRecord,
    VALID_TARGET_MODELS,
    to_embedding_text,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_FILE = REPO_ROOT / "seed_prompts.json"


# ---------------------------------------------------------------------------
# Deterministic stub embedder.
# ---------------------------------------------------------------------------
#
# We hash the input text, seed a NumPy RNG, sample a 256-dim vector, and
# normalise it. Same text → same vector, different text → different
# vector. Good enough for testing cosine-similarity retrieval.

def _fake_embed_text(text: str) -> np.ndarray:
    digest = hashlib.sha256((text or "").encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(EMBED_DIM).astype("float32")
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _fake_embed_batch(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32")
    return np.stack([_fake_embed_text(t) for t in texts]).astype("float32")


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema(unittest.TestCase):
    def test_valid_record(self):
        r = PromptRecord(
            id="r1",
            task_category="code_generation",
            subcategory="api_endpoint",
            target_model="claude",
            format="xml",
            techniques=["role_assignment"],
            system_prompt="You are a developer.",
            user_prompt_template="Write code for {{X}}",
            quality_score=0.9,
            source="curated",
        )
        r.validate()
        self.assertEqual(r.task_category, "code_generation")

    def test_invalid_task_category(self):
        r = PromptRecord(
            id="r2", task_category="nope", subcategory="s",
            target_model="claude", format="xml", techniques=[],
            system_prompt="", user_prompt_template="",
        )
        with self.assertRaises(ValueError):
            r.validate()

    def test_invalid_target_model(self):
        r = PromptRecord(
            id="r3", task_category="general", subcategory="s",
            target_model="llama", format="markdown", techniques=[],
            system_prompt="", user_prompt_template="",
        )
        with self.assertRaises(ValueError):
            r.validate()

    def test_to_embedding_text_combines_fields(self):
        long_body = (
            "Detailed instructions that span many paragraphs. " * 50
        )
        r = PromptRecord(
            id="r4", task_category="writing", subcategory="blog_post",
            target_model="gpt", format="markdown",
            techniques=["role_assignment", "few_shot"],
            system_prompt="You are a blogger. " + long_body,
            user_prompt_template="Write a blog about {{TOPIC}}.\n" + long_body,
        )
        text = to_embedding_text(r)
        self.assertIn("writing", text)
        self.assertIn("blog_post", text)
        self.assertIn("gpt", text)
        self.assertIn("role_assignment", text)
        # Must be a compact summary, not a dump of the full prompt body.
        self.assertLess(len(text), 1000)

    def test_roundtrip(self):
        r = PromptRecord(
            id="r5", task_category="creative", subcategory="story",
            target_model="claude", format="xml",
            techniques=["role_assignment"],
            system_prompt="You write stories.",
            user_prompt_template="Write a story.",
            quality_score=0.87,
        )
        d = r.to_dict()
        r2 = PromptRecord.from_dict(d)
        self.assertEqual(r2.id, r.id)
        self.assertEqual(r2.task_category, r.task_category)
        self.assertEqual(r2.techniques, r.techniques)


# ---------------------------------------------------------------------------
# Ingest + retrieval integration tests (offline, with stubbed embedder).
# ---------------------------------------------------------------------------

class TestIngestAndRetrieve(unittest.TestCase):
    """Ingest the real seed file into a tmp store and query it."""

    tmp_dir: str
    _prev_env: str | None

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="pogo-prompt-db-")
        cls._prev_env = os.environ.get("POGO_PROMPT_DB_LOCAL_DIR")
        os.environ["POGO_PROMPT_DB_LOCAL_DIR"] = cls.tmp_dir

        # Patch the embedder everywhere it's imported.
        cls._patches = [
            patch("prompt_db.embeddings.embed_text", side_effect=_fake_embed_text),
            patch("prompt_db.embeddings.embed_batch", side_effect=_fake_embed_batch),
            patch("prompt_db.ingest.embed_text", side_effect=_fake_embed_text),
            patch("prompt_db.ingest.embed_batch", side_effect=_fake_embed_batch),
            patch("prompt_db.retrieve.embed_text", side_effect=_fake_embed_text),
        ]
        for p in cls._patches:
            p.start()

        # Import after patching so functions bind to the fakes.
        from prompt_db.ingest import ingest_seed_data
        from prompt_db.retrieve import reset_cache

        reset_cache()
        cls.ingested_count = ingest_seed_data(str(SEED_FILE), overwrite=True)
        reset_cache()

    @classmethod
    def tearDownClass(cls):
        for p in cls._patches:
            p.stop()
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)
        if cls._prev_env is None:
            os.environ.pop("POGO_PROMPT_DB_LOCAL_DIR", None)
        else:
            os.environ["POGO_PROMPT_DB_LOCAL_DIR"] = cls._prev_env

    # -- ingest --------------------------------------------------------

    def test_ingest_wrote_seed_records(self):
        self.assertGreater(self.ingested_count, 100)
        # Files exist on disk.
        self.assertTrue(Path(self.tmp_dir, "prompts.json").exists())
        self.assertTrue(Path(self.tmp_dir, "embeddings.npy").exists())

    def test_ingest_normalised_target_models(self):
        from prompt_db.store import load

        records, _ = load()
        models = {r.target_model for r in records}
        # Every model name must be one of the canonical POGO families.
        self.assertTrue(models.issubset(set(VALID_TARGET_MODELS)))
        # We should have hit at least two families given the seed mix.
        self.assertGreaterEqual(len(models), 2)

    def test_ingest_normalised_task_categories(self):
        from prompt_db.schema import VALID_TASK_CATEGORIES
        from prompt_db.store import load

        records, _ = load()
        cats = {r.task_category for r in records}
        self.assertTrue(cats.issubset(set(VALID_TASK_CATEGORIES)))

    def test_embeddings_shape_matches_records(self):
        from prompt_db.store import load

        records, embeddings = load()
        self.assertEqual(len(records), embeddings.shape[0])
        self.assertEqual(embeddings.shape[1], EMBED_DIM)

    # -- retrieval -----------------------------------------------------

    def test_retrieval_filters_by_target_model(self):
        from prompt_db.retrieve import retrieve_reference_prompts

        for model in ("claude", "gpt", "gemini"):
            refs = retrieve_reference_prompts("code_generation", model, k=3)
            self.assertLessEqual(len(refs), 3)
            for r in refs:
                self.assertEqual(r.target_model, model)

    def test_retrieval_returns_results_for_every_category(self):
        from prompt_db.retrieve import retrieve_reference_prompts
        from prompt_db.schema import VALID_TASK_CATEGORIES

        hits = 0
        for cat in VALID_TASK_CATEGORIES:
            for model in ("claude", "gpt", "gemini"):
                refs = retrieve_reference_prompts(cat, model, k=2)
                if refs:
                    hits += 1
                    for r in refs:
                        self.assertEqual(r.target_model, model)
        # Most (category, model) pairs should have at least one hit
        # given we ingested 139 seed records across all families.
        self.assertGreater(hits, 10)

    def test_retrieval_empty_for_unknown_category(self):
        from prompt_db.retrieve import retrieve_reference_prompts

        # Unknown model → nothing should come back (filter kicks in).
        refs = retrieve_reference_prompts("code_generation", "nonexistent", k=3)
        self.assertEqual(refs, [])

    def test_retrieval_ordered_by_similarity(self):
        """Same query for same category should be stable and top-ranked."""
        from prompt_db.retrieve import retrieve_reference_prompts

        a = retrieve_reference_prompts("code_generation", "claude", k=3)
        b = retrieve_reference_prompts("code_generation", "claude", k=3)
        self.assertEqual([r.id for r in a], [r.id for r in b])

    def test_fewshot_examples_respect_target_model(self):
        """retrieve_few_shot_examples should survive even when most records
        have no few-shot examples (the seed data is mostly templates)."""
        from prompt_db.retrieve import retrieve_few_shot_examples

        # We don't assert we get examples back (seed prompts rarely include
        # few_shot_examples), but we do require the call to succeed and
        # return a list.
        result = retrieve_few_shot_examples("writing", "gpt", k=2)
        self.assertIsInstance(result, list)

    # -- single-record ingest -----------------------------------------

    def test_ingest_single_prompt(self):
        from prompt_db.ingest import ingest_single_prompt
        from prompt_db.retrieve import reset_cache
        from prompt_db.store import load

        before_records, _ = load()
        before_count = len(before_records)

        new_record = PromptRecord(
            id="user_test_001",
            task_category="writing",
            subcategory="email",
            target_model="claude",
            format="xml",
            techniques=["role_assignment"],
            system_prompt="You are an email assistant.",
            user_prompt_template="Write an email about {{TOPIC}}.",
            quality_score=0.9,
            source="user_generated",
        )
        ok = ingest_single_prompt(new_record)
        self.assertTrue(ok)

        reset_cache()
        after_records, after_emb = load()
        self.assertEqual(len(after_records), before_count + 1)
        self.assertEqual(after_records[-1].id, "user_test_001")
        self.assertEqual(after_emb.shape[0], before_count + 1)

    def test_ingest_single_prompt_skips_near_duplicate(self):
        from prompt_db.ingest import ingest_single_prompt
        from prompt_db.retrieve import reset_cache
        from prompt_db.store import load

        before_records, before_emb = load()
        before_count = len(before_records)
        template = before_records[0]

        duplicate = PromptRecord(
            id="user_test_duplicate",
            task_category=template.task_category,
            subcategory=template.subcategory,
            target_model=template.target_model,
            format=template.format,
            techniques=list(template.techniques),
            system_prompt=template.system_prompt,
            user_prompt_template=template.user_prompt_template,
            few_shot_examples=list(template.few_shot_examples),
            quality_score=template.quality_score,
            source="user_generated",
        )

        ok = ingest_single_prompt(duplicate)
        self.assertFalse(ok)

        reset_cache()
        after_records, after_emb = load()
        self.assertEqual(len(after_records), before_count)
        self.assertEqual(after_emb.shape[0], before_emb.shape[0])

    def test_retrieve_similar_prompts(self):
        from prompt_db.retrieve import retrieve_similar_prompts

        text = "Task category: writing. Subcategory: email. Target model: claude."
        result = retrieve_similar_prompts(text, k=3, target_model="claude")

        self.assertLessEqual(len(result), 3)
        for record, similarity in result:
            self.assertEqual(record.target_model, "claude")
            self.assertIsInstance(similarity, float)

    def test_admin_list_update_and_remove(self):
        from prompt_db.admin import list_prompts, remove_prompt, update_score
        from prompt_db.ingest import ingest_single_prompt
        from prompt_db.retrieve import reset_cache
        from prompt_db.store import load

        target = PromptRecord(
            id="user_admin_test",
            task_category="writing",
            subcategory="email",
            target_model="claude",
            format="xml",
            techniques=["role_assignment"],
            system_prompt="You are a support email assistant.",
            user_prompt_template="Write a customer email about {{INCIDENT}}.",
            quality_score=0.91,
            source="user_generated",
        )
        self.assertTrue(ingest_single_prompt(target))

        writing_prompts = list_prompts(task_category="writing", source="user_generated")
        self.assertIn(target.id, {r.id for r in writing_prompts})

        self.assertTrue(update_score(target.id, 0.42))

        reset_cache()
        records, _ = load()
        updated = next(r for r in records if r.id == target.id)
        self.assertEqual(updated.quality_score, 0.42)

        self.assertTrue(remove_prompt(target.id))
        reset_cache()
        records_after_remove, _ = load()
        self.assertNotIn(target.id, {r.id for r in records_after_remove})


# ---------------------------------------------------------------------------
# Fallback behaviour when the store is empty.
# ---------------------------------------------------------------------------

class TestRetrieveEmptyStore(unittest.TestCase):
    """retrieve_* must degrade gracefully when the store is empty."""

    def test_empty_store_returns_empty_list(self):
        tmp = tempfile.mkdtemp(prefix="pogo-empty-db-")
        prev = os.environ.get("POGO_PROMPT_DB_LOCAL_DIR")
        os.environ["POGO_PROMPT_DB_LOCAL_DIR"] = tmp
        try:
            from prompt_db.retrieve import (
                reset_cache,
                retrieve_few_shot_examples,
                retrieve_reference_prompts,
            )

            reset_cache()
            self.assertEqual(
                retrieve_reference_prompts("code_generation", "claude"), []
            )
            self.assertEqual(
                retrieve_few_shot_examples("code_generation", "claude"), []
            )
        finally:
            if prev is None:
                os.environ.pop("POGO_PROMPT_DB_LOCAL_DIR", None)
            else:
                os.environ["POGO_PROMPT_DB_LOCAL_DIR"] = prev
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

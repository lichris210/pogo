"""Tests for the Anthropic client wrapper introduced in Phase B3."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestAnthropicClientWrapper(unittest.TestCase):
    def setUp(self) -> None:
        from orchestrator import anthropic_client

        anthropic_client.reset_client_for_testing()
        self.module = anthropic_client

    def tearDown(self) -> None:
        self.module.reset_client_for_testing()

    def _patch_env(self, value: str | None):
        env = os.environ.copy()
        if value is None:
            env.pop("ANTHROPIC_API_KEY", None)
        else:
            env["ANTHROPIC_API_KEY"] = value
        return patch.dict(os.environ, env, clear=True)

    def _fake_anthropic_module(self, response):
        """Return a fake ``anthropic`` module whose Anthropic() returns
        a client whose ``messages.create`` returns *response*.
        """
        fake_client = MagicMock()
        fake_client.messages.create.return_value = response
        fake_anthropic_cls = MagicMock(return_value=fake_client)

        class APIError(Exception):
            pass

        fake_module = SimpleNamespace(
            Anthropic=fake_anthropic_cls,
            APIError=APIError,
        )
        return fake_module, fake_client

    def test_happy_path_returns_normalised_dict(self):
        response = SimpleNamespace(
            content=[SimpleNamespace(text="hello world")],
            usage=SimpleNamespace(input_tokens=12, output_tokens=8),
        )
        fake_anthropic, fake_client = self._fake_anthropic_module(response)

        with self._patch_env("sk-ant-test"), patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            result = self.module.create_message(
                model="claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": "hi"}],
                system="be helpful",
                max_tokens=500,
            )

        self.assertEqual(result["text"], "hello world")
        self.assertEqual(result["usage"], {
            "input_tokens": 12,
            "output_tokens": 8,
            "total_tokens": 20,
        })
        fake_client.messages.create.assert_called_once_with(
            model="claude-haiku-4-5-20251001",
            system="be helpful",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=500,
        )

    def test_missing_env_var_raises_clear_error(self):
        with self._patch_env(None):
            with self.assertRaises(self.module.AnthropicConfigError) as cm:
                self.module.create_message(
                    model="claude-haiku-4-5-20251001",
                    messages=[{"role": "user", "content": "hi"}],
                    system="x",
                )
        self.assertIn("ANTHROPIC_API_KEY", str(cm.exception))

    def test_api_error_propagates_to_caller(self):
        response_module, fake_client = self._fake_anthropic_module(None)

        class FakeAPIError(Exception):
            pass

        response_module.APIError = FakeAPIError
        fake_client.messages.create.side_effect = FakeAPIError("rate limited")

        with self._patch_env("sk-ant-test"), patch.dict(sys.modules, {"anthropic": response_module}):
            with self.assertRaises(FakeAPIError):
                self.module.create_message(
                    model="claude-haiku-4-5-20251001",
                    messages=[{"role": "user", "content": "hi"}],
                    system="x",
                )

    def test_client_is_cached_across_calls(self):
        response = SimpleNamespace(
            content=[SimpleNamespace(text="ok")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        fake_anthropic, _ = self._fake_anthropic_module(response)

        with self._patch_env("sk-ant-test"), patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            self.module.create_message(model="m", messages=[], system="s")
            self.module.create_message(model="m", messages=[], system="s")

        self.assertEqual(fake_anthropic.Anthropic.call_count, 1)

    def test_handles_empty_content_gracefully(self):
        response = SimpleNamespace(
            content=[],
            usage=SimpleNamespace(input_tokens=5, output_tokens=0),
        )
        fake_anthropic, _ = self._fake_anthropic_module(response)

        with self._patch_env("sk-ant-test"), patch.dict(sys.modules, {"anthropic": fake_anthropic}):
            result = self.module.create_message(
                model="claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": "hi"}],
                system="x",
            )
        self.assertEqual(result["text"], "")
        self.assertEqual(result["usage"]["total_tokens"], 5)


class TestAgentRouterUsesWrapper(unittest.TestCase):
    """Sanity check that ``invoke_agent_raw`` now goes through the wrapper."""

    def test_invoke_agent_raw_calls_create_message(self):
        from orchestrator import agent_router

        with patch(
            "orchestrator.agent_router.create_message",
            return_value={
                "text": "drafted",
                "usage": {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10},
            },
        ) as mock_create:
            result = agent_router.invoke_agent_raw(
                agent_name="prompt_architect",
                messages=[{"role": "user", "content": "x"}],
                system="be precise",
                model_id="claude-sonnet-4-6",
                max_tokens=512,
            )

        mock_create.assert_called_once_with(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "x"}],
            system="be precise",
            max_tokens=512,
        )
        self.assertEqual(result["text"], "drafted")
        self.assertEqual(result["model_id"], "claude-sonnet-4-6")
        self.assertEqual(result["usage"]["total_tokens"], 10)


if __name__ == "__main__":
    unittest.main()

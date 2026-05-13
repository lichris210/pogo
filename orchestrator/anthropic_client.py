"""Thin wrapper around the Anthropic SDK for all POGO agent calls.

Replaces the Bedrock ``invoke_model`` path. The wrapper centralises:
- API-key lookup (``ANTHROPIC_API_KEY``)
- Client caching across Lambda invocations
- Response normalisation so call sites get the same shape they got from
  the previous Bedrock parsing code (``text`` + ``usage`` dict)
"""

from __future__ import annotations

import os

_client = None


class AnthropicConfigError(RuntimeError):
    """Raised when the Anthropic client cannot be constructed."""


def _get_client():
    """Return a cached :class:`anthropic.Anthropic` client.

    Raises :class:`AnthropicConfigError` with a clear message when
    ``ANTHROPIC_API_KEY`` is missing.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise AnthropicConfigError(
                "ANTHROPIC_API_KEY is not set. Export it in your shell or "
                "configure it as a Lambda env var before invoking agents."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover — import-time failure
            raise AnthropicConfigError(
                "anthropic SDK not installed. Add `anthropic` to "
                "requirements.txt and re-deploy."
            ) from exc
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def reset_client_for_testing() -> None:
    """Drop the cached client. Tests use this to inject mocks."""
    global _client
    _client = None


def create_message(
    *,
    model: str,
    messages: list[dict],
    system: str,
    max_tokens: int = 2000,
) -> dict:
    """Call ``messages.create`` and return a normalised dict.

    The returned dict matches what the previous Bedrock-parsing code in
    ``agent_router.invoke_agent_raw`` produced::

        {"text": "...", "usage": {"input_tokens": N, "output_tokens": M,
                                   "total_tokens": N+M}}

    Anthropic errors (``anthropic.APIError`` subclasses) propagate
    unchanged; callers that need to handle them should catch
    ``anthropic.APIError``.
    """
    client = _get_client()
    response = client.messages.create(
        model=model,
        system=system,
        messages=messages,
        max_tokens=max_tokens,
    )
    text = response.content[0].text if response.content else ""
    input_tokens = int(getattr(response.usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(response.usage, "output_tokens", 0) or 0)
    return {
        "text": text,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }

"""LLM backends for :class:`~pyhealth.autoimprove.LLMProposer`.

Any callable ``llm(system, prompt) -> str`` works as a backend. This module
provides backends for Claude (``pip install "anthropic>=1.15"``), OpenAI and
OpenAI-compatible endpoints such as vLLM (``pip install "openai>=1.45"``), and
Gemini (``pip install "google-genai>=1.0"``).
"""

from __future__ import annotations

import os
import re
import urllib.parse
from collections.abc import Callable
from typing import Any

LLM = Callable[[str, str], str]
"""A backend: takes a system prompt and a user prompt, returns the reply text."""

MIN_ANTHROPIC = (1, 15)
"""Oldest ``anthropic`` release tested with the request :class:`ClaudeLLM` sends."""

MIN_OPENAI = (1, 45)
"""Oldest ``openai`` release with ``max_completion_tokens`` (:class:`OpenAILLM`)."""


class ClaudeLLM:
    """Calls Claude through the Anthropic SDK.

    Credentials come from the environment (``ANTHROPIC_API_KEY`` or an
    ``ant auth login`` profile). Only the prompts built by the proposer are
    sent: the search space, dataset sizes and schema, aggregate label counts
    and validation scores. A failed trial is described by its exception type
    and the code line that raised it, without the message. No patient records
    are sent.

    The request uses the server-side fallback beta and ``output_config``,
    which need ``anthropic>=1.15`` (:data:`MIN_ANTHROPIC`).

    Args:
        model: Claude model id.
        effort: Reasoning effort, ``"low"`` to ``"max"``.
        max_tokens: Upper bound on the reply length.
        fallback: Retry on another model if this one declines the request
            (server-side fallback).
        client: An ``anthropic.Anthropic`` client; created when omitted.

    Examples:
        >>> from pyhealth.autoimprove import ClaudeLLM, LLMProposer
        >>> proposer = LLMProposer(ClaudeLLM())  # doctest: +SKIP
    """

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "high",
        max_tokens: int = 16000,
        fallback: bool = True,
        client: Any = None,
    ):
        if client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    'ClaudeLLM needs the anthropic package: pip install "anthropic>=1.15"'
                ) from e
            found = tuple(int(x) for x in re.findall(r"\d+", anthropic.__version__)[:2])
            if found < MIN_ANTHROPIC:
                raise ImportError(
                    f"ClaudeLLM needs anthropic>=1.15, found {anthropic.__version__}: "
                    'pip install -U "anthropic>=1.15"'
                )
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallback = fallback

    def __call__(self, system: str, prompt: str) -> str:
        """Sends one request and returns the text of the reply.

        Raises:
            RuntimeError: If Claude declines the request.
        """
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": self.effort},
        }
        if self.fallback:
            response = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                **kwargs,
            )
        else:
            response = self.client.messages.create(**kwargs)
        if response.stop_reason == "refusal":
            raise RuntimeError("Claude declined the request")
        return "".join(b.text for b in response.content if b.type == "text")


class OpenAILLM:
    """Calls OpenAI or an OpenAI-compatible server through the OpenAI SDK.

    Credentials come from ``api_key`` or ``OPENAI_API_KEY``. Official OpenAI
    endpoints receive ``max_completion_tokens`` (which needs ``openai>=1.45``,
    :data:`MIN_OPENAI`, and is accepted by both chat and reasoning models);
    custom ``base_url`` endpoints (or ``OPENAI_BASE_URL``, such as vLLM, SGLang
    or Ollama) receive ``max_tokens`` and default ``api_key`` to ``"EMPTY"``
    when unset. Only the prompts built by the proposer are sent; no patient
    records are sent.

    Args:
        model: Model id (defaults to ``OPENAI_MODEL`` or ``"gpt-4o"``).
        max_tokens: Upper bound on the reply length.
        base_url: Optional API root, e.g. ``"http://localhost:8000/v1"``.
        api_key: Optional API key.
        extra_body: Optional extra request fields (e.g.
            ``{"chat_template_kwargs": {"enable_thinking": False}}`` for vLLM).
        client: An ``openai.OpenAI`` client; created when omitted.

    Examples:
        >>> from pyhealth.autoimprove import LLMProposer, OpenAILLM
        >>> proposer = LLMProposer(OpenAILLM())  # doctest: +SKIP
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 16000,
        base_url: str | None = None,
        api_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
        client: Any = None,
    ):
        if client is None:
            try:
                import openai
            except ImportError as e:
                raise ImportError(
                    'OpenAILLM needs the openai package: pip install "openai>=1.45"'
                ) from e
            found = tuple(
                int(x) for x in re.findall(r"\d+", openai.__version__)[:2]
            )
            if found < MIN_OPENAI:
                raise ImportError(
                    f"OpenAILLM needs openai>=1.45, found {openai.__version__}: "
                    'pip install -U "openai>=1.45"'
                )
            resolved_url = base_url or os.environ.get("OPENAI_BASE_URL")
            resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
            if resolved_url and not resolved_key:
                resolved_key = "EMPTY"
            client = openai.OpenAI(api_key=resolved_key, base_url=resolved_url)
        else:
            client_url = getattr(client, "base_url", None)
            resolved_url = base_url or (str(client_url) if client_url else None)
        host = (
            urllib.parse.urlparse(resolved_url).hostname
            if resolved_url
            else "api.openai.com"
        )
        is_openai = bool(
            host == "api.openai.com"
            or (host and host.endswith(".api.openai.com"))
        )
        self._token_key = "max_completion_tokens" if is_openai else "max_tokens"
        self.client = client
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body) if extra_body else None

    def __call__(self, system: str, prompt: str) -> str:
        """Sends one request and returns the text of the reply.

        Raises:
            RuntimeError: If the model declines the request or returns no text.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            self._token_key: self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        if getattr(message, "refusal", None):
            raise RuntimeError("OpenAI declined the request")
        content = message.content
        if isinstance(content, list):
            text = "".join(
                b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
                for b in content
            )
        else:
            text = content or ""
        if not text.strip():
            raise RuntimeError("OpenAI returned an empty reply")
        return text


class GeminiLLM:
    """Calls Gemini through the Google GenAI SDK.

    Credentials come from ``api_key``, ``GEMINI_API_KEY`` or
    ``GOOGLE_API_KEY``. Only the prompts built by the proposer are sent; no
    patient records are sent.

    Args:
        model: Gemini model id (defaults to ``GEMINI_MODEL`` or
            ``"gemini-2.5-pro"``).
        max_tokens: Upper bound on the output token budget (includes internal
            thinking tokens on thinking-enabled Gemini models; increase it if
            long reasoning exhausts the budget and leaves no reply text).
        api_key: Optional API key.
        client: A ``google.genai.Client``; created when omitted.

    Examples:
        >>> from pyhealth.autoimprove import GeminiLLM, LLMProposer
        >>> proposer = LLMProposer(GeminiLLM())  # doctest: +SKIP
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 16000,
        api_key: str | None = None,
        client: Any = None,
    ):
        if client is None:
            try:
                from google import genai
            except ImportError as e:
                raise ImportError(
                    'GeminiLLM needs the google-genai package: '
                    'pip install "google-genai>=1.0"'
                ) from e
            key = (
                api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )
            client = genai.Client(api_key=key) if key else genai.Client()
        self.client = client
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
        self.max_tokens = max_tokens

    def __call__(self, system: str, prompt: str) -> str:
        """Sends one request and returns the text of the reply.

        Raises:
            RuntimeError: If Gemini declines the request or returns no text.
        """
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "system_instruction": system,
                "max_output_tokens": self.max_tokens,
            },
        )
        feedback = getattr(response, "prompt_feedback", None)
        if getattr(feedback, "block_reason", None):
            raise RuntimeError("Gemini declined the request")
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []
            text = "".join(
                p.text
                for p in parts
                if getattr(p, "text", None) and not getattr(p, "thought", False)
            )
            if text.strip():
                return text
        text = getattr(response, "text", None) or ""
        if not text.strip():
            raise RuntimeError("Gemini returned an empty reply")
        return text

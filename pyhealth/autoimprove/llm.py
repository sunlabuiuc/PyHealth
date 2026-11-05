"""LLM backends for :class:`~pyhealth.autoimprove.LLMProposer`.

Any callable ``llm(system, prompt) -> str`` works as a backend. This module
provides one for Claude through the official ``anthropic`` SDK
(``pip install "anthropic>=1.15"``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

LLM = Callable[[str, str], str]
"""A backend: takes a system prompt and a user prompt, returns the reply text."""

MIN_ANTHROPIC = (1, 15)
"""Oldest ``anthropic`` release tested with the request :class:`ClaudeLLM` sends."""


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

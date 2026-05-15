"""Pluggable LLM backends for :class:`LLMEvidenceRetriever`.

The retriever is intentionally model-agnostic. Any callable taking a
prompt string and returning a response string satisfies the contract;
this module ships the two most common concrete backends:

- :class:`~pyhealth.models.StubLLMBackend` (defined in
  :mod:`pyhealth.models.llm_evidence_retriever`) — deterministic,
  keyword-based, offline. Used by the default test and example paths.
- :class:`OpenAIBackend` — thin wrapper around the OpenAI chat
  completions API. Requires the ``openai`` package and an
  ``OPENAI_API_KEY`` environment variable.

Additional backends (other hosted APIs, local Ollama,
HuggingFace pipelines, Azure OpenAI) can follow the same pattern — see
:class:`OpenAIBackend` as a reference.

Paper:
    M. Ahsan et al. "Retrieving Evidence from EHRs with LLMs:
    Possibilities and Challenges." Proceedings of Machine Learning
    Research, 2024.

Paper link:
    https://proceedings.mlr.press/v248/ahsan24a.html

Author:
    Arnab Karmakar (arnabk3@illinois.edu)
"""
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


class OpenAIBackend:
    """LLM backend that calls OpenAI's chat completions API.

    The ``openai`` package is imported lazily inside ``__init__`` so
    that importing :mod:`pyhealth.models` never fails when the OpenAI
    SDK is not installed. Callers are expected to install the optional
    dependency with ``pip install openai`` before instantiating this
    class.

    Attributes:
        model (str): OpenAI model identifier (e.g. ``"gpt-4o-mini"``).
        temperature (float): Sampling temperature for generation.
        max_tokens (int): Maximum tokens per completion.
        client: The underlying ``openai.OpenAI`` client instance.

    Example:
        >>> import os
        >>> os.environ["OPENAI_API_KEY"] = "sk-..."  # doctest: +SKIP
        >>> backend = OpenAIBackend(model="gpt-4o-mini")  # doctest: +SKIP
        >>> retriever = LLMEvidenceRetriever(backend=backend)  # doctest: +SKIP
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        base_url: Optional[str] = None,
        request_timeout: float = 30.0,
    ) -> None:
        """Initialize the OpenAI backend.

        Args:
            model (str): OpenAI model identifier. Defaults to
                ``"gpt-4o-mini"`` for cost-conscious evaluation.
            api_key (Optional[str]): Explicit API key. When ``None``,
                the ``OPENAI_API_KEY`` environment variable is used.
                Never commit keys — use a ``.env`` file or your shell
                environment.
            temperature (float): Sampling temperature. Defaults to
                ``0.0`` for reproducibility.
            max_tokens (int): Maximum tokens per completion. Defaults
                to ``256``, matching the retriever's short-JSON
                response shape.
            base_url (Optional[str]): Optional custom API base URL.
                Useful for Azure OpenAI or self-hosted gateways.
            request_timeout (float): Per-request timeout in seconds.

        Raises:
            ImportError: If the ``openai`` package is not installed.
            ValueError: If no API key can be resolved from the
                argument or the environment.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised via tests
            raise ImportError(
                "OpenAIBackend requires the 'openai' package. Install "
                "it with `pip install openai`."
            ) from exc

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAIBackend requires an API key. Set the "
                "OPENAI_API_KEY environment variable or pass api_key=. "
                "Never commit the key to source control."
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._request_timeout = request_timeout
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=base_url,
            timeout=request_timeout,
        )

    def __call__(self, prompt: str) -> str:
        """Send ``prompt`` to the configured model and return the text.

        The backend requests a JSON-object response format so the
        retriever's downstream JSON parser receives a well-formed
        payload even when the model would otherwise wrap its answer in
        prose.

        Args:
            prompt (str): The full prompt to send to the model.

        Returns:
            str: The model's response text (JSON-encoded by request).
                Returns ``"{}"`` when the API returns an empty message.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content: Any = response.choices[0].message.content
        return content or "{}"

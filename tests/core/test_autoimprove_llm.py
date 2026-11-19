"""Tests for OpenAILLM and GeminiLLM backends."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from pyhealth.autoimprove import GeminiLLM, OpenAILLM


class TestOpenAILLM(unittest.TestCase):
    def _client(
        self,
        content: object = "hello",
        refusal: str | None = None,
        base_url: str | None = None,
    ):
        calls: list[dict] = []

        def create(**kwargs):
            calls.append(kwargs)
            msg = SimpleNamespace(content=content, refusal=refusal)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        client = SimpleNamespace(
            base_url=base_url,
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )
        return client, calls

    def test_request_shape_and_token_parameter(self):
        client, calls = self._client(base_url="https://api.openai.com/v1/")
        extra = {"chat_template_kwargs": {"enable_thinking": False}}
        # An injected client's own base_url takes priority over OPENAI_BASE_URL
        with mock.patch.dict(
            os.environ, {"OPENAI_BASE_URL": "http://localhost:8000/v1"}, clear=True
        ):
            self.assertEqual(
                OpenAILLM(client=client, extra_body=extra)("sys", "hi"),
                "hello",
            )
            self.assertEqual(calls[0]["model"], "gpt-4o")
            self.assertEqual(calls[0]["max_completion_tokens"], 16000)
            self.assertNotIn("max_tokens", calls[0])
            self.assertEqual(calls[0]["extra_body"], extra)
            self.assertEqual(
                calls[0]["messages"],
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                ],
            )

        # Compatible endpoints (e.g. vLLM / Ollama) receive max_tokens
        vllm_client, vllm_calls = self._client(
            base_url="http://localhost:8000/v1/"
        )
        OpenAILLM(client=vllm_client)("sys", "hi")
        self.assertEqual(vllm_calls[0]["max_tokens"], 16000)
        self.assertNotIn("max_completion_tokens", vllm_calls[0])

    def test_env_fallbacks_and_vlm_content_blocks(self):
        parts = [SimpleNamespace(text="part1 "), {"text": "part2"}]
        client, calls = self._client(
            content=parts, base_url="http://127.0.0.1:8000/v1/"
        )

        created: list[dict] = []
        fake_openai = SimpleNamespace(
            __version__="1.45.0",
            OpenAI=lambda **kw: (created.append(kw) or client),
        )
        env = {
            "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
            "OPENAI_MODEL": "local-vlm",
        }
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.dict(sys.modules, {"openai": fake_openai}),
        ):
            llm = OpenAILLM()
            self.assertEqual(llm("sys", "hi"), "part1 part2")
        self.assertEqual(llm.model, "local-vlm")
        self.assertEqual(calls[0]["max_tokens"], 16000)
        self.assertEqual(
            created[0],
            {"api_key": "EMPTY", "base_url": "http://127.0.0.1:8000/v1"},
        )

    def test_refusal_and_empty_reply_raise(self):
        refused, _ = self._client(refusal="policy")
        with self.assertRaises(RuntimeError):
            OpenAILLM(client=refused)("sys", "hi")

        empty, _ = self._client(content="  ")
        with self.assertRaises(RuntimeError):
            OpenAILLM(client=empty)("sys", "hi")

    def test_missing_or_old_sdk_is_refused(self):
        with (
            mock.patch.dict(sys.modules, {"openai": None}),
            self.assertRaisesRegex(ImportError, "openai>=1.45"),
        ):
            OpenAILLM()

        old = SimpleNamespace(__version__="1.44.0", OpenAI=lambda **kw: None)
        with (
            mock.patch.dict(sys.modules, {"openai": old}),
            self.assertRaisesRegex(ImportError, "openai>=1.45"),
        ):
            OpenAILLM()

    @unittest.skipUnless(
        os.environ.get("OPENAI_API_KEY") and importlib.util.find_spec("openai"),
        "needs OPENAI_API_KEY and the openai package",
    )
    def test_live_request(self):
        reply = OpenAILLM(max_tokens=256)(
            "Answer in one word.", "What colour is the sky on a clear day?"
        )
        self.assertTrue(reply.strip())


class TestGeminiLLM(unittest.TestCase):
    def _client(self, parts=None, block_reason=None):
        calls: list[dict] = []
        if parts is None:
            parts = [
                SimpleNamespace(text="thinking", thought=True),
                SimpleNamespace(text="hello", thought=False),
            ]

        def generate_content(**kwargs):
            calls.append(kwargs)
            feedback = (
                SimpleNamespace(block_reason=block_reason)
                if block_reason
                else None
            )
            cand = SimpleNamespace(content=SimpleNamespace(parts=parts))
            return SimpleNamespace(
                prompt_feedback=feedback, candidates=[] if block_reason else [cand]
            )

        client = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        )
        return client, calls

    def test_request_shape_and_env_fallbacks(self):
        client, calls = self._client()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(GeminiLLM(client=client)("sys", "hi"), "hello")
        self.assertEqual(calls[0]["model"], "gemini-2.5-pro")
        self.assertEqual(calls[0]["contents"], "hi")
        self.assertEqual(
            calls[0]["config"],
            {"system_instruction": "sys", "max_output_tokens": 16000},
        )

        # Fallback to response.text when candidate parts carry no text
        text_only = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **kw: SimpleNamespace(
                    prompt_feedback=None, candidates=[], text="direct"
                )
            )
        )
        self.assertEqual(GeminiLLM(client=text_only)("sys", "hi"), "direct")

        # SDK Client initialized with GOOGLE_API_KEY and GEMINI_MODEL fallbacks
        created: list[dict] = []
        fake_genai = SimpleNamespace(
            Client=lambda **kw: (created.append(kw) or client)
        )
        fake_google = SimpleNamespace(genai=fake_genai)
        env = {"GOOGLE_API_KEY": "g-key", "GEMINI_MODEL": "gemini-2.5-flash"}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.dict(
                sys.modules, {"google": fake_google, "google.genai": fake_genai}
            ),
        ):
            llm = GeminiLLM()
        self.assertEqual(llm.model, "gemini-2.5-flash")
        self.assertEqual(created[0], {"api_key": "g-key"})

    def test_refusal_and_empty_reply_raise(self):
        blocked, _ = self._client(block_reason="SAFETY")
        with self.assertRaises(RuntimeError):
            GeminiLLM(client=blocked)("sys", "hi")

        empty, _ = self._client(parts=[SimpleNamespace(text="   ", thought=False)])
        with self.assertRaises(RuntimeError):
            GeminiLLM(client=empty)("sys", "hi")

    def test_missing_sdk_raises_import_error(self):
        with (
            mock.patch.dict(sys.modules, {"google": None, "google.genai": None}),
            self.assertRaisesRegex(ImportError, "google-genai>=1.0"),
        ):
            GeminiLLM()

    @unittest.skipUnless(
        (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        and importlib.util.find_spec("google") is not None
        and importlib.util.find_spec("google.genai") is not None,
        "needs GEMINI_API_KEY/GOOGLE_API_KEY and the google-genai package",
    )
    def test_live_request(self):
        reply = GeminiLLM(max_tokens=1024)(
            "Answer in one word.", "What colour is the sky on a clear day?"
        )
        self.assertTrue(reply.strip())


if __name__ == "__main__":
    unittest.main()

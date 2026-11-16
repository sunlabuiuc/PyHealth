"""Tests for OpenAILLM and GeminiLLM backends."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from pyhealth.autoimprove import GeminiLLM, OpenAILLM


class TestOpenAILLM(unittest.TestCase):
    def _client(self, content: object = "hello", refusal: str | None = None):
        calls: list[dict] = []

        def create(**kwargs):
            calls.append(kwargs)
            msg = SimpleNamespace(content=content, refusal=refusal)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        return client, calls

    def test_request_shape_and_extra_body(self):
        client, calls = self._client()
        extra = {"chat_template_kwargs": {"enable_thinking": False}}
        self.assertEqual(
            OpenAILLM(client=client, extra_body=extra)("sys", "hi"), "hello"
        )
        self.assertEqual(calls[0]["model"], "gpt-4o")
        self.assertEqual(calls[0]["max_tokens"], 16000)
        self.assertEqual(calls[0]["extra_body"], extra)
        self.assertEqual(
            calls[0]["messages"],
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
        )

    def test_vlm_content_blocks_and_base_url_without_key(self):
        parts = [SimpleNamespace(text="part1 "), {"text": "part2"}]
        client, _ = self._client(content=parts)
        self.assertEqual(OpenAILLM(client=client)("sys", "hi"), "part1 part2")

        created: list[dict] = []
        fake_openai = SimpleNamespace(
            OpenAI=lambda **kw: (created.append(kw) or client)
        )
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.dict(sys.modules, {"openai": fake_openai}),
        ):
            llm = OpenAILLM(
                model="local-vlm", base_url="http://localhost:8000/v1"
            )
        self.assertEqual(llm.model, "local-vlm")
        self.assertEqual(
            created[0],
            {"api_key": "EMPTY", "base_url": "http://localhost:8000/v1"},
        )

    def test_refusal_and_empty_reply_raise(self):
        refused, _ = self._client(refusal="policy")
        with self.assertRaises(RuntimeError):
            OpenAILLM(client=refused)("sys", "hi")

        empty, _ = self._client(content="  ")
        with self.assertRaises(RuntimeError):
            OpenAILLM(client=empty)("sys", "hi")

    def test_missing_sdk_raises_import_error(self):
        with (
            mock.patch.dict(sys.modules, {"openai": None}),
            self.assertRaisesRegex(ImportError, "openai>=1.0"),
        ):
            OpenAILLM()


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

    def test_request_shape_and_skips_thoughts(self):
        client, calls = self._client()
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

        # SDK Client initialized with GEMINI_API_KEY from environment
        created: list[dict] = []
        fake_genai = SimpleNamespace(
            Client=lambda **kw: (created.append(kw) or client)
        )
        fake_google = SimpleNamespace(genai=fake_genai)
        with (
            mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k1"}, clear=True),
            mock.patch.dict(
                sys.modules, {"google": fake_google, "google.genai": fake_genai}
            ),
        ):
            llm = GeminiLLM(model="gemini-2.5-flash")
        self.assertEqual(llm.model, "gemini-2.5-flash")
        self.assertEqual(created[0], {"api_key": "k1"})

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


if __name__ == "__main__":
    unittest.main()

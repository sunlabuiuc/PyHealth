import logging
import os
import re
import time
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn

from pyhealth.tasks.sdoh_utils import TARGET_CODES, codes_to_multihot

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """\
You are an assistant that extracts SDOH ICD-9 V-codes from clinical notes.
Return only the codes, comma-separated, inside triple backticks.
If no target codes are present, return None inside triple backticks.
Target codes: {codes}
"""


class SDOHICD9LLM(nn.Module):
    """Admission-level SDOH ICD-9 V-code detector using an LLM."""

    mode = "multilabel"

    def __init__(
        self,
        target_codes: Optional[Sequence[str]] = None,
        model_name: str = "gpt-4o-mini",
        prompt_template: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 100,
        max_chars: int = 100000,
        temperature: float = 0.0,
        sleep_s: float = 0.2,
        dry_run: bool = False,
    ) -> None:
        super().__init__()
        self.target_codes = list(target_codes) if target_codes else list(TARGET_CODES)
        self.model_name = model_name
        self.prompt_template = prompt_template or PROMPT_TEMPLATE
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.temperature = temperature
        self.sleep_s = sleep_s
        self.dry_run = dry_run
        self._client = None

        if not self.api_key and not self.dry_run:
            raise EnvironmentError(
                "OPENAI_API_KEY is required unless dry_run=True."
            )

        mode = "dry-run" if dry_run else "live"
        logger.info(
            "Initialized SDOHICD9LLM (mode=%s, model=%s, codes=%d)",
            mode, model_name, len(self.target_codes)
        )

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _call_openai_api(self, text: str) -> str:
        if self.dry_run:
            return "```None```"

        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "\n\n[Note truncated due to length...]"

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": self.prompt_template.format(
                        codes=", ".join(self.target_codes)
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Analyze this clinical note and identify SDOH codes:\n\n"
                        f"{text}"
                    ),
                },
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def _parse_llm_response(self, response: str) -> Set[str]:
        if not response:
            return set()

        matches = re.findall(r"```(.*?)```", response, re.DOTALL)
        if matches:
            response = matches[0].strip()
        else:
            response = response.strip()

        if response.lower().strip() == "none":
            return set()

        response = response.replace("Answer:", "").replace("Codes:", "").strip()
        for delimiter in [",", ";", " ", "\n"]:
            if delimiter in response:
                parts = [c.strip() for c in response.split(delimiter)]
                break
        else:
            parts = [response.strip()]

        valid = {code.upper() for code in parts if code.strip()}
        target_set = {code.upper() for code in self.target_codes}
        return {code for code in valid if code in target_set}

    def _predict_admission(
        self,
        notes: Iterable[str],
        note_categories: Optional[Iterable[str]] = None,
        chartdates: Optional[Iterable[str]] = None,
    ) -> Tuple[Set[str], List[dict]]:
        aggregated: Set[str] = set()
        note_results: List[dict] = []

        categories = list(note_categories) if note_categories is not None else []
        dates = list(chartdates) if chartdates is not None else []
        notes_list = list(notes)

        logger.debug("Processing admission with %d notes", len(notes_list))

        for idx, note in enumerate(notes_list):
            category = categories[idx] if idx < len(categories) else "Unknown"
            date = dates[idx] if idx < len(dates) else "Unknown"

            response = self._call_openai_api(note)
            predicted = self._parse_llm_response(response)
            aggregated.update(predicted)

            logger.debug(
                "Note %d/%d (%s, %s): predicted %s",
                idx + 1, len(notes_list), category, date, sorted(predicted) or "none"
            )

            note_results.append(
                {
                    "category": category,
                    "date": date,
                    "predicted_codes": sorted(predicted),
                    "llm_response": response,
                }
            )

            if self.sleep_s > 0 and not self.dry_run:
                time.sleep(self.sleep_s)

        logger.debug("Admission complete: aggregated codes %s", sorted(aggregated))
        return aggregated, note_results

    def forward(
        self,
        notes,
        note_categories=None,
        chartdates=None,
        label=None,
        **kwargs,
    ):
        if notes and isinstance(notes[0], str):
            notes_batch = [notes]
            categories_batch = [note_categories] if note_categories is not None else [None]
            dates_batch = [chartdates] if chartdates is not None else [None]
        else:
            notes_batch = notes
            categories_batch = note_categories or [None] * len(notes_batch)
            dates_batch = chartdates or [None] * len(notes_batch)

        batch_probs: List[torch.Tensor] = []
        for note_list, cats, dates in zip(
            notes_batch, categories_batch, dates_batch
        ):
            predicted, _ = self._predict_admission(note_list, cats, dates)
            batch_probs.append(codes_to_multihot(predicted, self.target_codes))

        y_prob = torch.stack(batch_probs, dim=0)
        if label is not None and isinstance(label, torch.Tensor):
            y_prob = y_prob.to(label.device)
            y_true = label
        else:
            y_true = label

        loss = torch.zeros(1, device=y_prob.device).sum()
        return {"loss": loss, "y_prob": y_prob, "y_true": y_true}

    def predict_admission_with_notes(
        self,
        notes: Iterable[str],
        note_categories: Optional[Iterable[str]] = None,
        chartdates: Optional[Iterable[str]] = None,
    ) -> Tuple[Set[str], List[dict]]:
        return self._predict_admission(notes, note_categories, chartdates)

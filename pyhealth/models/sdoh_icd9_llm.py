import os
import hashlib
import re
import time
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from pyhealth.tasks.sdoh_utils import TARGET_CODES


PROMPT_PATH = os.path.join(os.path.dirname(__file__), "sdoh_icd9_task.txt")


def _load_prompt_template() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


class SDOHICD9LLM:
    """Admission-level SDOH ICD-9 V-code detector using an LLM."""

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
        max_notes: Optional[int] = None,
        dry_run: bool = False,
    ) -> None:
        self.target_codes = list(target_codes) if target_codes else list(TARGET_CODES)
        self.model_name = model_name
        self.prompt_template = prompt_template or _load_prompt_template()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.temperature = temperature
        self.sleep_s = sleep_s
        self.max_notes = max_notes
        self.dry_run = dry_run
        self._client = None

        if not self.api_key and not self.dry_run:
            raise EnvironmentError(
                "OPENAI_API_KEY is required unless dry_run=True."
            )

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _call_openai_api(self, text: str) -> str:
        self._write_prompt_preview(text)

        if self.dry_run:
            return "```None```"

        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "\n\n[Note truncated due to length...]"

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.prompt_template.format(note=text)},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def _write_prompt_preview(self, text: str) -> None:
        prompt = self.prompt_template.format(note=text)
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
        filename = f"sdoh_prompt_{digest}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(prompt)

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
        if self.max_notes and self.max_notes > 0:
            notes_list = notes_list[: self.max_notes]
            categories = categories[: self.max_notes]
            dates = dates[: self.max_notes]

        for idx, note in enumerate(notes_list):
            category = categories[idx] if idx < len(categories) else "Unknown"
            date = dates[idx] if idx < len(dates) else "Unknown"
            response = self._call_openai_api(note)
            predicted = self._parse_llm_response(response)
            aggregated.update(predicted)

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

        return aggregated, note_results

    def predict_admission_with_notes(
        self,
        notes: Iterable[str],
        note_categories: Optional[Iterable[str]] = None,
        chartdates: Optional[Iterable[str]] = None,
    ) -> Tuple[Set[str], List[dict]]:
        return self._predict_admission(notes, note_categories, chartdates)

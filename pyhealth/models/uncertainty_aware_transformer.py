from __future__ import annotations

import random, re, string
from copy import deepcopy
from itertools import islice
from typing import Dict, List, Tuple, Union, Optional
import torch
import torch, torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models.base_model import BaseModel

from copy import deepcopy


class FrozenSampleDataset(SampleDataset):
    """Clone of SampleDataset that reuses existing processors/vocabs."""

    def __init__(
        self,
        samples,
        input_schema,
        output_schema,
        input_processors,
        output_processors,
        dataset_name: str = "",
        task_name: str = "",
    ):
        # shallow attributes
        self.samples, self.input_schema, self.output_schema = samples, input_schema, output_schema
        self.input_processors, self.output_processors = input_processors, output_processors
        self.dataset_name, self.task_name = dataset_name, task_name

        # quick lookup tables (optional)
        self.patient_to_index, self.record_to_index = {}, {}
        for i, s in enumerate(samples):
            if (pid := s.get("patient_id")) is not None:
                self.patient_to_index.setdefault(pid, []).append(i)
            if (rid := s.get("record_id", s.get("visit_id"))) is not None:
                self.record_to_index.setdefault(rid, []).append(i)


class UncertaintyAwareZeroShotClassifier(BaseModel):
    """Zero-shot text classifier with entropy-based uncertainty sampling."""

    def __init__(
        self,
        dataset: SampleDataset,
        model_name: str = "johnsnowlabs/JSL-MedLlama-3-8B-v2.0",
        prompt_template: str | None = None,
        batch_size: int = 2,
        similarity_threshold: float = 0.0,
        cache_dir: str | None = None,
        my_device: Optional[str] = None,
    ):
        super().__init__(dataset)

        # label vocab
        proc = next(iter(dataset.output_processors))
        self.output_processor = dataset.output_processors[proc]
        self.id2label = {i: n.strip() for n, i in self.output_processor.label_vocab.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.labels = [self.id2label[i] for i in range(len(self.id2label))]

        # settings
        self.batch_size = batch_size
        self.sim_thresh = similarity_threshold
        self.my_device = my_device or ("cuda" if torch.cuda.is_available() else "cpu")

        # model + tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16 if self.my_device == "cuda" else torch.float32,
        ).to(self.my_device).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.prompt_template = prompt_template

        self.results: List[Dict[str, Union[int, float, str]]] = []

    @staticmethod
    def _norm(t: str) -> str:
        return re.sub(f"[{re.escape(string.punctuation)}]", "", t.lower()).strip()

    def _fuzzy_match(self, raw: str) -> str | None:
        raw_n = self._norm(raw)
        best, best_len = None, 0
        for lab in self.labels:
            lab_n = self._norm(lab)
            for i in range(len(raw_n)):
                if (l := len(sub := raw_n[i : i + best_len + 1])) <= best_len:
                    continue
                if sub and sub in lab_n:
                    best, best_len = lab, l
        return best

    @torch.inference_mode()
    def _gen_answer(self, note: str) -> Tuple[str, float]:
        prompt = self.prompt_template.format(labels=", ".join(self.labels), text=note.strip())
        inp = self.tokenizer(prompt, return_tensors="pt").to(self.my_device)
        gen = self.model.generate(
            **inp,
            min_new_tokens=2,
            max_new_tokens=6,
            output_scores=True,
            do_sample=True,
            return_dict_in_generate=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        ans_tok = gen.sequences[0][inp.input_ids.size(-1) :]
        raw = self.tokenizer.decode(ans_tok, skip_special_tokens=True).rstrip(".").strip()

        max_ent = max(
            float(-(F.softmax(s[0], dim=-1) * F.log_softmax(s[0], dim=-1)).sum())
            for s in gen.scores
        ) if gen.scores else float("nan")
        
        return raw, max_ent

    def predict_dataset(
        self,
        max_batches: int | None = None,
        text_key: str = "transcription",
        label_key: str = "medical_specialty",
    ):
        """Run zero-shot prediction on the dataset and cache results."""
        self.results.clear()
        loader = get_dataloader(self.dataset, batch_size=self.batch_size, shuffle=True)
        total = len(loader) if max_batches in (None, -1) else max_batches
        for b, batch in enumerate(
            tqdm(islice(loader, max_batches), total=total, desc="Zero-shot generate")
        ):
            for i, note in enumerate(batch[text_key]):
                raw, ent = self._gen_answer(note)
                pred = self.label2id.get(self._fuzzy_match(raw), -1)
                self.results.append(
                    dict(
                        index=b * self.batch_size + i,
                        prediction=pred,
                        gt=int(batch[label_key][i]),
                        entropy=ent,
                        raw_answer=raw,
                    )
                )

    def _split(self, k: int | float) -> Tuple[List, List]:
        if not self.results:
            raise RuntimeError("Run predict_dataset() first")
        # sort *descending* so highest entropy (most uncertain) comes first
        ordered = sorted(self.results, key=lambda r: r["entropy"], reverse=True)
        if isinstance(k, float):
            if not 0 < k <= 1:
                raise ValueError("k must be in (0,1]")
            k = int(k * len(ordered))
        return ordered[:k], ordered[k:]

    def get_uncertain(self, k: int | float):
        """Return (most-uncertain, remaining) result dicts."""
        return self._split(k)

    # freeze helper to help copy PyHealth datasets correctly
    def _freeze(self, idx) -> FrozenSampleDataset:
        s = [deepcopy(self.dataset.samples[i]) for i in idx]
        return FrozenSampleDataset(
            s,
            self.dataset.input_schema,
            self.dataset.output_schema,
            self.dataset.input_processors,
            self.dataset.output_processors,
            self.dataset.dataset_name,
            self.dataset.task_name,
        )

    def get_uncertain_datasets(self, k: int | float):
        """Return (uncertain_ds, certain_ds) where uncertain_ds contains the k most-uncertain samples."""
        u, c = self.get_uncertain(k)
        u_idx = {r["index"] for r in u}
        c_idx = [i for i in range(len(self.dataset.samples)) if i not in u_idx]
        return self._freeze(u_idx), self._freeze(c_idx)

    def get_random(self, k: int | float):
        """Return (random_subset, remaining) result dicts, randomly selected from processed results."""
        if not self.results:
            raise RuntimeError("Run predict_dataset() first so there is a processed subset to sample from.")
        
        results_copy = self.results.copy()
        random.shuffle(results_copy)
        
        if isinstance(k, float):
            if not 0 < k <= 1:
                raise ValueError("k must be in (0,1] for float input.")
            k = int(k * len(results_copy))
    
        return results_copy[:k], results_copy[k:]

    def get_random_datasets(self, k: int | float):
        """Randomly split the processed results using `get_random`"""
        if not self.results:
            raise RuntimeError("Run predict_dataset() first so there is a processed subset to sample from.")
    
        selected, remaining = self.get_random(k)
        selected_idx = {r["index"] for r in selected}
        remaining_idx = {r["index"] for r in remaining}
    
        return (
            self._freeze(selected_idx),
            self._freeze(remaining_idx),
            selected,
            remaining,
        )

"""Tests for MedFlamingo model, VQARADDataset, and MedicalVQATask.

All tests use synthetic / pseudo data generated in memory or in temporary
directories.  No real datasets, internet access, or heavyweight model weights
are required.  The ``TestableMedFlamingo`` subclass replaces the production
CLIP vision encoder and OPT language model with lightweight stubs so the
entire test suite completes in under a few seconds on CPU.
"""

import json
import os
import shutil
import tempfile
import unittest
import warnings
from types import SimpleNamespace

from PIL import Image
import torch
import torch.nn as nn

from pyhealth.data import Patient, Event
from pyhealth.datasets import (
    VQARADDataset,
    create_sample_dataset,
    get_dataloader,
    split_by_sample,
)
from pyhealth.models.base_model import BaseModel
from pyhealth.models.medflamingo import MedFlamingo
from pyhealth.tasks import MedicalVQATask
from pyhealth.trainer import Trainer


REAL_VQARAD_ROOT = os.getenv("PYHEALTH_VQARAD_ROOT")

warnings.filterwarnings(
    "ignore",
    message=r"A newer version of litdata is available .*",
    category=UserWarning,
)


# ---------------------------------------------------------------------------
# Lightweight model stubs (no CLIP / OPT downloads)
# ---------------------------------------------------------------------------


class FakeBatch(dict):
    def to(self, device):
        return FakeBatch({key: value.to(device) for key, value in self.items()})


class FakeTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.last_text = ""

    def __call__(
        self,
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ):
        if isinstance(texts, str):
            texts = [texts]
        self.last_text = texts[0]
        seq_len = min(max(len(text.split()) for text in texts) + 1, max_length)
        input_ids = []
        attention_mask = []
        for row, text in enumerate(texts):
            tokens = [(row + idx) % 17 + 1 for idx, _ in enumerate(text.split()[:seq_len])]
            tokens = tokens + [0] * (seq_len - len(tokens))
            mask = [1 if token != 0 else 0 for token in tokens]
            if not any(mask):
                tokens[0] = 1
                mask[0] = 1
            input_ids.append(tokens)
            attention_mask.append(mask)
        return FakeBatch(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        )

    def decode(self, tokens, skip_special_tokens=True):
        return f"{self.last_text} synthetic answer"


class FakeLanguageInnerModel(nn.Module):
    def __init__(self, vocab_size=32, hidden_size=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)


class FakeLanguageModel(nn.Module):
    def __init__(self, hidden_size=8, num_hidden_layers=4):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
        )
        self.model = FakeLanguageInnerModel(hidden_size=hidden_size)

    def generate(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        max_new_tokens=16,
        **kwargs,
    ):
        # Accept either input_ids or inputs_embeds; generate() passes inputs_embeds
        # so that the xattn-conditioned representations are forwarded to the LLM.
        if inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device
        else:
            batch_size = input_ids.shape[0]
            device = input_ids.device
        return torch.full(
            (batch_size, min(max_new_tokens, 4)),
            fill_value=7,
            dtype=torch.long,
            device=device,
        )


class FakeVisionEncoder(nn.Module):
    def __init__(self, hidden_size=8, num_tokens=5):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.num_tokens = num_tokens
        self.proj = nn.Linear(1, hidden_size)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        pooled = pixel_values.float().reshape(batch_size, -1).mean(dim=1, keepdim=True)
        repeated = pooled.unsqueeze(1).repeat(1, self.num_tokens, 1)
        return SimpleNamespace(last_hidden_state=self.proj(repeated))


class TestableMedFlamingo(MedFlamingo):
    __test__ = False

    def _init_vision_encoder(self) -> None:
        self._vision_encoder = FakeVisionEncoder()
        if self.freeze_vision:
            for param in self._vision_encoder.parameters():
                param.requires_grad = False

    def _init_lang_model(self) -> None:
        self._lang_model = FakeLanguageModel()
        self._tokenizer = FakeTokenizer()
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if self.freeze_lm:
            for param in self._lang_model.parameters():
                param.requires_grad = False


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


class TestMedFlamingo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.vqarad_root = tempfile.mkdtemp()
        cls.vqarad_cache_dir = tempfile.mkdtemp()
        cls.samples = []
        labels = ["yes", "no", "yes", "no"]
        questions = [
            "is there a fracture",
            "is the study normal",
            "is there consolidation",
            "is there edema",
        ]

        for idx, (answer, question) in enumerate(zip(labels, questions)):
            image_path = os.path.join(cls.temp_dir, f"img_{idx}.png")
            image = Image.fromarray(
                torch.randint(0, 255, (16, 16, 3), dtype=torch.uint8).numpy(),
                mode="RGB",
            )
            image.save(image_path)
            cls.samples.append(
                {
                    "patient_id": f"patient-{idx // 2}",
                    "visit_id": f"visit-{idx}",
                    "image": image_path,
                    "question": question,
                    "answer": answer,
                }
            )

        cls.dataset = create_sample_dataset(
            samples=cls.samples,
            input_schema={
                "image": ("image", {"image_size": 16, "mode": "RGB"}),
                "question": "text",
            },
            output_schema={"answer": "multiclass"},
            dataset_name="test_medflamingo",
        )

        cls._create_vqarad_fixture(
            cls.vqarad_root,
            num_examples=8,
        )

    @classmethod
    def _create_vqarad_fixture(cls, root, num_examples):
        images_dir = os.path.join(root, "images")
        os.makedirs(images_dir, exist_ok=True)
        entries = []
        answers = ["yes", "no"] * (num_examples // 2)
        questions = [
            "is there a fracture",
            "is the study normal",
            "is there consolidation",
            "is there edema",
            "is there a mass",
            "is there pleural effusion",
            "is there cardiomegaly",
            "is there pneumothorax",
        ]

        for idx in range(num_examples):
            image_name = f"study_{idx}.png"
            image_path = os.path.join(images_dir, image_name)
            image = Image.fromarray(
                torch.randint(0, 255, (16, 16, 3), dtype=torch.uint8).numpy(),
                mode="RGB",
            )
            image.save(image_path)
            entries.append(
                {
                    "IMAGE_PATH": image_name,
                    "QUESTION": questions[idx % len(questions)],
                    "ANSWER": answers[idx % len(answers)],
                    "ANSWER_TYPE": "closed",
                    "QUESTION_TYPE": "presence",
                    "IMAGE_ORGAN": "chest",
                }
            )

        with open(os.path.join(root, "VQA_RAD Dataset Public.json"), "w") as f:
            json.dump(entries, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)
        shutil.rmtree(cls.vqarad_root)
        shutil.rmtree(cls.vqarad_cache_dir)

    def _build_vqarad_sample_dataset(self):
        dataset = VQARADDataset(
            root=self.vqarad_root,
            cache_dir=self.vqarad_cache_dir,
            num_workers=1,
        )
        return dataset.set_task(num_workers=1)

    # ------------------------------------------------------------------
    # MedicalVQATask unit tests
    # ------------------------------------------------------------------

    def test_medical_vqa_task_schema(self):
        """Task declares the expected input/output schema."""
        task = MedicalVQATask()
        self.assertEqual(task.task_name, "MedicalVQA")
        self.assertEqual(task.input_schema, {"image": "image", "question": "text"})
        self.assertEqual(task.output_schema, {"answer": "multiclass"})

    def test_medical_vqa_task_call_emits_correct_fields(self):
        """__call__ returns one sample per vqarad event with all required keys."""
        import polars as pl
        from datetime import datetime

        task = MedicalVQATask()

        # Patient expects a Polars DataFrame with columns:
        #   event_type, timestamp, vqarad/<attr>
        rows = [
            {
                "event_type": "vqarad",
                "timestamp": datetime(2020, 1, i + 1),
                "vqarad/image_path": f"/data/images/img_{i}.jpg",
                "vqarad/question": f"Is there a fracture? ({i})",
                "vqarad/answer": "yes" if i % 2 == 0 else "no",
            }
            for i in range(3)
        ]
        df = pl.DataFrame(rows)
        patient = Patient(patient_id="p-001", data_source=df)

        samples = task(patient)

        self.assertEqual(len(samples), 3)
        for sample in samples:
            self.assertIn("patient_id", sample)
            self.assertIn("image", sample)
            self.assertIn("question", sample)
            self.assertIn("answer", sample)
            self.assertEqual(sample["patient_id"], "p-001")

    def test_medical_vqa_task_call_empty_patient(self):
        """__call__ returns an empty list when the patient has no vqarad events."""
        import polars as pl

        task = MedicalVQATask()
        # DataFrame with required columns but zero rows
        df = pl.DataFrame({"event_type": [], "timestamp": []}).with_columns(
            pl.col("timestamp").cast(pl.Datetime)
        )
        patient = Patient(patient_id="p-empty", data_source=df)
        self.assertEqual(task(patient), [])

    # ------------------------------------------------------------------
    # MedFlamingo model unit tests
    # ------------------------------------------------------------------

    def test_model_initialization_standalone(self):
        """Standalone model (no dataset) initialises with expected defaults."""
        model = TestableMedFlamingo(dataset=None)
        self.assertIsInstance(model, MedFlamingo)
        self.assertIsInstance(model, BaseModel)
        self.assertEqual(model.vision_model_name, "openai/clip-vit-large-patch14")
        self.assertEqual(model.lang_model_name, "facebook/opt-6.7b")
        # FakeLanguageModel has 4 hidden layers; cross_attn_every_n_layers=4
        # yields exactly 1 xattn layer (4 // 4 = 1).
        self.assertEqual(len(model._xattn_layers), 1)
        self.assertEqual(model._tokenizer.pad_token, model._tokenizer.eos_token)
        # _fc must be None when no dataset is supplied
        self.assertIsNone(model._fc)

    def test_forward_smoke_with_dataset_batch(self):
        """forward() returns all required keys with correct batch and class dimensions."""
        model = TestableMedFlamingo(dataset=self.dataset)
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            output = model(**batch)

        self.assertIn("loss", output)
        self.assertIn("y_prob", output)
        self.assertIn("y_true", output)
        self.assertIn("logit", output)
        # Batch dimension
        self.assertEqual(output["logit"].shape[0], 2)
        self.assertEqual(output["y_prob"].shape[0], 2)
        self.assertEqual(output["y_true"].shape[0], 2)
        # Class dimension must match the vocabulary size inferred by the processor
        expected_num_classes = self.dataset.output_processors["answer"].size()
        self.assertEqual(output["logit"].shape[1], expected_num_classes)
        self.assertEqual(output["y_prob"].shape[1], expected_num_classes)

    def test_generate_smoke_single_image(self):
        """generate() returns a non-empty string for a single image + prompt."""
        model = TestableMedFlamingo(dataset=None)
        response = model.generate(
            images=[torch.randn(3, 16, 16)],
            prompt="what does the image show",
            max_new_tokens=8,
        )

        self.assertIsInstance(response, str)
        self.assertIn("synthetic answer", response)

    def test_generate_smoke_with_few_shot_examples(self):
        """generate() returns a string when few-shot context images are provided."""
        model = TestableMedFlamingo(dataset=None)
        response = model.generate(
            images=[torch.randn(3, 16, 16)],
            prompt="what is the main finding",
            few_shot_examples=[
                {
                    "image": torch.randn(3, 16, 16),
                    "text": "Q: is there a fracture?\nA: no",
                }
            ],
            max_new_tokens=8,
        )

        self.assertIsInstance(response, str)
        self.assertIn("synthetic answer", response)

    def test_generate_uses_inputs_embeds(self):
        """generate() passes inputs_embeds (not input_ids) so xattn conditioning applies."""
        seen_kwargs = {}

        original_generate = FakeLanguageModel.generate

        def patched_generate(self, **kwargs):
            seen_kwargs.update(kwargs)
            return original_generate(self, **kwargs)

        model = TestableMedFlamingo(dataset=None)
        model._lang_model.generate = lambda **kw: (seen_kwargs.update(kw) or original_generate(model._lang_model, **kw))

        model.generate(
            images=[torch.randn(3, 16, 16)],
            prompt="is there a fracture",
            max_new_tokens=4,
        )

        self.assertIn("inputs_embeds", seen_kwargs)
        self.assertNotIn("input_ids", seen_kwargs)

    def test_gradients_flow_through_xattn_layers(self):
        """Only xattn layers and the classification head receive gradients."""
        model = TestableMedFlamingo(dataset=self.dataset)
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        output = model(**batch)
        output["loss"].backward()

        trainable_with_grad = {
            name
            for name, param in model.named_parameters()
            if param.requires_grad and param.grad is not None
        }

        # xattn layers must receive gradients
        self.assertTrue(
            any(name.startswith("_xattn_layers") for name in trainable_with_grad)
        )
        # Frozen vision encoder must NOT receive gradients
        self.assertFalse(
            any(name.startswith("_vision_encoder") for name in trainable_with_grad)
        )
        # Frozen language model must NOT receive gradients
        self.assertFalse(
            any(name.startswith("_lang_model") for name in trainable_with_grad)
        )
        # Classification head must receive gradients
        self.assertTrue(any(name.startswith("_fc") for name in trainable_with_grad))
        # No other parameters should have gradients
        self.assertEqual(
            {
                name
                for name in trainable_with_grad
                if not (name.startswith("_xattn_layers") or name.startswith("_fc"))
            },
            set(),
            msg="Unexpected parameters received gradients",
        )

    # ------------------------------------------------------------------
    # VQARADDataset integration tests
    # ------------------------------------------------------------------

    def test_forward_smoke_with_vqarad_dataset_batch(self):
        """forward() works end-to-end on a batch from the VQARADDataset pipeline."""
        samples = self._build_vqarad_sample_dataset()
        try:
            model = TestableMedFlamingo(dataset=samples)
            loader = get_dataloader(samples, batch_size=2, shuffle=False)
            batch = next(iter(loader))

            with torch.no_grad():
                output = model(**batch)

            self.assertIn("loss", output)
            self.assertIn("y_prob", output)
            self.assertIn("y_true", output)
            self.assertIn("logit", output)
            self.assertEqual(output["logit"].shape[0], 2)
            self.assertEqual(
                output["logit"].shape[1],
                samples.output_processors["answer"].size(),
            )
        finally:
            samples.close()

    @unittest.skipUnless(
        REAL_VQARAD_ROOT,
        "set PYHEALTH_VQARAD_ROOT to run the real VQA-RAD batch smoke test",
    )
    def test_forward_with_real_vqarad_batch_if_available(self):
        real_cache_dir = tempfile.mkdtemp()
        try:
            dataset = VQARADDataset(
                root=REAL_VQARAD_ROOT,
                cache_dir=real_cache_dir,
                num_workers=1,
                dev=True,
            )
            samples = dataset.set_task(num_workers=1)
            try:
                model = TestableMedFlamingo(dataset=samples)
                loader = get_dataloader(samples, batch_size=2, shuffle=False)
                batch = next(iter(loader))

                with torch.no_grad():
                    output = model(**batch)

                self.assertIn("loss", output)
                self.assertIn("y_prob", output)
                self.assertIn("y_true", output)
                self.assertIn("logit", output)
            finally:
                samples.close()
        finally:
            shutil.rmtree(real_cache_dir)

    def test_trainer_with_small_vqarad_sample(self):
        """Trainer.train() and Trainer.evaluate() complete without error on tiny data."""
        samples = self._build_vqarad_sample_dataset()
        try:
            train_dataset, val_dataset, test_dataset = split_by_sample(
                samples,
                [0.5, 0.25, 0.25],
                seed=42,
            )
            train_loader = get_dataloader(train_dataset, batch_size=2, shuffle=True)
            val_loader = get_dataloader(val_dataset, batch_size=2, shuffle=False)
            test_loader = get_dataloader(test_dataset, batch_size=2, shuffle=False)

            model = TestableMedFlamingo(dataset=samples)
            trainer = Trainer(
                model=model,
                metrics=["accuracy"],
                device="cpu",
                enable_logging=False,
            )
            trainer.train(
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                epochs=1,
                load_best_model_at_last=False,
            )
            scores = trainer.evaluate(test_loader)

            self.assertIn("loss", scores)
            self.assertIn("accuracy", scores)
        finally:
            samples.close()


if __name__ == "__main__":
    unittest.main()

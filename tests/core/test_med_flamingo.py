import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from pyhealth.models import MedFlamingo


class FakeTokenizer:
    def __init__(self):
        self.token_to_id = {"<PAD>": 0}
        self.id_to_token = {0: "<PAD>"}

    def _encode(self, text):
        tokens = text.replace("\n", " \n ").split()
        if not tokens:
            return [0]
        ids = []
        for token in tokens:
            if token not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
            ids.append(self.token_to_id[token])
        return ids

    def __call__(
        self,
        texts,
        return_tensors="pt",
        padding=False,
        truncation=False,
    ):
        del truncation
        if return_tensors != "pt":
            raise ValueError("FakeTokenizer only supports return_tensors='pt'")
        if isinstance(texts, str):
            texts = [texts]

        encoded = [self._encode(text) for text in texts]
        max_len = max(len(ids) for ids in encoded) if padding else None

        input_ids = []
        attention_mask = []
        for ids in encoded:
            if max_len is not None:
                pad_len = max_len - len(ids)
                input_ids.append(ids + [0] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)
            else:
                input_ids.append(ids)
                attention_mask.append([1] * len(ids))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id == 0:
                continue
            tokens.append(self.id_to_token.get(int(token_id), f"tok{token_id}"))
        return " ".join(tokens).strip() or "decoded"


class FakeLangEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gated_cross_attn_layers = torch.nn.ModuleList(
            [torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)]
        )


class FakeConfig(dict):
    def __init__(self):
        super().__init__(model_type="llama", tie_word_embeddings=False)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def to_dict(self):
        return dict(self)


class FakeOpenFlamingoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lang_encoder = FakeLangEncoder()
        self.perceiver = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 4),
        )
        self.input_proj = torch.nn.Linear(1, 4)
        self.output_head = torch.nn.Linear(4, 256)
        self.loss_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.config = FakeConfig()
        self.generation_config = SimpleNamespace()

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        del kwargs
        return {"input_ids": input_ids}

    def forward(
        self,
        vision_x=None,
        lang_x=None,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        del kwargs
        del vision_x, attention_mask
        if lang_x is None:
            lang_x = input_ids
        if lang_x is None:
            raise ValueError("FakeOpenFlamingoModel.forward expected lang_x/input_ids")
        hidden = self.input_proj(lang_x.float().unsqueeze(-1) / 100.0)
        hidden = self.lang_encoder.gated_cross_attn_layers[0](hidden)
        hidden = torch.relu(hidden)
        hidden = self.perceiver(hidden)
        hidden = self.lang_encoder.gated_cross_attn_layers[1](hidden)
        logits = self.output_head(hidden) * self.loss_scale

        if labels is None:
            loss = logits.mean() * 0.0
        else:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return SimpleNamespace(loss=loss, logits=logits)

    def generate(self, vision_x, lang_x, attention_mask=None, **kwargs):
        del vision_x, attention_mask, kwargs
        suffix = torch.tensor([[5, 6]], device=lang_x.device)
        return torch.cat([lang_x, suffix], dim=1)


class OOMGenerateModel(FakeOpenFlamingoModel):
    def generate(self, vision_x, lang_x, attention_mask=None, **kwargs):
        del vision_x, lang_x, attention_mask, kwargs
        raise RuntimeError("CUDA out of memory")


class TestMedFlamingo(unittest.TestCase):
    def _create_temp_image(self, root: Path, name: str = "sample.png") -> str:
        image_path = root / name
        Image.new("RGB", (8, 8), color=(100, 100, 100)).save(image_path)
        return str(image_path)

    def _make_model(
        self,
        *,
        enable_lora: bool = False,
        lora_target_modules=None,
    ) -> MedFlamingo:
        with patch.object(MedFlamingo, "_initialize_components", return_value=None):
            model = MedFlamingo(
                dataset=None,
                llama_path="/tmp/unused",
                enable_lora=enable_lora,
                lora_target_modules=lora_target_modules,
            )
        model.model = FakeOpenFlamingoModel()
        model.tokenizer = FakeTokenizer()
        model.image_processor = lambda image: torch.zeros((3, 8, 8), dtype=torch.float32)
        model._effective_quantization = "fp16"
        model.to("cpu")
        return model

    def test_lazy_import_guard_for_8bit(self):
        original_find_spec = importlib.util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            if name == "bitsandbytes":
                return None
            return original_find_spec(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as llama_dir:
            with patch(
                "pyhealth.models.med_flamingo.importlib.util.find_spec",
                side_effect=fake_find_spec,
            ):
                with self.assertRaises(ImportError):
                    MedFlamingo(
                        dataset=None,
                        llama_path=llama_dir,
                        quantization="8bit",
                    )

    def test_target_module_resolution(self):
        model = self._make_model()
        target_modules = model._resolve_lora_target_modules()

        self.assertTrue(any("gated_cross_attn_layers" in name for name in target_modules))
        self.assertTrue(any("perceiver" in name for name in target_modules))

    def test_target_module_resolution_zero_match(self):
        model = self._make_model(lora_target_modules=["not.a.real.module"])
        with self.assertRaises(ValueError):
            model._resolve_lora_target_modules()

    def test_train_forward_returns_scalar_loss(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = self._create_temp_image(Path(temp_dir))
            model = self._make_model()

            model.train()
            outputs = model.forward(
                image_path=[image_path],
                question=["what is shown?"],
                answer=["a lesion"],
                question_id=["q1"],
            )

            self.assertIn("loss", outputs)
            self.assertEqual(outputs["loss"].dim(), 0)
            outputs["loss"].backward()
            self.assertIsNotNone(model.model.loss_scale.grad)

    def test_adapter_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = self._create_temp_image(root)
            adapter_dir = root / "adapter"

            model = self._make_model()
            model.configure_lora()
            model.save_lora_adapter(str(adapter_dir))

            self.assertTrue((adapter_dir / "adapter_config.json").exists())
            self.assertTrue((adapter_dir / "med_flamingo_config.json").exists())

            model_reloaded = self._make_model()
            model_reloaded.load_lora_adapter(str(adapter_dir), is_trainable=False)
            model_reloaded.eval()
            outputs = model_reloaded.forward(
                image_path=[image_path],
                question=["what is shown?"],
                answer=["ground truth"],
                question_id=["q2"],
            )

            self.assertIn("generated_text", outputs)
            self.assertEqual(len(outputs["generated_text"]), 1)
            self.assertNotEqual(outputs["generated_text"][0], "")

    def test_fit_lora_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = self._create_temp_image(root)
            model = self._make_model()
            model.configure_lora()

            train_batches = [
                {
                    "image_path": [image_path],
                    "question": ["what is shown?"],
                    "answer": ["lesion"],
                    "question_id": ["qfit"],
                }
            ]
            summary = model.fit_lora(
                train_dataloader=train_batches,
                val_dataloader=None,
                epochs=1,
                output_dir=str(root / "outputs"),
            )

            self.assertIn("best_metric", summary)
            self.assertTrue((root / "outputs" / "best_adapter" / "adapter_config.json").exists())
            self.assertTrue((root / "outputs" / "last_adapter" / "adapter_config.json").exists())
            self.assertTrue((root / "outputs" / "metrics_history.json").exists())
            self.assertTrue((root / "outputs" / "fit_summary.json").exists())

    def test_oom_message_generation_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = self._create_temp_image(Path(temp_dir))
            model = self._make_model()
            model.model = OOMGenerateModel()
            model.eval()

            with self.assertRaisesRegex(RuntimeError, "quantization='8bit'"):
                model.forward(
                    image_path=[image_path],
                    question=["what is shown?"],
                    answer=["ground truth"],
                    question_id=["q3"],
                )


if __name__ == "__main__":
    unittest.main()

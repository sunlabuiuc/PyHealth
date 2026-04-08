import shutil
import tempfile
import unittest
import warnings
from types import SimpleNamespace

from PIL import Image
import torch
import torch.nn as nn

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.base_model import BaseModel
from pyhealth.models.medflamingo import MedFlamingo

warnings.filterwarnings(
    "ignore",
    message=r"A newer version of litdata is available .*",
    category=UserWarning,
)


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

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=16, **kwargs):
        batch_size = input_ids.shape[0]
        generated = torch.full(
            (batch_size, min(max_new_tokens, 4)),
            fill_value=7,
            dtype=torch.long,
            device=input_ids.device,
        )
        return generated


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


class TestMedFlamingo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.samples = []
        labels = ["yes", "no", "yes", "no"]
        questions = [
            "is there a fracture",
            "is the study normal",
            "is there consolidation",
            "is there edema",
        ]

        for idx, (answer, question) in enumerate(zip(labels, questions)):
            image_path = f"{cls.temp_dir}/img_{idx}.png"
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

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_model_initialization_standalone(self):
        model = TestableMedFlamingo(dataset=None)
        self.assertIsInstance(model, MedFlamingo)
        self.assertIsInstance(model, BaseModel)
        self.assertEqual(model.vision_model_name, "openai/clip-vit-large-patch14")
        self.assertEqual(model.lang_model_name, "facebook/opt-6.7b")
        self.assertEqual(len(model._xattn_layers), 1)
        self.assertEqual(model._tokenizer.pad_token, model._tokenizer.eos_token)
        #TODO: should we mirror the intended production hidden sizes more closely?

    def test_forward_smoke_with_dataset_batch(self):
        model = TestableMedFlamingo(dataset=self.dataset)
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            output = model(**batch)

        self.assertIn("loss", output)
        self.assertIn("y_prob", output)
        self.assertIn("y_true", output)
        self.assertIn("logit", output)
        self.assertEqual(output["logit"].shape[0], 2)
        self.assertEqual(output["y_prob"].shape[0], 2)
        self.assertEqual(output["y_true"].shape[0], 2)
        self.assertEqual(
            output["logit"].shape[1],
            self.dataset.output_processors["answer"].size(),
        )
        #TODO: should we also pin an expected class count here once the vqa-rad answer?

    def test_generate_smoke_single_image(self):
        model = TestableMedFlamingo(dataset=None)
        response = model.generate(
            images=[torch.randn(3, 16, 16)],
            prompt="what does the image show",
            max_new_tokens=8,
        )

        self.assertIsInstance(response, str)
        self.assertIn("synthetic answer", response)

    def test_generate_smoke_with_few_shot_examples(self):
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
        #TODO: should we assert a more specific few-shot prompt format?

    def test_gradients_flow_through_xattn_layers(self):
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

        self.assertTrue(
            any(name.startswith("_xattn_layers") for name in trainable_with_grad)
        )
        self.assertFalse(
            any(name.startswith("_vision_encoder") for name in trainable_with_grad)
        )
        self.assertFalse(
            any(name.startswith("_lang_model") for name in trainable_with_grad)
        )
        self.assertTrue(any(name.startswith("_fc") for name in trainable_with_grad))
        self.assertEqual(
            {
                name
                for name in trainable_with_grad
                if not (name.startswith("_xattn_layers") or name.startswith("_fc"))
            },
            set(),
        )
        #TODO: should this be phrased as xattn-only, or xattn-plus-classification-head for the multiclass path?


if __name__ == "__main__":
    unittest.main()

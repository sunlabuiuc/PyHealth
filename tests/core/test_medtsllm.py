"""Tiny offline GPT-2/Llama model, gradients, Trainer and checkpoint tests.

Contributors: Thi Van Trinh Tran and Zexi Li.
All weights are initialized locally from tiny configs, never downloaded.
"""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from transformers import GPT2Config, GPT2Model, LlamaConfig, LlamaModel

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.medtsllm import MedTsLLM
from pyhealth.tasks.ecg_delineation_ludb import ECGDelineationLUDB
from pyhealth.trainer import Trainer


class _Tokenizer:
    pad_token, eos_token = "<pad>", "<eos>"

    def __call__(self, texts, **kwargs):
        width = max(1, max(len(text) for text in texts))
        return SimpleNamespace(input_ids=torch.tensor([
            [1 + ord(char) % 30 for char in text] + [0] * (width - len(text))
            for text in texts
        ], dtype=torch.long))


def _dataset(all_ignored=False):
    samples = []
    for pid in (1, 2):
        labels = np.arange(512) % 4
        ignored = np.ones(512, dtype=bool) if all_ignored else np.zeros(512, dtype=bool)
        ignored[:7] = True
        samples.append({
            "patient_id": str(pid), "record_id": str(pid),
            "signal": np.sin(np.arange(512) / 20 + pid).astype(np.float32),
            "dense_label": labels, "ignore_mask": ignored,
            "label": {"time": np.flatnonzero(~ignored).tolist(),
                      "values": labels[~ignored].tolist()},
            "patient_context": f'{{"ID":{pid}}}', "lead_index": 0,
            "start_sample": 0, "end_sample_exclusive": 512, "manifest_order": 0,
        })
    return create_sample_dataset(
        samples, ECGDelineationLUDB.input_schema, ECGDelineationLUDB.output_schema,
        in_memory=True,
    )


def _model(dataset, kind="gpt2", prompt="none", backbone=None):
    if backbone is None:
        if kind == "gpt2":
            backbone = GPT2Model(GPT2Config(
                vocab_size=32, n_embd=8, n_layer=1, n_head=2, n_positions=1024,
                resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
            ))
        else:
            backbone = LlamaModel(LlamaConfig(
                vocab_size=32, hidden_size=8, intermediate_size=16,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=1024, attention_dropout=0, use_cache=False,
            ))
    return MedTsLLM(
        dataset, llm_name="injected-synthetic", backbone=backbone,
        tokenizer=_Tokenizer(), prompt_mode=prompt, d_model=4, num_tokens=4,
        dropout=0,
    )


class TestMedTsLLM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dataset = _dataset()
        self.batch = next(iter(get_dataloader(self.dataset, batch_size=2)))

    def test_both_backbones_masked_loss_and_adapter_gradients(self):
        for kind in ("gpt2", "llama"):
            with self.subTest(kind=kind):
                model = _model(self.dataset, kind).train()
                self.assertFalse(model.llm.training)
                output = model(**self.batch)
                self.assertEqual(output["dense_logit"].shape, (2, 4, 512))
                valid = ~self.batch["ignore_mask"]
                self.assertEqual(output["logit"].shape, (1010, 4))
                expected = F.cross_entropy(
                    output["dense_logit"].permute(0, 2, 1)[valid],
                    self.batch["dense_label"][valid],
                )
                torch.testing.assert_close(output["loss"], expected)
                output["loss"].backward()
                self.assertTrue(all(parameter.grad is None
                                    for parameter in model.llm.parameters()))
                self.assertIsNotNone(model.patch_embedding.weight.grad)
                self.assertIsNotNone(model.mapping_layer.weight.grad)
                self.assertIsNotNone(model.output_projection[1].weight.grad)

    def test_packed_mask_or_dense_target_disagreement_is_an_error(self):
        model = _model(self.dataset)
        changed = dict(self.batch)
        changed["ignore_mask"] = self.batch["ignore_mask"].clone()
        changed["ignore_mask"][0, 7] = True
        with self.assertRaisesRegex(ValueError, "times disagree"):
            model(**changed)
        changed = dict(self.batch)
        changed["dense_label"] = self.batch["dense_label"].clone()
        changed["dense_label"][0, 7] = 0
        with self.assertRaisesRegex(ValueError, "dense frozen"):
            model(**changed)

    def test_all_ignored_batch_does_not_update_adam_momentum(self):
        model = _model(self.dataset)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
        model(**self.batch)["loss"].backward()
        optimizer.step()  # establish non-zero optimizer momentum first
        optimizer.zero_grad(set_to_none=True)
        before = {name: parameter.detach().clone()
                  for name, parameter in model.named_parameters()}
        state = copy.deepcopy(optimizer.state_dict())
        ignored_batch = next(iter(get_dataloader(_dataset(True), batch_size=2)))
        output = model(**ignored_batch)
        self.assertEqual(output["supervised_points"], 0)
        self.assertEqual(output["logit"].shape, (0, 4))
        output["loss"].backward()
        optimizer.step()
        for name, parameter in model.named_parameters():
            self.assertIsNone(parameter.grad)
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        after = optimizer.state_dict()
        self.assertEqual(state["param_groups"], after["param_groups"])
        for key, values in state["state"].items():
            for field, value in values.items():
                torch.testing.assert_close(
                    value, after["state"][key][field], rtol=0, atol=0,
                )

    def test_standard_checkpoint_includes_backbone_and_refreshes_buffer(self):
        model = _model(self.dataset).eval()
        original = model(**self.batch)["dense_logit"].detach()
        self.assertTrue(any(key.startswith("llm.") for key in model.state_dict()))
        other = _model(self.dataset).eval()
        other.load_state_dict(model.state_dict())
        torch.testing.assert_close(other.word_embeddings,
                                   other.llm.get_input_embeddings().weight)
        torch.testing.assert_close(other(**self.batch)["dense_logit"], original)

    def test_compact_round_trip_and_wrong_backbone_rejection(self):
        model = _model(self.dataset).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapters.pt"
            model.save_adapters(path, provenance={"seed": 0})
            other = _model(self.dataset, backbone=copy.deepcopy(model.llm)).eval()
            self.assertEqual(other.load_adapters(path), {"seed": 0})
            torch.testing.assert_close(other(**self.batch)["dense_logit"],
                                       model(**self.batch)["dense_logit"])
            with self.assertRaises(FileExistsError):
                model.save_adapters(path)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                _model(self.dataset).load_adapters(path)
            wrong_prompt = _model(self.dataset, prompt="all",
                                  backbone=copy.deepcopy(model.llm))
            with self.assertRaisesRegex(ValueError, "configuration"):
                wrong_prompt.load_adapters(path)

    def test_historical_adapter_format_is_not_silently_partially_loaded(self):
        model = _model(self.dataset)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"model_state": {}}, path)
            with self.assertRaisesRegex(ValueError, "Unverified"):
                model.load_adapters(path)

    def test_exact_prompt_components_and_intentional_removal(self):
        model = _model(self.dataset, prompt="all")
        signal = torch.arange(512, dtype=torch.float32).unsqueeze(0)
        prompt = model._prompt_text(signal, ['{"ID":1}'])[0]
        self.assertTrue(prompt.startswith(
            "Dataset: LUDB 12-lead ECG, sampled at 500 Hz. "
        ))
        self.assertIn(
            "Task: label every ECG sample as background, P, QRS, or T.", prompt,
        )
        self.assertIn('Patient information: {"ID":1}', prompt)
        self.assertIn("Signal summary: mean=255.5;std=", prompt)
        removed = model._prompt_text(signal, [""])[0]
        self.assertNotIn("Patient information:", removed)
        self.assertIn("Signal summary:", removed)

    def test_native_trainer_one_tiny_step_and_evaluate(self):
        model = _model(self.dataset)
        trainer = Trainer(
            model, device="cpu", metrics=["accuracy"], enable_logging=False,
        )
        loader = get_dataloader(self.dataset, batch_size=2)
        trainer.train(
            loader, epochs=1, steps_per_epoch=1, load_best_model_at_last=False,
        )
        scores = trainer.evaluate(loader)
        self.assertIn("loss", scores)
        self.assertTrue(np.isfinite(scores["loss"]))

    def test_reordered_class_vocabulary_is_not_silently_accepted(self):
        changed = copy.copy(self.dataset)
        changed.output_processors = dict(self.dataset.output_processors)
        changed.output_processors["label"] = copy.deepcopy(
            self.dataset.output_processors["label"]
        )
        changed.output_processors["label"].base.label_vocab = {
            3: 0, 2: 1, 1: 2, 0: 3,
        }
        with self.assertRaisesRegex(ValueError, "class order"):
            _model(changed)

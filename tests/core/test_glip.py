"""Tests for the GLIP-T port and the MIU-VL prompts (random weights, offline)."""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image
from transformers import BertTokenizerFast

from pyhealth.datasets import create_sample_dataset
from pyhealth.models import GLIP, GLIPDetector, build_caption
from pyhealth.models import glip as glip_module
from pyhealth.models.miu_vl import (
    MLM_TEMPLATES,
    POLYP_MANUAL_PROMPT,
    mlm_attributes,
    mlm_prompts,
)

_WORDS = ["pink", "yellow", "red", "color", "oval", "bump", "in", "rectum", "polyp", ".", ","]


def _tokenizer(directory):
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *_WORDS]
    try:  # transformers 5 takes the vocabulary itself
        tok = BertTokenizerFast(vocab={w: i for i, w in enumerate(words)})
    except TypeError:
        tok = None
    if tok is None or len(tok.get_vocab()) != len(words):
        vocab = Path(directory) / "vocab.txt"
        vocab.write_text("\n".join(words))
        tok = BertTokenizerFast(vocab_file=str(vocab))
    return tok


class TestHelpers(unittest.TestCase):
    def test_build_caption(self):
        caption, spans = build_caption([POLYP_MANUAL_PROMPT, "polyp"])
        self.assertEqual(caption, "pink yellow red color oval bump in rectum. polyp")
        self.assertEqual([caption[a:b] for a, b in spans], ["pink yellow red color oval bump", "polyp"])

    def test_phrase_tokens(self):
        with tempfile.TemporaryDirectory() as d:
            tok = _tokenizer(d)
            caption, spans = build_caption([POLYP_MANUAL_PROMPT, "polyp"])
            enc = tok([caption])[0]
            self.assertEqual(glip_module._phrase_tokens(enc, spans), [[1, 2, 3, 4, 5, 6], [10]])
            with self.assertRaises(ValueError):
                glip_module._phrase_tokens(enc, [(500, 505)])

    def test_resize_shape(self):
        # GLIP's test resize: short side 800, long side at most 1333.
        self.assertEqual(glip_module._resize_shape(574, 500), (800, 918))
        self.assertEqual(glip_module._resize_shape(500, 574), (918, 800))
        self.assertEqual(glip_module._resize_shape(2000, 500), (333, 1332))
        self.assertEqual(glip_module._resize_shape(800, 900), (900, 800))

    def test_anchors_and_decode(self):
        anchors = glip_module._anchors([(2, 3)] * 5, torch.device("cpu"))
        # Detectron anchors: stride 8, size 64, centred on (stride - 1) / 2.
        np.testing.assert_allclose(anchors[0][0], [-28.0, -28.0, 35.0, 35.0])
        np.testing.assert_allclose(anchors[0][1], [-20.0, -28.0, 43.0, 35.0])
        self.assertEqual(anchors[4].shape, (6, 4))
        decoded = glip_module._decode(torch.zeros(1, 4), anchors[0][:1])
        np.testing.assert_allclose(decoded, anchors[0][:1], atol=1e-5)
        big = glip_module._decode(torch.tensor([[0.0, 0.0, 100.0, 0.0]]), anchors[0][:1])
        self.assertAlmostEqual(float(big[0, 2] - big[0, 0] + 1), 64 * 1000 / 16, places=2)

    def test_fit_reads_raw_memory(self):
        t = torch.arange(2 * 2 * 4 * 4, dtype=torch.float32).view(2, 2, 4, 4)
        self.assertIs(glip_module._fit(t, 4, 4), t)
        small = glip_module._fit(t, 2, 2)
        np.testing.assert_array_equal(small[1].reshape(-1), t[1].reshape(-1)[:8])

    def test_postprocess_nms_and_scores(self):
        anchors = [torch.tensor([[0.0, 0.0, 9.0, 9.0], [1.0, 0.0, 10.0, 9.0], [30.0, 30.0, 39.0, 39.0]])]
        boxes = [torch.zeros(1, 4, 1, 3)]
        centers = [torch.full((1, 1, 1, 3), 10.0)]
        logits = [torch.tensor([[[3.0, 3.0, -9.0], [2.0, 2.0, -9.0], [-9.0, -9.0, 4.0]]])]
        out = glip_module._postprocess(boxes, centers, logits, anchors, [[0, 1], [2]], (40, 40))
        # Anchors 0 and 1 overlap (IoU 9/11 with +1 pixels): NMS keeps one.
        self.assertEqual(sorted(out["labels"].tolist()), [0, 1])
        expected = math.sqrt(torch.sigmoid(torch.tensor(3.0)) * torch.sigmoid(torch.tensor(10.0)))
        self.assertAlmostEqual(float(out["scores"].max()), max(expected, math.sqrt(
            torch.sigmoid(torch.tensor(4.0)) * torch.sigmoid(torch.tensor(10.0)))), places=5)

    def test_dyrelu_and_attention_mask(self):
        relu = glip_module._DyReLU(8)
        x = torch.randn(2, 8, 3, 3)
        self.assertEqual(relu(x).shape, x.shape)
        attn = glip_module._BiMultiHeadAttention(16, 12, 32, 4).eval()
        v, lang = torch.randn(1, 5, 16), torch.randn(1, 3, 12)
        mask = torch.tensor([[1, 1, 0]])
        out_v, _ = attn(v, lang, mask)
        lang2 = lang.clone()
        lang2[0, 2] = 100.0
        out_v2, _ = attn(v, lang2, mask)
        torch.testing.assert_close(out_v, out_v2)


class TestGLIP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.tok = _tokenizer(cls.tmp.name)
        torch.manual_seed(0)
        cls.detector = GLIPDetector(tokenizer=cls.tok).eval()
        cls.image = Path(cls.tmp.name) / "img.png"
        rng = np.random.default_rng(0)
        Image.fromarray(rng.integers(0, 255, (40, 56, 3), dtype=np.uint8)).save(cls.image)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_preprocess(self):
        x = self.detector.preprocess(str(self.image))
        self.assertEqual(tuple(x.shape), (3, 800, 1120))
        arr = np.asarray(Image.open(self.image).convert("RGB").resize((1120, 800), Image.BILINEAR))
        bgr = torch.tensor(arr[..., ::-1].copy()).permute(2, 0, 1).float()
        mean = torch.tensor(glip_module._PIXEL_MEAN).view(3, 1, 1)
        std = torch.tensor(glip_module._PIXEL_STD).view(3, 1, 1)
        torch.testing.assert_close(x, (bgr - mean) / std)
        same = self.detector.preprocess(np.asarray(Image.open(self.image).convert("RGB")))
        torch.testing.assert_close(x, same)

    def test_checkpoint_round_trip(self):
        state = {f"module.{k}": v for k, v in self.detector.state_dict().items()}
        state["module.rpn.anchor_generator.cell_anchors.0"] = torch.zeros(1, 4)
        state["module.language_backbone.body.model.embeddings.position_ids"] = torch.arange(512)
        path = Path(self.tmp.name) / "ckpt.pth"
        torch.save({"model": state, "iteration": 1}, path)
        other = GLIPDetector(tokenizer=self.tok)
        other.load_checkpoint(str(path))
        for (k, a), b in zip(self.detector.state_dict().items(), other.state_dict().values()):
            torch.testing.assert_close(a, b, msg=k)

    def test_model_forward(self):
        samples = [{"patient_id": "p", "image": str(self.image), "boxes": [[1.0, 2.0, 3.0, 4.0]]}]
        dataset = create_sample_dataset(samples, {"image": "raw"}, {"boxes": "raw"})
        model = GLIP(dataset, prompt=[POLYP_MANUAL_PROMPT, "polyp"], tokenizer=self.tok)
        model.detector = self.detector
        with torch.no_grad():
            out = model(image=[str(self.image)], boxes=[[[1.0, 2.0, 3.0, 4.0]]])
        pred = out["y_pred"][0]
        self.assertEqual(out["y_true"], [{"boxes": [[1.0, 2.0, 3.0, 4.0]]}])
        self.assertLessEqual(len(pred["scores"]), 100)
        self.assertEqual(pred["scores"], sorted(pred["scores"], reverse=True))
        self.assertTrue(set(pred["labels"]) <= {0, 1})
        boxes = np.array(pred["boxes"]).reshape(-1, 4)
        self.assertTrue((boxes[:, 2:] > 0).all() and (boxes[:, 0] + boxes[:, 2] <= 56).all())
        # Per-sample prompts replace the fixed prompt.
        with torch.no_grad():
            per = model(image=[str(self.image)], prompt=[["polyp"]])["y_pred"][0]
        self.assertTrue(set(per["labels"]) <= {0})
        model.prompt = None
        with self.assertRaises(ValueError):
            model(image=[str(self.image)])

    def test_pretrained_download_and_path(self):
        with mock.patch.object(GLIPDetector, "load_checkpoint") as load, mock.patch(
            "huggingface_hub.hf_hub_download", return_value="/tmp/glip.pth"
        ) as download, mock.patch.object(GLIPDetector, "__init__", return_value=None):
            GLIP(pretrained=True)
            download.assert_called_once_with(
                glip_module.HF_REPO_ID, glip_module.HF_FILENAME, revision=glip_module.HF_REVISION
            )
            load.assert_called_with("/tmp/glip.pth")
            GLIP(pretrained="/local/glip.pth")
            load.assert_called_with("/local/glip.pth")


class _FakeMLM(torch.nn.Module):
    """Predicts a fixed ranking and records how it was called."""

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.calls = []
        self.device = torch.device("cpu")

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        ids = kwargs["input_ids"]
        logits = torch.zeros(1, ids.shape[1], self.vocab_size)
        logits[..., 5:9] = torch.tensor([4.0, 3.0, 2.0, 1.0])
        return SimpleNamespace(logits=logits)


class TestMIUVLPrompts(unittest.TestCase):
    def test_mlm_prompts_from_attributes(self):
        words = {"color": ["pink", "red"], "shape": ["oval", "round"], "location": ["rectum", "colon"]}
        prompts = mlm_prompts("polyp", "bump", topk=2, attributes=words)
        self.assertEqual(prompts[1], {"prefix": "red color, round shape ", "name": "bump",
                                      "suffix": " in colon"})
        self.assertEqual(build_caption(prompts[:1])[0], "pink color, oval shape bump in rectum")
        self.assertEqual(mlm_prompts("polyp", topk=1, attributes=words)[0]["name"], "polyp")

    def test_mlm_attributes_uses_attention_mask(self):
        with tempfile.TemporaryDirectory() as d:
            tok = _tokenizer(d)
            model = _FakeMLM(len(tok.get_vocab()))
            words = mlm_attributes("polyp", topk=2, model=model, tokenizer=tok)
        self.assertEqual(set(words), set(MLM_TEMPLATES))
        self.assertEqual(words["color"], ["pink", "yellow"])
        # MIU-VL's script passed the segment ids as the attention mask.
        for call in model.calls:
            self.assertTrue(bool(call["attention_mask"].all()))
        with self.assertRaises(ValueError):
            mlm_attributes("polyp", model=model)

    def test_mlm_attributes_loads_model(self):
        with tempfile.TemporaryDirectory() as d:
            tok = _tokenizer(d)
            model = _FakeMLM(len(tok.get_vocab()))
            with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tok), \
                    mock.patch("transformers.AutoModelForMaskedLM.from_pretrained", return_value=model):
                prompts = mlm_prompts("polyp", "bump", topk=1)
        self.assertEqual(prompts, [{"prefix": "pink color, pink shape ", "name": "bump",
                                    "suffix": " in pink"}])


if __name__ == "__main__":
    unittest.main()

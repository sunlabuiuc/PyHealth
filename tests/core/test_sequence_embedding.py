import os
import tempfile
import unittest
from typing import Any, cast

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import LevelEmbedding, SequenceEmbeddingModel
from pyhealth.processors import LevelConfig, SequenceProcessorV2

NAN = float("nan")
# stays (series) -> labs (series) with tokens (test, unit) and a value
LABS: list[LevelConfig] = [
    {"ordering": "series", "tokens": 2, "values": [1]},
    {"ordering": "series"},
]
# stays -> ECG windows [leads, samples], spatial along the samples, with a rhythm token
ECG: list[LevelConfig] = [
    {"ordering": "series", "tokens": 1, "values": [2, 4], "spatial": [False, True]},
    {"ordering": "sequence"},
]


def _dataset(**fields: Any):
    schema = {
        name: ("sequence_v2", {"levels": levels})
        for name, (levels, _) in fields.items()
    }
    samples = [
        {
            "patient_id": f"p{i}",
            **{name: records[i] for name, (_, records) in fields.items()},
            "y": i % 2,
        }
        for i in range(len(next(iter(fields.values()))[1]))
    ]
    return create_sample_dataset(samples, schema, {"y": "binary"})


def _labs(*stays: list[tuple[float, str, str, float]]) -> dict:
    return {
        "time": [10.0 * i for i in range(len(stays))],
        "children": [
            {
                "time": [row[0] for row in stay],
                "tokens": [(row[1], row[2]) for row in stay],
                "values": [[row[3]] for row in stay],
            }
            for stay in stays
        ],
    }


LAB_RECORDS = [
    _labs(
        [(0.0, "Glucose", "mg/dL", 100.0), (1.0, "WBC", "10^3/uL", 7.0)],
        [(2.0, "Glucose", "mg/dL", 90.0)],
    ),
    _labs([(0.0, "WBC", "10^3/uL", 8.0)]),
]
LAB_DIMS: dict[str, list[LevelEmbedding]] = {
    "labs": [{"tokens": (4, 2), "values": 3}, {}]
}


def _layer(model: SequenceEmbeddingModel, field: str, level: int, name: str) -> Any:
    """One embedding layer of a field's level (``token_<k>``, ``value_proj``, ...)."""
    return cast(Any, model.embedding_layers[field]).layer(level, name)


def _batch(dataset, field: str):
    proc = dataset.input_processors[field]
    assert isinstance(proc, SequenceProcessorV2)
    return proc.collate([dataset[i][field] for i in range(len(dataset))])


class TestConfiguration(unittest.TestCase):
    def setUp(self):
        self.dataset = _dataset(labs=(LABS, LAB_RECORDS))

    def test_rejects_bad_configurations(self):
        bad: list[dict[str, Any]] = [
            {"embedding_dim": {}},  # labs missing
            {"embedding_dim": {**LAB_DIMS, "other": [{}]}},
            {"embedding_dim": {"labs": [{"tokens": (4, 2), "values": 3}]}},  # one level
            {"embedding_dim": {"labs": [{"tokens": (4,), "values": 3}, {}]}},  # K = 2
            {"embedding_dim": {"labs": [{"tokens": (4, 0), "values": 3}, {}]}},
            {"embedding_dim": {"labs": [{"tokens": (4, 2)}, {}]}},  # values missing
            {
                "embedding_dim": {
                    "labs": [{"tokens": (4, 2), "values": 3}, {"values": 2}]
                }
            },
            {
                "embedding_dim": {
                    "labs": [{"tokens": (4, 2), "values": 3}, {"tokens": (2,)}]
                }
            },  # level 1 has no tokens
            {
                "embedding_dim": {
                    "labs": [{"tokens": (4, 2), "values": 3, "time": 1}, {}]
                }
            },
            {
                "embedding_dim": LAB_DIMS,
                "flatten_spatial": {"labs": [False, True]},
            },  # no spatial
            {
                "embedding_dim": LAB_DIMS,
                "flatten_spatial": {"labs": [True, False]},
            },  # level 1 has no values
            {"embedding_dim": LAB_DIMS, "flatten_spatial": {"labs": [True]}},
            {"embedding_dim": LAB_DIMS, "flatten_spatial": {"other": [True]}},
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SequenceEmbeddingModel(self.dataset, **kwargs)

    def test_embedding_dim_per_level(self):
        model = SequenceEmbeddingModel(self.dataset, embedding_dim=LAB_DIMS)
        self.assertEqual(model.embedding_dim_per_level("labs"), (4 + 2 + 3, 0))

    def test_ignores_fields_of_other_processors(self):
        samples = [
            {"patient_id": "p0", "labs": LAB_RECORDS[0], "age": [60.0], "y": 0},
            {"patient_id": "p1", "labs": LAB_RECORDS[1], "age": [70.0], "y": 1},
        ]
        dataset = create_sample_dataset(
            samples,
            {"labs": ("sequence_v2", {"levels": LABS}), "age": "tensor"},
            {"y": "binary"},
        )
        model = SequenceEmbeddingModel(dataset, embedding_dim=LAB_DIMS)
        with self.assertRaises(ValueError):  # "age" is not a V2 field
            SequenceEmbeddingModel(dataset, embedding_dim={**LAB_DIMS, "age": [{}]})
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        self.assertEqual(set(model(batch)), {"labs"})


class TestLevels(unittest.TestCase):
    def test_all_empty_numeric_records(self) -> None:
        dataset = _dataset(x=([
            {"ordering": "set", "values": [2, 4], "spatial": [False, True]},
        ], [{"values": []}, {"values": []}]))
        for flatten in (True, False):
            model = SequenceEmbeddingModel(
                dataset, embedding_dim={"x": [{"values": 3}]},
                flatten_spatial={"x": [flatten]},
            )
            out = model({"x": _batch(dataset, "x")})["x"][0]
            self.assertEqual(
                out["embedding"].shape, (0, 3) if flatten else (0, 4, 3)
            )
            self.assertEqual(out["offsets"].tolist(), [0, 0, 0])
            out["embedding"].sum().backward()

    def test_level_embeddings_offsets_and_time(self):
        dataset = _dataset(labs=(LABS, LAB_RECORDS))
        model = SequenceEmbeddingModel(dataset, embedding_dim=LAB_DIMS)
        batch = _batch(dataset, "labs")
        out = model({"labs": batch})["labs"]
        self.assertEqual(out[0]["embedding"].shape, (4, 4 + 2 + 3))
        self.assertEqual(out[0]["offsets"].tolist(), [0, 2, 3, 4])
        self.assertEqual(out[0]["time"].tolist(), [0.0, 1.0, 2.0, 0.0])
        self.assertIsNone(out[1]["embedding"])  # level 1 has only time
        self.assertEqual(out[1]["offsets"].tolist(), [0, 2, 3])  # batch offsets
        self.assertEqual(out[1]["time"].tolist(), [0.0, 10.0, 0.0])

        # The embedding is the namespaces' embeddings and the projected values.
        tokens, values = batch[0], batch[1]
        expected = torch.cat(
            [
                _layer(model, "labs", 0, "token_0")(tokens[:, 0]),
                _layer(model, "labs", 0, "token_1")(tokens[:, 1]),
                _layer(model, "labs", 0, "value_proj")(values),
            ],
            dim=1,
        )
        torch.testing.assert_close(out[0]["embedding"], expected)

    def test_pad_embeds_to_zero(self):
        levels: list[LevelConfig] = [
            {
                "ordering": "series",
                "tokens": 1,
                "resample": {"step": 1.0, "method": "last"},
            },
            {},
        ]
        records = [{"children": [{"time": [0.0, 2.0], "tokens": ["A", "B"]}]}] * 2
        dataset = _dataset(codes=(levels, records))
        dims: dict[str, list[LevelEmbedding]] = {"codes": [{"tokens": (5,)}, {}]}
        model = SequenceEmbeddingModel(dataset, embedding_dim=dims)
        out = model({"codes": _batch(dataset, "codes")})["codes"][0]["embedding"]
        self.assertEqual(out.shape, (6, 5))
        self.assertTrue(torch.all(out[1] == 0))  # step 1 has no record: <pad>
        self.assertTrue(torch.any(out[0] != 0))

    def test_flatten_spatial(self):
        windows = [
            {
                "children": [
                    {
                        "time": [0.0],
                        "tokens": ["sinus"],
                        "values": torch.rand(1, 2, 4).tolist(),
                    }
                ]
            },
            {
                "children": [
                    {
                        "time": [0.0, 1.0],
                        "tokens": ["afib", "sinus"],
                        "values": torch.rand(2, 2, 4).tolist(),
                    }
                ]
            },
        ]
        dataset = _dataset(ecg=(ECG, windows))
        dims: dict[str, list[LevelEmbedding]] = {
            "ecg": [{"tokens": (3,), "values": 5}, {}]
        }
        batch = _batch(dataset, "ecg")

        flat = SequenceEmbeddingModel(dataset, embedding_dim=dims)
        self.assertEqual(flat({"ecg": batch})["ecg"][0]["embedding"].shape, (3, 3 + 5))

        model = SequenceEmbeddingModel(
            dataset, embedding_dim=dims, flatten_spatial={"ecg": [False, True]}
        )
        out = model({"ecg": batch})["ecg"][0]["embedding"]
        self.assertEqual(out.shape, (3, 4, 3 + 5))  # one embedding per sample position
        tokens, values = batch[0], batch[1]
        # The rhythm token is broadcast over the positions...
        token = _layer(model, "ecg", 0, "token_0")(tokens)
        torch.testing.assert_close(out[:, :, :3], token[:, None, :].expand(3, 4, 3))
        # ...and the leads are projected at each position.
        torch.testing.assert_close(
            out[:, :, 3:], _layer(model, "ecg", 0, "value_proj")(values.transpose(1, 2))
        )


class TestMask(unittest.TestCase):
    def setUp(self):
        levels: list[LevelConfig] = [
            {
                "ordering": "series",
                "values": [2],
                "impute": {"method": "zero"},
                "missing_indicator": True,
            },
            {},
        ]
        records = [
            {"children": [{"time": [0.0, 1.0], "values": [[1.0, NAN], [2.0, 3.0]]}]},
            {"children": [{"time": [0.0], "values": [[NAN, 4.0]]}]},
        ]
        self.dataset = _dataset(vitals=(levels, records))
        self.dims: dict[str, list[LevelEmbedding]] = {"vitals": [{"values": 6}, {}]}
        self.batch = _batch(self.dataset, "vitals")
        self.values, self.mask = self.batch[0], self.batch[1]

    def embed(self, mode: Any):
        model = SequenceEmbeddingModel(self.dataset, embedding_dim=self.dims, mask=mode)
        return model, model({"vitals": self.batch})["vitals"][0]["embedding"]

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            unknown: Any = "gated"  # not supported (yet)
            SequenceEmbeddingModel(self.dataset, embedding_dim=self.dims, mask=unknown)

    def test_ignore(self):
        model, out = self.embed("ignore")
        torch.testing.assert_close(
            out, _layer(model, "vitals", 0, "value_proj")(self.values)
        )

    def test_concat(self):
        model, out = self.embed("concat")
        self.assertEqual(_layer(model, "vitals", 0, "value_proj").in_features, 4)
        x = torch.cat([self.values, self.mask.float()], dim=1)
        torch.testing.assert_close(out, _layer(model, "vitals", 0, "value_proj")(x))


class TestPretrain(unittest.TestCase):
    def setUp(self):
        self.dataset = _dataset(labs=(LABS, LAB_RECORDS))
        handle, self.path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(handle, "w") as f:
            f.write("Glucose 1 2 3 4\nunused 9 9 9 9\n")
        self.addCleanup(os.remove, self.path)

    def entry(self, **changes: Any) -> Any:
        return {
            "field": "labs",
            "level": 0,
            "namespace": 0,
            "path": self.path,
            **changes,
        }

    def test_loads_and_freezes(self):
        model = SequenceEmbeddingModel(
            self.dataset, embedding_dim=LAB_DIMS, pretrain=[self.entry(freeze=True)]
        )
        embedding = _layer(model, "labs", 0, "token_0")
        proc = self.dataset.input_processors["labs"]
        assert isinstance(proc, SequenceProcessorV2)
        index = proc.vocab(0, 0)["Glucose"]
        self.assertEqual(embedding.weight[index].tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertTrue(torch.all(embedding.weight[0] == 0))  # <pad>
        self.assertFalse(embedding.weight.requires_grad)

    def test_rejects_bad_entries(self):
        bad: list[Any] = [
            [{"field": "labs", "level": 0, "path": self.path}],  # no namespace
            [self.entry(field="other")],
            [self.entry(level=1)],  # level 1 has no tokens
            [self.entry(namespace=2)],
            [self.entry(), self.entry()],
        ]
        for pretrain in bad:
            with self.subTest(pretrain=pretrain), self.assertRaises(ValueError):
                SequenceEmbeddingModel(
                    self.dataset, embedding_dim=LAB_DIMS, pretrain=pretrain
                )
        with self.assertRaises(NotImplementedError):
            SequenceEmbeddingModel(
                self.dataset,
                embedding_dim=LAB_DIMS,
                pretrain=[self.entry(normalize="l2")],
            )
        with self.assertRaises(FileNotFoundError):
            SequenceEmbeddingModel(
                self.dataset,
                embedding_dim=LAB_DIMS,
                pretrain=[self.entry(path=self.path + ".missing")],
            )


class TestTraining(unittest.TestCase):
    def test_dataloader_batch_and_gradients(self):
        dataset = _dataset(labs=(LABS, LAB_RECORDS))
        model = SequenceEmbeddingModel(dataset, embedding_dim=LAB_DIMS)
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        out = model(batch)["labs"]
        out[0]["embedding"].sum().backward()
        self.assertIsNotNone(_layer(model, "labs", 0, "value_proj").weight.grad)
        self.assertIsNotNone(_layer(model, "labs", 0, "token_1").weight.grad)


if __name__ == "__main__":
    unittest.main()

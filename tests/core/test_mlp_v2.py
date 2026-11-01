import copy
import tempfile
import unittest
from collections.abc import Sequence, Sized
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import LevelEmbedding, LogisticRegression_V2, MLP_V2
from pyhealth.models.sequence_embedding import _FieldEmbedding
from pyhealth.processors import LevelConfig, SequenceProcessorV2
from pyhealth.trainer import Trainer


class TestUnifiedPooling(unittest.TestCase):
    def setUp(self) -> None:
        self.samples: list[dict[str, Any]] = [
            {
                "patient_id": "p0",
                "events": {
                    "time": [1.0, 3.0],
                    "values": [[10.0], [20.0]],
                    "children": [
                        {"values": [[2.0], [6.0]]},
                        {"values": []},
                    ],
                },
                "static": [[1.0, 2.0], [3.0, 4.0]],
                "y": 0,
            },
            {
                "patient_id": "p1",
                "events": {"time": [], "values": [], "children": []},
                "static": [[5.0, 6.0], [7.0, 8.0]],
                "y": 1,
            },
            {
                "patient_id": "p2",
                "events": {
                    "time": [2.0],
                    "values": [[30.0]],
                    "children": [{"values": [[8.0]]}],
                },
                "static": [[9.0, 10.0], [11.0, 12.0]],
                "y": 0,
            },
        ]
        self.schema: dict[str, Any] = {
            "events": (
                "sequence_v2",
                {
                    "levels": [
                        {"ordering": "set", "values": [1]},
                        {"ordering": "series", "values": [1]},
                    ]
                },
            ),
            "static": "tensor",
        }
        self.dims: dict[str, list[LevelEmbedding]] = {
            "events": [{"values": 1}, {"values": 1}]
        }
        self.dataset = create_sample_dataset(self.samples, self.schema, {"y": "binary"})
        self.batch = next(iter(get_dataloader(self.dataset, batch_size=3)))

    def test_hierarchical_mean_and_static_flattening(self) -> None:
        model = LogisticRegression_V2(self.dataset, self.dims)
        with torch.no_grad():
            for level in range(2):
                projection = cast(
                    nn.Linear,
                    cast(
                        _FieldEmbedding, model.embedding.embedding_layers["events"]
                    ).layer(level, "value_proj"),
                )
                projection.weight.fill_(1)
                cast(torch.Tensor, projection.bias).zero_()
        torch.testing.assert_close(
            model(**self.batch, embed=True)["embed"],
            torch.tensor(
                [
                    [15.0, 2.0, 1.0, 2.0, 3.0, 4.0],
                    [0.0, 0.0, 5.0, 6.0, 7.0, 8.0],
                    [30.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                ]
            ),
        )

    def test_mlp_transforms_every_level_and_statics(self) -> None:
        model = MLP_V2(
            self.dataset, self.dims, hidden_dim=1, num_layers=1, dropout=0
        )
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    module.weight.fill_(1)
                    module.bias.fill_(2)
        # Leaves: [6, 10] -> mean 8; empty children -> 0.
        # Parents: [12 + 8 + 2, 22 + 0 + 2] -> mean 23.
        # An empty sample stays zero despite the MLP biases.
        torch.testing.assert_close(
            model(**self.batch, embed=True)["embed"],
            torch.tensor([[23.0, 12.0], [0.0, 28.0], [46.0, 44.0]]),
        )

    def test_static_only_uses_nonlinearity(self) -> None:
        dataset = create_sample_dataset(
            [
                {"patient_id": "p0", "x": [-2.0, 3.0], "y": 0},
                {"patient_id": "p1", "x": [2.0, -3.0], "y": 1},
            ],
            {"x": "tensor"},
            {"y": "binary"},
        )
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        model = MLP_V2(dataset, {}, hidden_dim=2, num_layers=2, dropout=0)
        with torch.no_grad():
            for module in cast(nn.Sequential, model.static_encoders["x"]):
                if isinstance(module, nn.Linear):
                    module.weight.copy_(torch.eye(2))
                    module.bias.zero_()
        torch.testing.assert_close(
            model(**batch, embed=True)["embed"], torch.tensor([[0.0, 3.0], [2.0, 0.0]])
        )
        torch.testing.assert_close(
            LogisticRegression_V2(dataset, {})(**batch, embed=True)["embed"],
            batch["x"].float(),
        )

    def test_order_invariance(self) -> None:
        samples = copy.deepcopy(self.samples)
        samples[0]["events"]["children"][0]["values"].reverse()
        for role in ("time", "values", "children"):
            samples[0]["events"][role].reverse()
        # An untimed sequence permits testing permutation independently of sorting.
        schema = copy.deepcopy(self.schema)
        schema["events"][1]["levels"][1]["ordering"] = "sequence"
        for sample in samples:
            del sample["events"]["time"]
        dataset = create_sample_dataset(samples, schema, {"y": "binary"})
        original = copy.deepcopy(self.samples)
        for sample in original:
            del sample["events"]["time"]
        original = create_sample_dataset(original, schema, {"y": "binary"})
        for cls in (MLP_V2, LogisticRegression_V2):
            model = cls(original, self.dims).eval()
            torch.testing.assert_close(
                model(**next(iter(get_dataloader(original, batch_size=3))))["logit"],
                model(**next(iter(get_dataloader(dataset, batch_size=3))))["logit"],
            )

    def test_per_field_settings_and_gradients(self) -> None:
        model = MLP_V2(
            self.dataset,
            self.dims,
            hidden_dim={"events": 3, "static": 5},
            num_layers={"events": 1},
            dropout={"static": 0},
        )
        self.assertEqual(model.fc.in_features, 8)
        self.assertEqual(len(cast(nn.Sequential, model.static_encoders["static"])), 4)
        model(**self.batch)["loss"].backward()
        self.assertIsNotNone(
            cast(nn.Sequential, model.static_encoders["static"])[0].weight.grad
        )
        self.assertIsNotNone(
            cast(nn.Sequential, cast(nn.ModuleList, model.encoders["events"])[0])[
                0
            ].weight.grad
        )

    def test_constructor_validation(self) -> None:
        for kwargs in cast(
            list[dict[str, Any]],
            [
                {"hidden_dim": 0},
                {"num_layers": 0},
                {"dropout": 1},
                {"hidden_dim": {"missing": 2}},
                {"num_layers": True},
            ],
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MLP_V2(self.dataset, self.dims, **kwargs)
        for cls in (MLP_V2, LogisticRegression_V2):
            with self.assertRaises(ValueError):
                cls(self.dataset, self.dims, mask=cast(Any, "invalid"))
            with self.assertRaises(ValueError):
                cls(self.dataset, {})
            dataset = create_sample_dataset(
                [{"patient_id": str(i), "x": ["a"], "y": i} for i in range(2)],
                {"x": "sequence"},
                {"y": "binary"},
            )
            with self.assertRaises(TypeError):
                cls(dataset, {})
            two_labels = create_sample_dataset(
                [{**sample, "z": 1 - sample["y"]} for sample in self.samples],
                self.schema,
                {"y": "binary", "z": "binary"},
            )
            with self.assertRaisesRegex(ValueError, "exactly one label"):
                cls(two_labels, self.dims)
            no_inputs = create_sample_dataset(
                [{"patient_id": str(i), "y": i} for i in range(2)], {}, {"y": "binary"}
            )
            with self.assertRaisesRegex(ValueError, "At least one input field"):
                cls(no_inputs, {})
            times_only = create_sample_dataset(
                [
                    {"patient_id": str(i), "t": {"time": [0.0, 1.0]}, "y": i}
                    for i in range(2)
                ],
                {"t": ("sequence_v2", {"levels": [{"ordering": "series"}]})},
                {"y": "binary"},
            )
            with self.assertRaisesRegex(ValueError, "nothing to encode"):
                cls(times_only, {"t": [{}]})

    def test_per_step_means_boundaries_and_empty_samples(self) -> None:
        samples = copy.deepcopy(self.samples)
        for sample in samples:
            sample["y"] = {"time": [0.0, 1.0, 2.0, 3.0, 5.0], "values": [0, 1, 0, 1, 0]}
        dataset = create_sample_dataset(
            samples, self.schema, {"y": "timeseries_binary"}
        )
        batch = next(iter(get_dataloader(dataset, batch_size=3)))
        for cls in (MLP_V2, LogisticRegression_V2):
            model = cls(dataset, self.dims).eval()
            embed = model(**batch, embed=True)["embed"]
            # Predict at each individual parent time using single-record samples
            # to construct a reference independent of prefix implementation.
            reference = copy.deepcopy(self.samples)
            reference[0]["events"]["time"] = [1.0]
            reference[0]["events"]["values"] = [[10.0]]
            reference[0]["events"]["children"] = [{"values": [[2.0], [6.0]]}]
            reference[1]["events"] = {
                "time": [3.0],
                "values": [[20.0]],
                "children": [{"values": []}],
            }
            reference = create_sample_dataset(reference, self.schema, {"y": "binary"})
            reference_model = cls(reference, self.dims).eval()
            reference_model.load_state_dict(model.state_dict())
            records = reference_model(
                **next(iter(get_dataloader(reference, batch_size=3))), embed=True
            )["embed"]
            width = model.field_dims["events"]
            torch.testing.assert_close(embed[0, :width], torch.zeros(width))
            torch.testing.assert_close(
                embed[1:3, :width], records[0, :width].expand(2, -1)
            )
            torch.testing.assert_close(
                embed[3:5, :width], records[:2, :width].mean(0).expand(2, -1)
            )
            torch.testing.assert_close(embed[5:10, :width], torch.zeros(5, width))
            torch.testing.assert_close(embed[10:12, :width], torch.zeros(2, width))
            torch.testing.assert_close(
                embed[12:15, :width], records[2, :width].expand(3, -1)
            )
            torch.testing.assert_close(
                embed[:5, width:], records[0, width:].expand(5, -1)
            )
            changed = copy.deepcopy(batch)
            # Changing a future parent cannot affect the first three predictions.
            changed["events"] = tuple(t.clone() for t in batch["events"])
            schema = cast(
                SequenceProcessorV2, dataset.input_processors["events"]
            ).batch_schema()
            changed["events"][schema.index((1, "values"))][1] += 1000
            torch.testing.assert_close(
                model(**changed)["logit"][:3], model(**batch)["logit"][:3]
            )

    def test_resampled_and_all_empty_prefixes(self) -> None:
        for empty in (False, True):
            samples = [
                {
                    "patient_id": str(i),
                    "x": {
                        "time": [] if empty else [0.0, 1.0, 2.0],
                        "values": [] if empty else [[2.0], [4.0], [9.0]],
                    },
                    "y": {"time": [0.0, 1.0, 2.0, 4.0], "values": [1.0, 2.0, 3.0, 4.0]},
                }
                for i in range(2)
            ]
            levels: list[LevelConfig] = [{"ordering": "series", "values": [1]}]
            if not empty:
                levels[0]["resample"] = {"step": 1.0, "method": "last"}
            dataset = create_sample_dataset(
                samples,
                {"x": ("sequence_v2", {"levels": levels})},
                {"y": "timeseries_regression"},
            )
            batch = next(iter(get_dataloader(dataset, batch_size=2)))
            for cls in (MLP_V2, LogisticRegression_V2):
                model = cls(dataset, {"x": [{"values": 1}]}).eval()
                out = model(**batch, embed=True)
                if empty:
                    torch.testing.assert_close(
                        out["embed"], torch.zeros_like(out["embed"])
                    )
                else:
                    x = cast(nn.ModuleList, model.encoders["x"])[0](
                        model.embedding({"x": batch["x"]})["x"][0]["embedding"]
                    )
                    torch.testing.assert_close(out["embed"][0], x[0])
                    torch.testing.assert_close(out["embed"][1], x[:2].mean(0))
                    torch.testing.assert_close(
                        out["embed"][2:4], x[:3].mean(0).expand(2, -1)
                    )
                self.assertTrue(torch.isfinite(out["loss"]))
                out["loss"].backward()

    def test_three_levels_mixed_namespaces_and_spatial_values(self) -> None:
        dataset = create_sample_dataset(
            [
                {
                    "patient_id": str(i),
                    "x": {
                        "tokens": ["stay"],
                        "children": [
                            {
                                "children": [
                                    {
                                        "tokens": [("a", "unit"), ("b", "unit")],
                                        "values": [
                                            [[1.0, 2.0], [3.0, 4.0]],
                                            [[5.0, 6.0], [7.0, 8.0]],
                                        ],
                                    },
                                    {"tokens": [], "values": []},
                                ]
                            }
                        ],
                    },
                    "y": i,
                }
                for i in range(2)
            ],
            {
                "x": (
                    "sequence_v2",
                    {
                        "levels": [
                            {
                                "ordering": "set",
                                "tokens": 2,
                                "values": [2, 2],
                                "spatial": [False, True],
                            },
                            {"ordering": "sequence"},
                            {"ordering": "set", "tokens": 1},
                        ]
                    },
                )
            },
            {"y": "binary"},
        )
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        dims: dict[str, list[LevelEmbedding]] = {
            "x": [{"tokens": (2, 3), "values": 4}, {}, {"tokens": (2,)}]
        }
        for cls in (MLP_V2, LogisticRegression_V2):
            model = cls(dataset, dims).eval()
            embedded = model.embedding({"x": batch["x"]})["x"]
            self.assertEqual(embedded[0]["embedding"].shape, (4, 9))
            leaves = cast(nn.ModuleList, model.encoders["x"])[0](
                embedded[0]["embedding"]
            )
            middle = cast(nn.ModuleList, model.encoders["x"])[1](
                torch.stack(
                    [
                        leaves[:2].mean(0),
                        torch.zeros(leaves.shape[1]),
                        leaves[2:].mean(0),
                        torch.zeros(leaves.shape[1]),
                    ]
                )
            )
            expected = cast(nn.ModuleList, model.encoders["x"])[2](
                torch.cat(
                    [embedded[2]["embedding"], middle.reshape(2, 2, -1).mean(1)], dim=1
                )
            )
            out = model(**batch, embed=True)
            torch.testing.assert_close(out["embed"], expected)
            out["loss"].backward()
            self.assertIsNotNone(
                cast(_FieldEmbedding, model.embedding.embedding_layers["x"])
                .layer(0, "token_1")
                .weight.grad
            )

    def test_sample_label_modes(self) -> None:
        for mode, labels, width in (
            ("binary", [0, 1, 0], 1),
            ("multiclass", ["a", "b", "c"], 3),
            ("multilabel", [["a"], ["b"], ["a", "b"]], 2),
            ("regression", [1.0, 2.0, 3.0], 1),
        ):
            samples = copy.deepcopy(self.samples)
            for sample, label in zip(samples, labels):
                sample["y"] = label
            dataset = create_sample_dataset(samples, self.schema, {"y": mode})
            batch = next(iter(get_dataloader(dataset, batch_size=3)))
            for cls in (MLP_V2, LogisticRegression_V2):
                with self.subTest(mode=mode, model=cls.__name__):
                    out = cls(dataset, self.dims)(**batch)
                    self.assertEqual(out["logit"].shape, (3, width))
                    self.assertTrue(torch.isfinite(out["loss"]))
                    out["loss"].backward()

    def test_per_step_static_only_and_trainer(self) -> None:
        for mode, metric in (
            ("timeseries_binary", "roc_auc"),
            ("timeseries_regression", "mae"),
        ):
            samples = copy.deepcopy(self.samples)
            for sample in samples:
                sample["y"] = {"time": [0.0, 1.0, 3.0], "values": [0.0, 1.0, 0.0]}
            dataset = create_sample_dataset(samples, self.schema, {"y": mode})
            loader = get_dataloader(dataset, batch_size=2)
            for cls in (MLP_V2, LogisticRegression_V2):
                trainer = Trainer(
                    model=cls(dataset, self.dims),
                    metrics=[metric],
                    device="cpu",
                    enable_logging=False,
                )
                trainer.train(train_dataloader=loader, epochs=1)
                self.assertEqual(
                    [
                        len(y)
                        for y in cast(
                            Sequence[Sized],
                            cast(list[object], trainer.inference(loader))[0],
                        )
                    ],
                    [3, 3, 3],
                )
                self.assertIn(metric, trainer.evaluate(loader))
            dataset = create_sample_dataset(samples, {"static": "tensor"}, {"y": mode})
            batch = next(iter(get_dataloader(dataset, batch_size=3)))
            for cls in (MLP_V2, LogisticRegression_V2):
                model = cls(dataset, {}).eval()
                out = model(**batch, embed=True)
                torch.testing.assert_close(
                    out["embed"],
                    model.static_encoders["static"](
                        batch["static"].flatten(1)
                    ).repeat_interleave(3, dim=0),
                )

    def test_per_step_requires_timed_top_level(self) -> None:
        samples = copy.deepcopy(self.samples)
        for sample in samples:
            del sample["events"]["time"]
            sample["y"] = {"time": [0.0], "values": [1.0]}
        for ordering in ("set", "sequence"):
            schema = copy.deepcopy(self.schema)
            schema["events"][1]["levels"][1]["ordering"] = ordering
            dataset = create_sample_dataset(samples, schema, {"y": "timeseries_binary"})
            for cls in (MLP_V2, LogisticRegression_V2):
                with self.assertRaisesRegex(ValueError, "series"):
                    cls(dataset, self.dims)

    def test_missing_indicators_and_pretrained_tokens(self) -> None:
        dataset = create_sample_dataset(
            [
                {
                    "patient_id": str(i),
                    "x": {"tokens": ["a", "b"], "values": [[float("nan")], [float(i)]]},
                    "y": i,
                }
                for i in range(2)
            ],
            {
                "x": (
                    "sequence_v2",
                    {
                        "levels": [
                            {
                                "ordering": "set",
                                "tokens": 1,
                                "values": [1],
                                "impute": {"method": "zero"},
                                "missing_indicator": True,
                            }
                        ]
                    },
                )
            },
            {"y": "binary"},
        )
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "tokens.txt").write_text("a 1 2\nb 3 4\n", encoding="utf-8")
            for cls in (MLP_V2, LogisticRegression_V2):
                model = cls(
                    dataset,
                    {"x": [{"tokens": (2,), "values": 1}]},
                    mask="concat",
                    pretrain=[
                        {
                            "field": "x",
                            "level": 0,
                            "namespace": 0,
                            "path": str(Path(directory, "tokens.txt")),
                            "freeze": True,
                        }
                    ],
                )
                self.assertEqual(
                    cast(_FieldEmbedding, model.embedding.embedding_layers["x"])
                    .layer(0, "value_proj")
                    .in_features,
                    2,
                )
                torch.testing.assert_close(
                    model.embedding({"x": batch["x"]})["x"][0]["embedding"][:, :2],
                    torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]]),
                )
                self.assertFalse(
                    cast(_FieldEmbedding, model.embedding.embedding_layers["x"])
                    .layer(0, "token_0")
                    .weight.requires_grad
                )
                out = model(**batch)
                self.assertTrue(torch.isfinite(out["loss"]))
                out["loss"].backward()

    def test_multiple_fields_sparse_labels_and_device(self) -> None:
        samples = copy.deepcopy(self.samples)
        for i, sample in enumerate(samples):
            sample["other"] = {"time": [0.0, 2.0], "values": [[1.0], [5.0]]}
            sample["y"] = {
                "time": [0.0, 2.0, 4.0][: i + 1],
                "values": [0.0, 1.0, 0.0][: i + 1],
            }
        schema = copy.deepcopy(self.schema)
        schema["other"] = (
            "sequence_v2",
            {"levels": [{"ordering": "series", "values": [1]}]},
        )
        dataset = create_sample_dataset(samples, schema, {"y": "timeseries_binary"})
        batch = next(iter(get_dataloader(dataset, batch_size=3)))
        dims: dict[str, list[LevelEmbedding]] = {**self.dims, "other": [{"values": 1}]}
        for cls in (MLP_V2, LogisticRegression_V2):
            model = cls(dataset, dims).eval()
            out = model(**batch, embed=True)
            torch.testing.assert_close(
                out["loss"],
                nn.functional.binary_cross_entropy_with_logits(
                    out["logit"], batch["y"][0]
                ),
            )
            self.assertEqual(out["logit"].shape, (6, 1))
            width = model.field_dims["events"]
            x = cast(nn.ModuleList, model.encoders["other"])[0](
                model.embedding({"other": batch["other"]})["other"][0]["embedding"]
            )
            torch.testing.assert_close(
                out["embed"][0, width : width + model.field_dims["other"]], x[0]
            )
            torch.testing.assert_close(
                out["embed"][-1, width : width + model.field_dims["other"]],
                x[-2:].mean(0),
            )
            if torch.cuda.is_available():
                model.cuda()
                cuda_out = model(**batch, embed=True)
                torch.testing.assert_close(
                    cuda_out["embed"].cpu(), out["embed"], rtol=1e-4, atol=1e-5
                )
                torch.testing.assert_close(
                    cuda_out["logit"].cpu(), out["logit"], rtol=1e-4, atol=1e-5
                )
                cuda_out["loss"].backward()


if __name__ == "__main__":
    unittest.main()

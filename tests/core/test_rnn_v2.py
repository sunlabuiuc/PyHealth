import unittest
from typing import Any, cast

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import LevelEmbedding, RNN_V2
from pyhealth.models.rnn import _gaps, _pool
from pyhealth.processors import LevelConfig
from pyhealth.trainer import Trainer

# visits (series, timed) -> codes (set)
VISITS: list[LevelConfig] = [
    {"ordering": "set", "tokens": 1},
    {"ordering": "series", "tokens": 1},
]
VISIT_DIMS: dict[str, list[LevelEmbedding]] = {
    "visits": [{"tokens": (4,)}, {"tokens": (3,)}]
}


def _visits(*visits: tuple[float, str, list[str]]) -> dict:
    return {
        "time": [v[0] for v in visits],
        "tokens": [v[1] for v in visits],
        "children": [{"tokens": v[2]} for v in visits],
    }


VISIT_RECORDS = [
    _visits(
        (0.0, "er", ["A", "B"]), (1.5, "op", []), (3.0, "er", ["C"])
    ),  # an empty visit
    _visits((0.0, "op", ["B"])),
    {"time": [], "tokens": [], "children": []},  # no visits at all
    _visits((0.5, "er", ["A", "C", "D"]), (2.0, "op", ["D"])),
]


def _dataset(output: str = "binary", **fields: Any):
    schema = {
        name: ("sequence_v2", {"levels": levels}) if levels is not None else "tensor"
        for name, (levels, _) in fields.items()
    }
    n = len(next(iter(fields.values()))[1])
    samples = [
        {
            "patient_id": f"p{i}",
            **{name: records[i] for name, (_, records) in fields.items()},
            "y": i % 2,
        }
        for i in range(n)
    ]
    return create_sample_dataset(samples, schema, {"y": output})


def _model(dataset, **kwargs: Any) -> RNN_V2:
    model = RNN_V2(dataset, dropout=0.0, **kwargs)
    model.eval()
    return model


def _batch(dataset) -> dict[str, Any]:
    return next(iter(get_dataloader(dataset, batch_size=len(dataset))))


def _encode(model: RNN_V2, batch: dict[str, Any], field: str):
    embedded = model.embedding({field: batch[field]})[field]
    return embedded, cast(Any, model.encoders[field])(embedded)


class TestConfiguration(unittest.TestCase):
    def setUp(self):
        self.dataset = _dataset(visits=(VISITS, VISIT_RECORDS))

    def test_rejects_bad_arguments(self):
        bad: list[dict[str, Any]] = [
            {"hidden_dim": 0},
            {"rnn_type": "Transformer"},
            {"num_layers": {"visits": 0}},
            {"dropout": 1.5},
            {"hidden_dim": {"other": 8}},  # not a field
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RNN_V2(self.dataset, embedding_dim=VISIT_DIMS, **kwargs)

    def test_needs_exactly_one_label(self):
        samples = [
            {"patient_id": f"p{i}", "visits": VISIT_RECORDS[i], "y": i % 2, "z": i % 2}
            for i in range(2)
        ]
        dataset = create_sample_dataset(
            samples,
            {"visits": ("sequence_v2", {"levels": VISITS})},
            {"y": "binary", "z": "binary"},
        )
        with self.assertRaisesRegex(ValueError, "exactly one label"):
            RNN_V2(dataset, embedding_dim=VISIT_DIMS)

    def test_rejects_a_level_with_nothing_to_encode(self):
        times: list[LevelConfig] = [{"ordering": "series"}]  # time only
        records = [{"time": [0.0, 1.0]}, {"time": [0.5]}]
        dataset = _dataset(x=(times, records))
        with self.assertRaisesRegex(ValueError, "nothing to encode"):
            RNN_V2(dataset, embedding_dim={"x": [{}]})
        # The time since the previous record is something to encode.
        model = _model(dataset, embedding_dim={"x": [{}]}, time_feature=True)
        self.assertEqual(model(**_batch(dataset))["logit"].shape, (2, 1))

    def test_rejects_v1_sequences_sets_and_tokens(self):
        for spec, value in (("sequence", ["A", "B"]), ("multi_hot", ["A"])):
            samples = [{"patient_id": f"p{i}", "x": value, "y": i} for i in range(2)]
            dataset = create_sample_dataset(samples, {"x": spec}, {"y": "binary"})
            with self.subTest(spec=spec), self.assertRaises(TypeError):
                RNN_V2(dataset, embedding_dim={})


class TestBottomUp(unittest.TestCase):
    def test_matches_a_loop_reference(self):
        dataset = _dataset(visits=(VISITS, VISIT_RECORDS))
        model = _model(dataset, embedding_dim=VISIT_DIMS, hidden_dim=5)
        batch = _batch(dataset)
        out, (summary, states, time) = _encode(model, batch, "visits")
        encoder = cast(Any, model.encoders["visits"])
        codes, visits = out[0], out[1]
        expected = []
        for b in range(len(dataset)):
            lo, hi = int(visits["offsets"][b]), int(visits["offsets"][b + 1])
            if lo == hi:
                expected.append(torch.zeros(5))
                continue
            rows = []
            for v in range(lo, hi):
                c0, c1 = int(codes["offsets"][v]), int(codes["offsets"][v + 1])
                pooled = (
                    codes["embedding"][c0:c1].mean(0) if c1 > c0 else torch.zeros(4)
                )
                rows.append(torch.cat([visits["embedding"][v], pooled]))
            _, last = encoder.encoders[1](torch.stack(rows).unsqueeze(0))
            expected.append(last[0])
        torch.testing.assert_close(summary, torch.stack(expected))
        self.assertTrue(torch.all(summary[2] == 0))  # no visits
        self.assertEqual(states.shape, (6, 5))  # one state per visit
        self.assertEqual(time.tolist(), [0.0, 1.5, 3.0, 0.0, 0.5, 2.0])

    def test_set_pooling(self):
        x = torch.tensor([[1.0, -1.0], [3.0, 5.0], [2.0, 2.0]])
        offsets = torch.tensor([0, 2, 2, 3])  # the middle group is empty
        self.assertEqual(
            _pool(x, offsets).tolist(), [[2.0, 2.0], [0.0, 0.0], [2.0, 2.0]]
        )

    def test_time_feature(self):
        time = torch.tensor([0.0, 1.5, 3.0, 0.0, 0.5, 2.0])
        offsets = torch.tensor([0, 3, 4, 4, 6])
        self.assertEqual(_gaps(time, offsets).tolist(), [0.0, 1.5, 1.5, 0.0, 0.0, 1.5])
        dataset = _dataset(visits=(VISITS, VISIT_RECORDS))
        plain = _model(dataset, embedding_dim=VISIT_DIMS, hidden_dim=5)
        timed = _model(
            dataset, embedding_dim=VISIT_DIMS, hidden_dim=5, time_feature=True
        )
        width = lambda m: cast(Any, m.encoders["visits"]).encoders[1].input_size
        self.assertEqual(width(timed), width(plain) + 1)
        with torch.no_grad():
            self.assertEqual(timed(**_batch(dataset))["logit"].shape, (4, 1))

    def test_batch_without_any_records(self):
        empty = {"time": [], "tokens": [], "children": []}
        dataset = _dataset(visits=(VISITS, [VISIT_RECORDS[0], empty, empty]))
        model = _model(dataset, embedding_dim=VISIT_DIMS, hidden_dim=5)
        batch = next(iter(get_dataloader(dataset.subset([1, 2]), batch_size=2)))
        _, (summary, _, _) = _encode(model, batch, "visits")
        self.assertEqual(summary.tolist(), [[0.0] * 5] * 2)


    def test_lstm(self):
        dataset = _dataset(visits=(VISITS, VISIT_RECORDS))
        model = _model(
            dataset,
            embedding_dim=VISIT_DIMS,
            hidden_dim=5,
            rnn_type="LSTM",
            num_layers=2,
        )
        result = model(**_batch(dataset))
        self.assertEqual(result["logit"].shape, (4, 1))


class TestStatics(unittest.TestCase):
    def setUp(self):
        ages = [[60.0, 1.0], [70.0, 0.0], [50.0, 1.0], [80.0, 0.0]]
        self.dataset = _dataset(visits=(VISITS, VISIT_RECORDS), age=(None, ages))

    def test_used_directly_or_projected(self):
        direct = _model(self.dataset, embedding_dim=VISIT_DIMS, hidden_dim=5)
        self.assertEqual(direct.fc.in_features, 5 + 2)
        projected = _model(
            self.dataset, embedding_dim=VISIT_DIMS, hidden_dim=5, static_dim={"age": 3}
        )
        self.assertEqual(projected.fc.in_features, 5 + 3)
        result = projected(**_batch(self.dataset), embed=True)
        self.assertEqual(result["embed"].shape, (4, 8))

    def test_per_field_settings(self):
        dataset = _dataset(
            visits=(VISITS, VISIT_RECORDS), stays=(VISITS, VISIT_RECORDS)
        )
        dims = {**VISIT_DIMS, "stays": VISIT_DIMS["visits"]}
        model = _model(
            dataset,
            embedding_dim=dims,
            hidden_dim={"visits": 4, "stays": 6},
        )
        self.assertEqual(model.fc.in_features, 4 + 6)
        self.assertEqual(model(**_batch(dataset))["logit"].shape, (4, 1))


class TestPerStep(unittest.TestCase):
    def labels(self, times: list[float]) -> dict[str, Any]:
        return {"time": times, "values": [i % 2 for i in range(len(times))]}

    def dataset(self, levels, records, times: tuple[float, ...] = (0.0, 1.0, 2.0)):
        samples = [
            {
                "patient_id": f"p{i}",
                "x": records[i],
                "age": [float(i)],
                "y": self.labels(list(times)),
            }
            for i in range(len(records))
        ]
        return create_sample_dataset(
            samples,
            {"x": ("sequence_v2", {"levels": levels}), "age": "tensor"},
            {"y": "timeseries_binary"},
        )

    def test_top_level_must_be_a_series(self):
        for top in ("set", "sequence"):
            levels: list[LevelConfig] = [
                {"ordering": "set", "tokens": 1},
                {"ordering": top},
            ]
            dataset = self.dataset(levels, [{"children": [{"tokens": ["A"]}]}] * 2)
            with self.subTest(top=top), self.assertRaisesRegex(ValueError, "Wrap"):
                RNN_V2(dataset, embedding_dim={"x": [{"tokens": (4,)}, {}]})

    def test_no_bidirectional(self):
        records = [_visits((0.0, "er", ["A"]))] * 2
        dataset = self.dataset(VISITS, records)
        with self.assertRaises(ValueError):
            RNN_V2(
                dataset, embedding_dim={"x": VISIT_DIMS["visits"]}, bidirectional=True
            )

    def test_each_label_uses_the_records_at_or_before_it(self):
        records = [
            # Label 0 sees nothing; label 1 sees visit 0; label 2 sees visit 1.
            _visits((0.5, "er", ["A"]), (2.0, "op", ["B"])),
            # Label 0 sees nothing; labels 1 and 2 see the visit.
            _visits((1.0, "er", ["C"])),
        ]
        dataset = self.dataset(VISITS, records)
        model = _model(
            dataset,
            embedding_dim={"x": VISIT_DIMS["visits"]},
            hidden_dim=5,
        )
        batch = _batch(dataset)
        result = model(**batch, embed=True)
        self.assertEqual(result["logit"].shape, (6, 1))  # 3 labels each
        _, (_, states, _) = _encode(model, batch, "x")
        embed = result["embed"][:, :5]
        self.assertTrue(torch.all(embed[0] == 0))
        torch.testing.assert_close(embed[1], states[0])
        torch.testing.assert_close(embed[2], states[1])
        self.assertTrue(torch.all(embed[3] == 0))
        torch.testing.assert_close(embed[4], states[2])
        torch.testing.assert_close(embed[5], states[2])
        # The static age goes with every label record of its sample.
        self.assertEqual(result["embed"][:, 5].tolist(), [0.0] * 3 + [1.0] * 3)

    def test_labels_between_records(self):
        records = [_visits((0.0, "er", ["A"]), (2.0, "op", ["B"]))] * 2
        dataset = self.dataset(VISITS, records, times=(1.5, 2.5))
        model = _model(dataset, embedding_dim={"x": VISIT_DIMS["visits"]}, hidden_dim=5)
        batch = _batch(dataset)
        embed = model(**batch, embed=True)["embed"][:, :5]
        _, (_, states, _) = _encode(model, batch, "x")
        torch.testing.assert_close(embed[0], states[0])  # 1.5: the visit at 0
        torch.testing.assert_close(embed[1], states[1])  # 2.5: the visit at 2

    def test_batch_without_any_records(self):
        empty = {"time": [], "tokens": [], "children": []}
        records = [_visits((0.0, "er", ["A"])), empty, empty]
        dataset = self.dataset(VISITS, records)
        model = _model(dataset, embedding_dim={"x": VISIT_DIMS["visits"]}, hidden_dim=5)
        batch = next(iter(get_dataloader(dataset.subset([1, 2]), batch_size=2)))
        embed = model(**batch, embed=True)["embed"]
        self.assertEqual(embed.shape, (6, 6))  # 3 labels each, 5 + age
        self.assertTrue(torch.all(embed[:, :5] == 0))

    def test_resampled_top_level(self):
        levels: list[LevelConfig] = [
            {
                "ordering": "series",
                "values": [1],
                "resample": {"step": 1.0, "method": "last"},
            }
        ]
        records = [{"time": [0.0, 1.0, 2.0], "values": [[1.0], [2.0], [3.0]]}] * 2
        dataset = self.dataset(levels, records)
        model = _model(dataset, embedding_dim={"x": [{"values": 2}]}, hidden_dim=5)
        batch = _batch(dataset)
        result = model(**batch, embed=True)
        _, (_, states, _) = _encode(model, batch, "x")
        # Record i is at time i, so the label at time s sees record s.
        torch.testing.assert_close(result["embed"][:3, :5], states[:3])


class TestTraining(unittest.TestCase):
    def test_trains(self):
        dataset = _dataset(visits=(VISITS, VISIT_RECORDS))
        model = RNN_V2(
            dataset, embedding_dim=VISIT_DIMS, hidden_dim=8
        )
        loader = get_dataloader(dataset, batch_size=2, shuffle=True, seed=0)
        trainer = Trainer(model=model, enable_logging=False)
        trainer.train(train_dataloader=loader, epochs=2)
        result = model(**_batch(dataset))
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and torch.any(g != 0) for g in grads))


if __name__ == "__main__":
    unittest.main()

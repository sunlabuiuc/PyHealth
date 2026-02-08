import unittest

from pyhealth.datasets.sample_dataset import InMemorySampleDataset
from pyhealth.datasets.splitter import sample_balanced


def _make_dataset(num_pos: int, num_neg: int) -> InMemorySampleDataset:
    samples = []
    for i in range(num_pos):
        samples.append({
            "patient_id": f"p{i}",
            "record_id": f"p{i}",
            "label": 1,
        })
    for j in range(num_neg):
        idx = num_pos + j
        samples.append({
            "patient_id": f"n{idx}",
            "record_id": f"n{idx}",
            "label": 0,
        })

    return InMemorySampleDataset(
        samples=samples,
        input_schema={},
        output_schema={"label": "binary"},
    )


def _count_labels(dataset):
    pos = 0
    neg = 0
    for i in range(len(dataset)):
        label = dataset[i]["label"]
        label_val = int(label.item()) if hasattr(label, "item") else int(label)
        if label_val == 1:
            pos += 1
        else:
            neg += 1
    return pos, neg


class TestSamplePatientBalanced(unittest.TestCase):
    def test_respects_ratio_without_cap(self):
        # 3 positives, 5 negatives, ratio=0.5 -> desired_neg=2 (min with available)
        dataset = _make_dataset(num_pos=3, num_neg=5)
        balanced = sample_balanced(dataset, ratio=0.5, subsample=1.0, seed=42)

        pos, neg = _count_labels(balanced)
        self.assertEqual(pos, 3)
        self.assertEqual(neg, 2)
        self.assertEqual(len(balanced), 5)

    def test_respects_cap_and_scales_both_sides(self):
        # 3 positives, 10 negatives, ratio=3 -> desired_neg=9, total=12.
        # cap = floor(13 * 0.5) = 6 -> proportional scale keeps pos=1, neg=3.
        dataset = _make_dataset(num_pos=3, num_neg=10)
        balanced = sample_balanced(dataset, ratio=3.0, subsample=0.5, seed=99)

        pos, neg = _count_labels(balanced)
        self.assertEqual(pos, 1)
        self.assertEqual(neg, 3)
        self.assertEqual(len(balanced), 4)


if __name__ == "__main__":
    unittest.main()

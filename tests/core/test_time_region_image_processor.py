import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.processors import TimeRegionImageProcessor, get_processor


class TestTimeRegionImageProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.dark = str(Path(cls.folder.name) / "dark.png")
        cls.split = str(Path(cls.folder.name) / "split.png")
        Image.new("L", (20, 10), 0).save(cls.dark)
        # The left half is white, the right half black.
        pixels = np.zeros((10, 20), dtype=np.uint8)
        pixels[:, :10] = 255
        Image.fromarray(pixels).save(cls.split)
        cls.processor = TimeRegionImageProcessor(num_regions=2, image_size=4)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_registered(self):
        self.assertIs(get_processor("time_region_image"), TimeRegionImageProcessor)

    def test_crops_each_box(self):
        left, right = [0, 0, 10, 10], [10, 0, 20, 10]
        regions, hours, tag = self.processor.process(
            ([self.split], [2.0], [[left, right]])
        )
        self.assertEqual(regions.shape, (1, 2, 3, 4, 4))
        self.assertEqual(regions[0, 0].min(), 1.0)
        self.assertEqual(regions[0, 1].max(), 0.0)
        self.assertEqual(hours.tolist(), [2.0])
        self.assertEqual(tag, "image")

    def test_temporal_interface_matches_time_image_processor(self):
        from pyhealth.processors.base_processor import ModalityType

        self.assertEqual(self.processor.modality(), ModalityType.IMAGE)
        boxes = [[0, 0, 10, 10], [10, 0, 20, 10]]
        value = ([self.split, self.dark], [5.0, 2.0], [boxes, boxes])
        regions, hours, _ = self.processor.process(value)
        out = self.processor.process_temporal(value)
        self.assertEqual(set(out), {"value", "time"})
        self.assertTrue(torch.equal(out["value"], regions))
        self.assertEqual(out["time"].tolist(), hours.tolist())
        self.assertEqual(out["time"].tolist(), [2.0, 5.0])

    def test_boxes_follow_their_image_when_sorted_by_time(self):
        white, black = [0, 0, 10, 10], [10, 0, 20, 10]
        regions, hours, _ = self.processor.process(
            (
                [self.split, self.split],
                [5.0, 1.0],
                [[black, black], [white, white]],
            )
        )
        self.assertEqual(hours.tolist(), [1.0, 5.0])
        self.assertEqual(regions[0].min(), 1.0)
        self.assertEqual(regions[1].max(), 0.0)

    def test_box_without_area_gives_zeros(self):
        regions, _, _ = self.processor.process(
            ([self.split], [0.0], [[[0, 0, 0, 0], [5, 5, 5, 9]]])
        )
        self.assertEqual(regions.abs().sum(), 0)

    def test_float_boxes_are_rounded(self):
        regions, _, _ = self.processor.process(
            ([self.split], [0.0], [[[0.4, 0.0, 9.6, 10.0], [0.0, 0.0, 1.0, 1.0]]])
        )
        self.assertEqual(regions[0, 0].min(), 1.0)

    def test_rejects_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, "match"):
            self.processor.process(([self.dark], [0.0, 1.0], [[]]))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.processor.process(([], [], []))
        with self.assertRaisesRegex(ValueError, "2 boxes"):
            self.processor.process(([self.dark], [0.0], [[[0, 0, 1, 1]]]))
        with self.assertRaises(FileNotFoundError):
            self.processor.process(
                (["/missing.png"], [0.0], [[[0, 0, 1, 1], [0, 0, 1, 1]]])
            )

    def test_size_and_grayscale(self):
        self.assertEqual(self.processor.size(), 3)
        gray = TimeRegionImageProcessor(num_regions=1, image_size=4, mode="L")
        self.assertEqual(gray.size(), 1)
        regions, _, _ = gray.process(([self.dark], [0.0], [[[0, 0, 4, 4]]]))
        self.assertEqual(regions.shape, (1, 1, 1, 4, 4))

    def test_batch_is_a_list_of_tuples(self):
        box = [[0, 0, 10, 10]]
        samples = [
            {"patient_id": f"p{i}", "cxr": ([self.split], [1.0], [box]), "y": i % 2}
            for i in range(2)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={
                "cxr": ("time_region_image", {"num_regions": 1, "image_size": 4})
            },
            output_schema={"y": "binary"},
            in_memory=True,
        )
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        self.assertIsInstance(batch["cxr"], list)
        self.assertIsInstance(batch["cxr"][0][0], torch.Tensor)
        self.assertEqual(batch["cxr"][0][0].shape, (1, 1, 3, 4, 4))


if __name__ == "__main__":
    unittest.main()

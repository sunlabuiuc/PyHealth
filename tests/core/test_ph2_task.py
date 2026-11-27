# Contributor: Aayan Verma (aayanv2)
# Contributor: Bryson Lee (brysonl3)
# Contributor: Xin Zhang (xinz13)
# Paper: A Study of Artifacts on Melanoma Classification under Diffusion-Based
# Perturbations (Jin and Ghassemi, CHIL 2025)
# https://proceedings.mlr.press/v287/jin25b.html
"""Tests for PH2ArtifactClassification and its view/artifact helpers.

The view and artifact tests use tiny synthetic arrays; the task tests build a
4-image PH2-style archive in a temporary folder. No real data is used.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from pyhealth.datasets import PH2Dataset, SampleDataset
from pyhealth.tasks import PH2ArtifactClassification
from pyhealth.tasks.dermoscopy_artifact_views import (
    ARTIFACTS,
    VIEWS,
    add_artifact,
    bbox_mask,
    gaussian_blur,
    lesion_bbox,
    make_view,
)
from tests.core.test_ph2_dataset import make_ph2_root

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _toy() -> tuple[np.ndarray, np.ndarray]:
    """A 32x32 random image with a 8x8 lesion in the middle."""
    rng = np.random.default_rng(0)
    image = rng.uniform(0.1, 0.9, (32, 32, 3)).astype(np.float32)
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:20, 12:20] = True
    return image, mask


class TestViews(unittest.TestCase):
    """Tests for the 12 dataset modes."""

    def test_there_are_twelve_views(self) -> None:
        """The paper's twelve dataset modes are all defined and distinct."""
        self.assertEqual(len(VIEWS), 12)
        self.assertEqual(len(set(VIEWS)), 12)

    def test_whole_lesion_background(self) -> None:
        """Lesion-only and background-only views split the image exactly."""
        image, mask = _toy()
        np.testing.assert_array_equal(make_view(image, mask, "whole"), image)
        lesion = make_view(image, mask, "lesion")
        background = make_view(image, mask, "background")
        np.testing.assert_array_equal(lesion[mask], image[mask])
        self.assertEqual(float(lesion[~mask].sum()), 0.0)
        np.testing.assert_array_equal(background[~mask], image[~mask])
        self.assertEqual(float(background[mask].sum()), 0.0)
        np.testing.assert_allclose(lesion + background, image)

    def test_bbox_hides_the_lesion_outline(self) -> None:
        """The bbox view blacks out the lesion and keeps pixels outside the box."""
        image, mask = _toy()
        view = make_view(image, mask, "bbox")
        self.assertEqual(float(view[mask].sum()), 0.0)
        np.testing.assert_array_equal(view[0], image[0])  # outside the box

    def test_bbox_coverage_grows(self) -> None:
        """bbox70 and bbox90 grow to the target coverage, and no further."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[45:55, 45:55] = True
        tight = bbox_mask(mask).mean()
        mid = bbox_mask(mask, 0.7).mean()
        big = bbox_mask(mask, 0.9).mean()
        self.assertAlmostEqual(tight, 0.01)
        self.assertTrue(tight < mid < big < 1.0)
        self.assertGreaterEqual(mid, 0.7)
        self.assertGreaterEqual(big, 0.9)
        self.assertLess(mid, 0.72)  # the smallest box that reaches 70%
        self.assertLess(big, 0.93)

    def test_bbox_reaches_coverage_for_a_corner_lesion(self) -> None:
        """A tiny lesion in a corner still grows to the coverage target."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[99, 99] = True
        for target in (0.7, 0.9):
            box = bbox_mask(mask, target)
            self.assertGreaterEqual(box.mean(), target)
            self.assertTrue(box[99, 99])  # the lesion stays inside the box

    def test_bbox_of_empty_mask_does_not_crash(self) -> None:
        """An empty mask still gives a valid box."""
        empty = np.zeros((16, 16), dtype=bool)
        self.assertEqual(lesion_bbox(empty), (8, 8, 8, 8))
        self.assertGreaterEqual(bbox_mask(empty, 0.7).mean(), 0.7)

    def test_low_pass_smooths_and_high_pass_keeps_detail(self) -> None:
        """Low-pass smooths; low-pass plus high-pass rebuilds the image."""
        image, mask = _toy()
        low = make_view(image, mask, "whole_low")
        high = make_view(image, mask, "whole_high")
        self.assertLess(low.std(), image.std())
        self.assertEqual(high.min() >= 0.0 and high.max() <= 1.0, True)
        # low + (high - 0.5) rebuilds the image except where high was clipped
        rebuilt = low + (high - 0.5)
        inside = (high > 0) & (high < 1)
        np.testing.assert_allclose(rebuilt[inside], image[inside], atol=1e-5)

    def test_filters_apply_to_the_region_view(self) -> None:
        """Filters are applied after masking the region."""
        image, mask = _toy()
        expected = gaussian_blur(make_view(image, mask, "lesion"))
        np.testing.assert_allclose(make_view(image, mask, "lesion_low"), expected)

    def test_gaussian_blur_preserves_constant_image(self) -> None:
        """Blurring a flat image changes nothing."""
        flat = np.full((9, 9, 3), 0.25, dtype=np.float32)
        np.testing.assert_allclose(gaussian_blur(flat), flat, atol=1e-6)

    def test_unknown_view_raises(self) -> None:
        """An unknown view name raises ValueError."""
        image, mask = _toy()
        with self.assertRaises(ValueError):
            make_view(image, mask, "sepia")


class TestArtifacts(unittest.TestCase):
    """Tests for the hand-drawn artifacts."""

    def test_every_artifact_changes_pixels_but_not_the_lesion(self) -> None:
        """Each artifact alters the image, stays in [0, 1] and spares the lesion."""
        image, mask = _toy()
        for artifact in ARTIFACTS:
            with self.subTest(artifact=artifact):
                out = add_artifact(image, mask, artifact, seed=1)
                self.assertEqual(out.shape, image.shape)
                self.assertTrue(out.min() >= 0.0 and out.max() <= 1.0)
                self.assertFalse(np.array_equal(out, image))
                np.testing.assert_array_equal(out[mask], image[mask])

    def test_deterministic_given_seed(self) -> None:
        """The same seed gives the same artifact; another seed differs."""
        image, mask = _toy()
        a = add_artifact(image, mask, "gel_bubble", seed=3)
        b = add_artifact(image, mask, "gel_bubble", seed=3)
        c = add_artifact(image, mask, "gel_bubble", seed=4)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_unknown_artifact_raises(self) -> None:
        """An unknown artifact name raises ValueError."""
        image, mask = _toy()
        with self.assertRaises(ValueError):
            add_artifact(image, mask, "hair")


class TestPH2ArtifactClassification(unittest.TestCase):
    """Tests for the PH2ArtifactClassification task."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build one synthetic archive and dataset for all task tests."""
        cls.tmp = Path(tempfile.mkdtemp())
        make_ph2_root(cls.tmp / "PH2Dataset", with_missing_mask=False)
        cls.dataset = PH2Dataset(
            root=str(cls.tmp / "PH2Dataset"), cache_dir=str(cls.tmp / "cache")
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the temporary folder."""
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _samples(self, **kwargs: Any) -> SampleDataset:
        """Run the task on the synthetic dataset with a tiny image size."""
        return self.dataset.set_task(PH2ArtifactClassification(image_size=16, **kwargs))

    def test_samples_and_labels(self) -> None:
        """One sample per image, labelled 1 for melanoma, with the right shape."""
        samples = self._samples()
        self.assertEqual(len(samples), 4)
        by_id = {samples[i]["image_id"]: samples[i] for i in range(len(samples))}
        self.assertEqual(
            {k: float(v["malignant"]) for k, v in by_id.items()},
            {"IMD001": 0.0, "IMD002": 0.0, "IMD003": 1.0, "IMD004": 1.0},
        )
        self.assertEqual(tuple(by_id["IMD001"]["image"].shape), (3, 16, 16))
        samples.close()

    def test_view_changes_the_input(self) -> None:
        """Selecting another view changes the model input."""
        whole = self._samples(view="whole")
        lesion = self._samples(view="lesion")
        a, b = whole[0]["image"], lesion[0]["image"]
        self.assertFalse(np.allclose(a.numpy(), b.numpy()))
        whole.close()
        lesion.close()

    def test_artifact_changes_the_input_not_the_label(self) -> None:
        """Adding an artifact changes the input but not the label."""
        plain = self._samples()
        ruled = self._samples(artifact="ruler")
        self.assertFalse(
            np.allclose(plain[0]["image"].numpy(), ruled[0]["image"].numpy())
        )
        self.assertEqual(float(plain[0]["malignant"]), float(ruled[0]["malignant"]))
        plain.close()
        ruled.close()

    def test_render_matches_manual_pipeline(self) -> None:
        """render equals resize, view and ImageNet normalisation done by hand."""
        task = PH2ArtifactClassification(view="lesion_low", image_size=16)
        (event,) = self.dataset.get_patient("IMD001").get_events(event_type="ph2")
        out = task.render(event.image_path, event.mask_path, "IMD001")
        image, mask = task._load(event.image_path, event.mask_path)
        expected = (
            make_view(image, mask, "lesion_low") - _IMAGENET_MEAN
        ) / _IMAGENET_STD
        np.testing.assert_allclose(out, expected.transpose(2, 0, 1), atol=1e-6)

    def test_invalid_arguments_raise(self) -> None:
        """Bad view, artifact or image_size raises ValueError."""
        with self.assertRaises(ValueError):
            PH2ArtifactClassification(view="sepia")
        with self.assertRaises(ValueError):
            PH2ArtifactClassification(artifact="hair")
        with self.assertRaises(ValueError):
            PH2ArtifactClassification(image_size=2)

"""Tests for RetinaUNetCTDataset.

All tests use tiny synthetic volumes (<=3 patients, 4x16x16 each) so the
full suite finishes in milliseconds and requires neither MIMIC nor any
demo dataset.
"""

import numpy as np
import pytest

from pyhealth.datasets.retina_unet_ct_dataset import RetinaUNetCTDataset
from pyhealth.tasks.retina_unet_detection import RetinaUNetDetectionTask


def _make_synthetic_volume(seed: int = 0):
    """Return a (4, 16, 16) volume and a matching instance mask.

    Slice 0: empty. Slice 1: one lesion. Slice 2: two lesions. Slice 3: empty.
    """
    rng = np.random.default_rng(seed)
    volume = rng.standard_normal((4, 16, 16)).astype(np.float32)
    mask = np.zeros((4, 16, 16), dtype=np.int32)

    mask[1, 2:6, 2:6] = 1

    mask[2, 4:8, 4:8] = 1
    mask[2, 10:14, 10:14] = 2

    return volume, mask


def _make_two_patients():
    vol_a, mask_a = _make_synthetic_volume(seed=1)
    vol_b, mask_b = _make_synthetic_volume(seed=2)
    return {"p1": vol_a, "p2": vol_b}, {"p1": mask_a, "p2": mask_b}


def test_in_memory_construction_and_length():
    volumes, masks = _make_two_patients()
    ds = RetinaUNetCTDataset(volumes=volumes, masks=masks)

    assert len(ds) == 2 * 4  # 2 patients * 4 slices


def test_sample_shapes_match_task_contract():
    volumes, masks = _make_two_patients()
    ds = RetinaUNetCTDataset(volumes=volumes, masks=masks)

    sample = ds[0]
    assert set(sample.keys()) == {"patient_id", "slice_idx", "image", "mask"}
    assert sample["image"].shape == (16, 16, 1)
    assert sample["mask"].shape == (16, 16)
    assert sample["patient_id"] in {"p1", "p2"}


def test_skip_empty_slices_drops_background_only_slices():
    volumes, masks = _make_two_patients()
    ds = RetinaUNetCTDataset(
        volumes=volumes, masks=masks, skip_empty_slices=True
    )

    # slices 0 and 3 are empty in the synthetic fixture, so only 2 of
    # 4 slices survive per patient.
    assert len(ds) == 2 * 2
    for i in range(len(ds)):
        assert np.any(ds[i]["mask"])


def test_hu_window_clips_and_scales_to_unit_range():
    volumes = {
        "p1": np.array(
            [[[-2000.0, 0.0], [100.0, 500.0]]], dtype=np.float32
        )  # (1, 2, 2)
    }
    masks = {"p1": np.zeros((1, 2, 2), dtype=np.int32)}
    ds = RetinaUNetCTDataset(
        volumes=volumes, masks=masks, hu_window=(-1000.0, 400.0)
    )

    image = ds[0]["image"].squeeze(-1)
    assert image.min() >= 0.0 and image.max() <= 1.0
    # -2000 clips to -1000, maps to 0.0
    assert image[0, 0] == pytest.approx(0.0)
    # 500 clips to 400, maps to 1.0
    assert image[1, 1] == pytest.approx(1.0)


def test_get_patient_returns_all_slices_in_order():
    volumes, masks = _make_two_patients()
    ds = RetinaUNetCTDataset(volumes=volumes, masks=masks)

    slices = ds.get_patient("p1")
    assert len(slices) == 4
    assert [s["slice_idx"] for s in slices] == [0, 1, 2, 3]
    assert all(s["patient_id"] == "p1" for s in slices)


def test_stats_reports_patient_and_slice_counts():
    volumes, masks = _make_two_patients()
    ds = RetinaUNetCTDataset(
        volumes=volumes, masks=masks, skip_empty_slices=True
    )
    assert ds.stats() == {"num_patients": 2, "num_slices": 4}


def test_set_task_produces_detection_samples():
    volumes, masks = _make_two_patients()
    ds = RetinaUNetCTDataset(
        volumes=volumes, masks=masks, skip_empty_slices=True
    )

    processed = ds.set_task(RetinaUNetDetectionTask())

    assert len(processed) == len(ds)
    for sample in processed:
        assert "boxes" in sample and "labels" in sample
        assert sample["boxes"].shape[1] == 4
        assert sample["boxes"].shape[0] == sample["labels"].shape[0]
    # the 2-lesion slice (slice 2) should contribute a sample with 2 boxes
    assert any(s["boxes"].shape[0] == 2 for s in processed)


def test_default_task_is_retina_unet_detection():
    volumes, masks = _make_two_patients()
    ds = RetinaUNetCTDataset(volumes=volumes, masks=masks)

    assert isinstance(ds.default_task, RetinaUNetDetectionTask)


def test_mismatched_volumes_and_masks_raise():
    volumes = {"p1": np.zeros((2, 4, 4), dtype=np.float32)}
    masks = {"p1": np.zeros((3, 4, 4), dtype=np.int32)}  # wrong shape
    with pytest.raises(ValueError):
        RetinaUNetCTDataset(volumes=volumes, masks=masks)

    with pytest.raises(ValueError):
        RetinaUNetCTDataset(
            volumes={"p1": np.zeros((1, 2, 2), dtype=np.float32)},
            masks={"p2": np.zeros((1, 2, 2), dtype=np.int32)},  # wrong key
        )


def test_disk_mode_loads_from_npy_files(tmp_path):
    volumes, masks = _make_two_patients()

    for pid in volumes:
        pdir = tmp_path / pid
        pdir.mkdir()
        np.save(pdir / "volume.npy", volumes[pid])
        np.save(pdir / "mask.npy", masks[pid])

    # also leave a non-patient directory to make sure it is ignored
    (tmp_path / "notes").mkdir()

    ds = RetinaUNetCTDataset(root=str(tmp_path))
    assert len(ds) == 2 * 4
    assert set(ds.iter_patients()) == {"p1", "p2"}

    # spot-check: slicing from disk matches slicing from memory
    ds_mem = RetinaUNetCTDataset(volumes=volumes, masks=masks)
    for i in range(len(ds)):
        np.testing.assert_array_equal(ds[i]["mask"], ds_mem[i]["mask"])


def test_missing_root_raises():
    with pytest.raises(FileNotFoundError):
        RetinaUNetCTDataset(root="/nonexistent/path/xyz")

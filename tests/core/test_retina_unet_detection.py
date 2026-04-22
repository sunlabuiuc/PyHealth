import numpy as np
from pyhealth.tasks.retina_unet_detection import RetinaUNetDetectionTask


def test_process_sample():
    task = RetinaUNetDetectionTask()

    image = np.random.rand(64, 64, 3)
    mask = np.zeros((64, 64))
    mask[10:20, 15:25] = 1

    sample = {"image": image, "mask": mask}

    output = task.process_sample(sample)

    assert "boxes" in output
    assert "labels" in output
    assert output["boxes"].shape[1] == 4


def test_empty_mask():
    task = RetinaUNetDetectionTask()

    image = np.random.rand(64, 64, 3)
    mask = np.zeros((64, 64))

    sample = {"image": image, "mask": mask}

    output = task.process_sample(sample)

    assert output["boxes"].shape[0] == 0

def test_single_object_bbox_correct():
    task = RetinaUNetDetectionTask()

    image = np.random.rand(64, 64, 3)
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[10:20, 15:25] = 1

    sample = {"image": image, "mask": mask}
    output = task.process_sample(sample)

    box = output["boxes"][0]

    assert np.array_equal(box, np.array([15, 10, 24, 19]))

def test_multiple_objects():
    task = RetinaUNetDetectionTask()

    image = np.random.rand(64, 64, 3)
    mask = np.zeros((64, 64), dtype=np.int32)

    mask[5:10, 5:10] = 1
    mask[20:30, 25:35] = 2

    sample = {"image": image, "mask": mask}
    output = task.process_sample(sample)

    assert output["boxes"].shape[0] == 2

def test_small_object_filtered():
    task = RetinaUNetDetectionTask(min_area=50)

    image = np.random.rand(64, 64, 3)
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[10:12, 10:12] = 1  # too small

    sample = {"image": image, "mask": mask}
    output = task.process_sample(sample)

    assert output["boxes"].shape[0] == 0

def test_mask_to_bbox():
    task = RetinaUNetDetectionTask()

    binary_mask = np.zeros((32, 32), dtype=bool)
    binary_mask[8:16, 10:20] = True

    bbox = task._mask_to_bbox(binary_mask)

    assert bbox == [10, 8, 19, 15]

def test_collate_fn():
    task = RetinaUNetDetectionTask()

    samples = []
    for _ in range(3):
        image = np.random.rand(64, 64, 3)
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[10:20, 10:20] = 1
        samples.append({"image": image, "mask": mask})

    processed = [task.process_sample(s) for s in samples]
    batch = task.collate_fn(processed)

    assert len(batch["images"]) == 3
    assert len(batch["boxes"]) == 3


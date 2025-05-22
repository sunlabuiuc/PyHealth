from pyhealth.tasks.binary_ECG_classification import BinaryECGClassification
from pyhealth.datasets import PTBXL
import torch
def test_binary_ECG_classification():
    dataset = PTBXL(
            root="/home/sbgray2/ptbxl_data/" \
            "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
            config_path="pyhealth/datasets/configs/ptbxl.yaml",
            use_high_res=False,
        )
    print("building dataset...")
    dataset.build_from_events(limit=5)

    task = BinaryECGClassification(dataset)
    task.preprocess()
    print("Testing preprocess method...")
    # Confirm sample count
    assert len(task.samples) == len(dataset)

    # Check fields and tensor types
    sample = task[0]
    assert "signal" in sample and "label" in sample

    assert isinstance(sample["signal"], torch.Tensor)
    assert isinstance(sample["label"], torch.Tensor)

    assert sample["signal"].shape == (12, 2496)
    assert sample["label"].ndim == 0  # scalar tensor for class label

    print("✅ Preprocess correctly transforms samples to torch tensors.")
    print("Testing __call__ method...")
    all_outputs = []
    for i in range(len(dataset)):
        patient = dataset[i]
        outputs = task(patient)
        assert isinstance(outputs, list)
        assert "signal" in outputs[0]
        assert "label" in outputs[0]
        assert outputs[0]["signal"].shape == (12, 2496)
        all_outputs.extend(outputs)

    assert len(all_outputs) == len(dataset)
    print(" __call__ method works correctly.")
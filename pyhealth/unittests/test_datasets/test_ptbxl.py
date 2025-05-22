from pyhealth.datasets import PTBXL

def test_ptbxl_dataset():
    dataset = PTBXL(
        root="/home/sbgray2/ptbxl_data/" \
        "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
        config_path="pyhealth/datasets/configs/ptbxl.yaml",
        use_high_res=False
    )
    dataset.build_from_events(limit=5)

    assert len(dataset) > 0
    sample = dataset[0]
    assert "signal" in sample
    assert "label" in sample
    assert sample["signal"].shape[0] == 12  # channels
    assert sample["signal"].shape[1] == 2496  # length

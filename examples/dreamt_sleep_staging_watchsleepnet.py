from pyhealth.datasets import DREAMTDataset
from pyhealth.datasets.sample_dataset import SampleDataset
from pyhealth.models import WatchSleepNet
from pyhealth.tasks import SleepStagingDREAMT

PAPER = {
    "num_classes": 3,
    "input_length": 750,
    "num_input_channels": 1,
    "res_channels": (32, 64, 128, 256),
    "res_strides": (5, 5, 5, 2),
    "kernel_size": 5,
    "hidden_dim": 256,
    "tcn_layers": 3,
    "lstm_layers": 4,
    "num_heads": 16,
    "dropout": 0.2,
    "use_tcn": True,
    "use_attention": True,
    "bidirectional": True,
    "focal_gamma": 0.0,
    "ignore_index": -100,
}

if __name__ == "__main__":
    root_path = ""
    dataset = DREAMTDataset(root_path).set_task(SleepStagingDREAMT())
    assert isinstance(dataset, SampleDataset)

    model = WatchSleepNet(dataset)

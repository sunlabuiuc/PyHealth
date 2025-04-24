from pyhealth.datasets import TUSZDataset
from pyhealth.processors import SignalProcessor



# root = "c:/dlh/v2.0.3/edf/dev"
# dataset = TUSZDataset(
#     root=root,
# )

processor = SignalProcessor(sampling_rate=200)

segments = processor.process(
    (14.2442,57.9846, "c:/dlh/v2.0.3/edf/dev/aaaaaajy/s001_2003/02_tcp_le/aaaaaajy_s001_t000.edf")
)

print(segments.shape)  # Should be (num_segments, 32, 8000)

#dataset.stats()
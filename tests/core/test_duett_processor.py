import numpy as np
import torch
from pyhealth.processors.duett_processor import DuETTTimeSeriesProcessor

def test_duett_processor():
    processor = DuETTTimeSeriesProcessor()
    
    raw_1 = np.array([[2.0, 10.0],[4.0, 0.0],[0.0, 0.0]])
    counts_1 = np.array([[1, 1],[1, 0], [0, 0]])
    raw_2 = np.array([[6.0, 20.0],[0.0, 30.0],[0.0, 0.0]])
    counts_2 = np.array([[1, 1], [0, 1],[0, 0]])
    
    samples = [{"x_ts": (raw_1, counts_1)}, {"x_ts": (raw_2, counts_2)}, {"x_ts": None}]
    processor.fit(samples, "x_ts")
    
    assert processor.size() == 4
    
    assert processor.is_token() is False
    assert processor.schema() == ("value",)
    assert processor.dim() == (2,)
    assert processor.spatial() == (True, False)
    
    out_tensor = processor.process((raw_1, counts_1))
    assert out_tensor.shape == (3, 4)
    assert torch.isclose(out_tensor[0, 0], torch.tensor(-1.2247), atol=1e-4)
    
    try:
        DuETTTimeSeriesProcessor().fit([{"x_ts": None}], "x_ts")
        assert False
    except ValueError:
        pass
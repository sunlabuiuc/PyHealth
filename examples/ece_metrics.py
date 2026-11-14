"""Minimal Expected Calibration Error (ECE) examples."""

import numpy as np

from pyhealth.metrics.calibration import (
    ece_confidence_binary,
    ece_confidence_multiclass,
)


binary_prob = np.array([0.1, 0.8])
binary_label = np.array([0, 1])
print("Binary ECE:", ece_confidence_binary(binary_prob, binary_label, bins=2))

multiclass_prob = np.array(
    [
        [0.2, 0.2, 0.6],
        [0.2, 0.31, 0.49],
        [0.1, 0.1, 0.8],
    ]
)
multiclass_label = np.array([2, 1, 2])
print(
    "Multiclass ECE:",
    ece_confidence_multiclass(multiclass_prob, multiclass_label, bins=2),
)

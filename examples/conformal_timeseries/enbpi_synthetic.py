"""EnbPI prediction intervals on a synthetic time series.

Builds an AR(1) series whose noise level doubles halfway through the test
period, trains an EnbPI ensemble of small MLPs on the first part, and runs
the online predict, reveal and update loop over the rest. It prints overall
miscoverage and width, and the lowest and highest miscoverage over 100-step
windows, for two feedback batch sizes ``s``.

Expect overall miscoverage a little above the 0.1 target and a rise in
rolling miscoverage after the shift. The residual window still holds ``T``
residuals from before the shift and needs about ``T`` steps to catch up
(Remark 2 of the paper). Adaptive methods such as ACI target this lag.

Paper:
    Xu, Chen, and Yao Xie. "Conformal prediction interval for dynamic
    time-series." ICML 2021.

Run:
    python examples/conformal_timeseries/enbpi_synthetic.py
"""

import numpy as np
import torch

from pyhealth.calib.predictionset import EnbPI
from pyhealth.datasets import create_sample_dataset
from pyhealth.metrics import regression_metrics_fn
from pyhealth.models import MLP

N_TRAIN = 300
N_TEST = 900
LAGS = 3
ALPHA = 0.1


def make_series(seed: int = 0):
    """AR(1) series whose noise standard deviation doubles mid-test."""
    rng = np.random.default_rng(seed)
    n = LAGS + N_TRAIN + N_TEST
    shift_at = LAGS + N_TRAIN + N_TEST // 2
    y = np.zeros(n)
    for t in range(1, n):
        noise_sd = 1.0 if t < shift_at else 2.0
        y[t] = 0.8 * y[t - 1] + rng.normal(scale=noise_sd)
    samples = [
        {"patient_id": "p0", "x": y[t - LAGS : t].tolist(), "y": float(y[t])}
        for t in range(LAGS, n)
    ]
    return create_sample_dataset(samples, {"x": "tensor"}, {"y": "regression"})


def main():
    torch.manual_seed(0)
    series = make_series()
    train = series.subset(range(N_TRAIN))
    test = series.subset(range(N_TRAIN, N_TRAIN + N_TEST))

    for feedback_size in (1, 50):
        enbpi = EnbPI(
            model_fn=lambda: MLP(
                dataset=series,
                feature_keys=["x"],
                label_key="y",
                mode="regression",
                hidden_dim=16,
            ),
            alpha=ALPHA,
            n_bootstrap=20,
            train_kwargs={"epochs": 20},
            feedback_size=feedback_size,
            random_state=0,
        )
        enbpi.calibrate(train)
        out = enbpi.predict_sequential(test)

        scores = regression_metrics_fn(
            out["y_true"],
            out["y_pred"],
            metrics=["miscoverage_pi", "interval_width", "rolling_miscoverage_pi@100"],
            y_lower=out["y_lower"],
            y_upper=out["y_upper"],
        )
        rolling = scores["rolling_miscoverage_pi@100"]
        print(f"s = {feedback_size}")
        print(f"  target miscoverage   {ALPHA:.2f}")
        print(f"  miscoverage          {scores['miscoverage_pi']:.3f}")
        print(f"  mean width           {scores['interval_width']:.3f}")
        print(
            f"  rolling miscoverage  {rolling.min():.3f} min, {rolling.max():.3f} max"
        )


if __name__ == "__main__":
    main()

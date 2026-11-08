pyhealth.models.XGBoostModel
============================

Gradient-boosted trees (XGBoost) on fixed-width tabular and bag-of-codes
features. The model is fit once with ``model.fit(train_loader, val_loader)``,
then evaluated, saved and interpreted through the usual PyHealth APIs.

Install the optional extra first:

.. code-block:: bash

    pip install "pyhealth[xgboost]"

.. code-block:: python

    from pyhealth.datasets import get_dataloader
    from pyhealth.models import XGBoostModel
    from pyhealth.trainer import Trainer

    model = XGBoostModel(
        sample_dataset,
        bag_of_codes=True,           # code sequences -> per-sample code counts
        n_estimators=400, max_depth=4, learning_rate=0.05,
        scale_pos_weight="balanced", # negatives / positives in the training labels
    )
    model.fit(get_dataloader(train_ds, 256), get_dataloader(val_ds, 256))
    trainer = Trainer(model=model)
    trainer.evaluate(get_dataloader(test_ds, 256))
    trainer.save_ckpt("xgb.ckpt")    # the fitted trees round-trip
    model.mean_abs_shap(test_ds)     # global TreeSHAP ranking

Things to know:

- ``Trainer.train`` raises a ``TypeError`` for this model: fit it with
  ``model.fit``.
- **Inputs.** Columns are built from the fields in ``input_schema`` order, which
  matters because column subsampling depends on the order. Padded code
  sequences raise an error unless you pass ``bag_of_codes=True``.
- **Missing values.** ``NaN`` stays missing and XGBoost learns where it goes at
  each split. This gives different trees from median imputation.
- **Imbalance.** ``scale_pos_weight`` inflates probabilities. Recalibrate on a
  patient-grouped calibration split (e.g. Platt scaling) if you need calibrated
  risks; the model does not calibrate silently.
- **Memory.** The training matrix is held in memory as dense float32
  (``n_samples x n_columns x 4`` bytes).
- **macOS.** PyTorch's wheel bundles its own ``libomp.dylib``. XGBoost loads
  Homebrew's ``libomp``. Two OpenMP runtimes in one process can crash
  multithreaded fits with a segfault or ``OMP: Error #179``. Make the two share
  one runtime, for example by replacing ``<site-packages>/torch/lib/libomp.dylib``
  with a symlink to ``$(brew --prefix libomp)/lib/libomp.dylib``, or pass
  ``n_jobs=1``.

Example: ``examples/mortality_prediction/mortality_mimic3_xgboost.py`` compares
XGBoost with ``LogisticRegression`` and ``MLP`` on the same patient split.

.. autoclass:: pyhealth.models.XGBoostModel
    :members: fit, forward, explain, mean_abs_shap, build_feature_matrix, predict_numpy
    :show-inheritance:

.. autoclass:: pyhealth.models.GradientBoostedTreeModel
    :members:
    :show-inheritance:

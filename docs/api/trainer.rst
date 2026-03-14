Trainer
=======

:class:`~pyhealth.trainer.Trainer` handles the PyTorch training loop for you.
Rather than writing your own epoch loop, loss backward pass, optimizer step,
and metric evaluation, you hand the Trainer your model and data loaders and
let it manage the details — including early stopping when validation
performance plateaus and automatic reloading of the best checkpoint at the end.

A Typical Training Run
-----------------------

Here is what a full training setup looks like. The data loaders come from
``get_dataloader()`` in :mod:`pyhealth.datasets`, which knows how to work with
PyHealth's LitData caching format:

.. code-block:: python

    from pyhealth.trainer import Trainer
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader   = get_dataloader(val_ds,   batch_size=32, shuffle=False)
    test_loader  = get_dataloader(test_ds,  batch_size=32, shuffle=False)

    trainer = Trainer(
        model=model,
        metrics=["roc_auc_macro", "pr_auc_macro", "f1_macro"],
        device="cuda",
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        epochs=50,
        monitor="roc_auc_macro",
        monitor_criterion="max",
        patience=10,
    )

    scores = trainer.evaluate(test_loader)
    # {'roc_auc_macro': 0.85, 'pr_auc_macro': 0.79, 'f1_macro': 0.72, 'loss': 0.31}

Setting Up the Trainer
-----------------------

``Trainer(model, metrics=None, device=None, enable_logging=True, output_path=None, exp_name=None)``

- **model** — your instantiated PyHealth model.
- **metrics** — the metric names you want computed at validation and test time
  (e.g. ``["roc_auc_macro", "f1_macro"]``). See :doc:`metrics` for the full
  list of supported strings.
- **device** — ``"cuda"`` or ``"cpu"``; defaults to auto-detecting a GPU.
- **enable_logging** — when enabled, the Trainer creates a timestamped folder
  under ``output_path`` with a ``log.txt`` and model checkpoints.
- **output_path** / **exp_name** — where and how to name the output folder.

Controlling the Training Loop
------------------------------

``trainer.train()`` accepts these key arguments beyond the data loaders:

- **epochs** — the maximum number of training epochs.
- **optimizer_class** / **optimizer_params** — which optimizer to use and how
  to configure it. Defaults to ``Adam`` with a learning rate of ``1e-3``.
- **weight_decay** — L2 regularisation strength. Default ``0.0``.
- **max_grad_norm** — if set, clips gradients to this norm before each update,
  which can help stabilise training on noisy medical data.
- **monitor** / **monitor_criterion** — the metric to watch on the validation
  set (e.g. ``"roc_auc_macro"``) and whether higher is better (``"max"``) or
  lower is better (``"min"``). The Trainer saves a checkpoint whenever this
  metric improves.
- **patience** — how many epochs without improvement to wait before stopping
  early.
- **load_best_model_at_last** — when ``True`` (the default), the Trainer
  restores the best checkpoint at the end of training rather than keeping the
  weights from the final epoch.

Getting the Test Scores
------------------------

``trainer.train()`` prints test scores to the console when a
``test_dataloader`` is provided, but it does not return them as a Python
object. To capture results for downstream use, call ``evaluate()`` separately:

.. code-block:: python

    scores = trainer.evaluate(test_loader)
    # scores is a plain dict, e.g. {'roc_auc_macro': 0.85, 'loss': 0.31}

    import json
    with open("results.json", "w") as f:
        json.dump(scores, f, indent=2)

API Reference
-------------

.. autoclass:: pyhealth.trainer.Trainer
    :members:
    :undoc-members:
    :show-inheritance:

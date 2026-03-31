PyHealth Architecture Overview
==============================

This page describes how all PyHealth components connect, from raw data files
to a trained, evaluated model. Every stage has its own dedicated reference
page — this overview is here to show how they fit together.

Pipeline at a Glance
---------------------

.. code-block:: text

    Raw CSV / Parquet files
            │
            ▼
    config.yaml  (table schemas, patient_id col, timestamp col, attributes)
            │
            ▼
    BaseDataset subclass  ──── loads tables, caches as global_event_df.parquet
            │   .unique_patient_ids → List[str]
            │   .get_patient(id)    → Patient
            │   .iter_patients()    → Iterator[Patient]
            │   .stats()            → prints patient/event counts
            │
            ▼
    BaseTask subclass  (__call__(patient) → List[Dict])
            │   .input_schema  = {"feature": "processor_name", ...}
            │   .output_schema = {"label": "binary" | "multiclass" | ...}
            │
    dataset.set_task(task, num_workers=N)
            │
            ▼
    SampleDataset  ──── len(ds), ds[i], patient_to_index, record_to_index
            │   Backed by LitData streaming files
            │   Processors fitted during set_task, applied at load time
            │
    get_dataloader(dataset, batch_size=32, shuffle=True)
            │
            ▼
    Model(dataset, ...)  ──── BaseModel subclass (RNN, Transformer, MLP, …)
            │   EmbeddingModel routes features via processor.is_token()
            │   forward(**batch) → {"loss", "y_prob", "y_true", "logit"}
            │
            ▼
    Trainer(model, metrics=[...], device=...)
            │   .train(train_dl, val_dl, test_dl, epochs=20, ...)
            │   .evaluate(test_dl) → Dict[metric_name, value]
            │
            ├──▶ Calibration  (pyhealth.calib)
            │       TemperatureScaling / HistogramBinning / KCal / …
            │       LABEL / SCRIB / FavMac / …  (conformal prediction sets)
            │
            └──▶ Interpretability  (pyhealth.interpret)
                    GradientSaliency / IntegratedGradients / DeepLift / SHAP / LIME / …


Stage 1: Raw Data → BaseDataset
---------------------------------

See :doc:`datasets` for the full reference.

PyHealth reads raw CSV or Parquet files using Polars, joins tables according
to a ``config.yaml`` schema, and writes a compact
``global_event_df.parquet`` cache. On subsequent runs with the same
configuration it reads from the cache rather than re-parsing source files.

**Native datasets** (MIMIC-III, MIMIC-IV, eICU, OMOP, and many others) have
built-in schemas — just pass a ``root`` path and a list of ``tables``:

.. code-block:: python

    from pyhealth.datasets import MIMIC3Dataset

    if __name__ == '__main__':
        dataset = MIMIC3Dataset(
            root="/data/mimiciii/1.4",
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir=".cache",
            dev=True,   # cap at 1 000 patients during exploration
        )

**Custom datasets** subclass ``BaseDataset`` and provide a ``config_path``
pointing to your own ``config.yaml``. If files need preprocessing (e.g.
merging multiple CSVs), define ``preprocess_<table_name>(self, df)`` on the
subclass — it receives a narwhals LazyFrame and must return one.

Key init params: ``root``, ``tables``, ``config_path`` (custom only),
``cache_dir``, ``num_workers``, ``dev``.

A UUID derived from ``(root, tables, dataset_name, dev)`` is appended to
``cache_dir``, so different configurations never overwrite each other.


Stage 2: Patient and Event Objects
------------------------------------

See :doc:`data` for the full reference.

Once a dataset is loaded, ``Patient.get_events()`` is the primary query
method:

.. code-block:: python

    events = patient.get_events(
        event_type="diagnoses_icd",              # must match the table name in config.yaml
        start=datetime(2020, 1, 1),
        end=datetime(2020, 6, 1),
        filters=[("icd_code", "==", "250.00")],
    )

``Event`` attributes to keep in mind:

- ``event.timestamp`` — always use this; PyHealth normalises ``charttime``,
  ``admittime``, etc. into a single property.
- ``event.attr_dict`` / ``event["col_name"]`` / ``event.col_name`` — access
  attribute values. All column names are **lowercased** at ingest time.


Stage 3: Task Definition → set_task
--------------------------------------

See :doc:`tasks` for the full reference.

A ``BaseTask`` subclass defines three things:

- ``task_name: str`` — must be assigned (not just annotated).
- ``input_schema`` / ``output_schema`` — dicts mapping sample keys to
  processor string aliases (e.g. ``"sequence"``, ``"binary"``).
- ``__call__(self, patient) → List[Dict]`` — extracts features from one
  ``Patient``; return ``[]`` to skip a patient.

``dataset.set_task(task, num_workers=N)`` iterates all patients, collects
samples, fits processors, and writes LitData ``.ld`` streaming files to disk.
The result is a ``SampleDataset``.

.. important::

   All code calling ``set_task()`` must live inside
   ``if __name__ == '__main__':``. PyHealth uses multiprocessing internally
   and will crash without this guard.


Stage 4: Processors → SampleDataset
--------------------------------------

See :doc:`processors` for the full reference.

When ``set_task()`` runs:

1. ``SampleBuilder.fit(samples)`` — calls ``processor.fit(samples, field)``
   for every schema field.
2. ``SampleBuilder.transform(sample)`` — calls ``processor.process(value)``
   for every field, writing tensors to disk.

The key signal for model routing is ``processor.is_token()``:

- ``True`` → ``nn.Embedding`` (discrete token indices, e.g. medical codes)
- ``False`` → ``nn.Linear`` (continuous values, e.g. time series, images)


Stage 5: Model Initialization and Forward Pass
------------------------------------------------

See :doc:`models` for the full reference.

.. code-block:: python

    from pyhealth.models import RNN

    model = RNN(dataset=sample_dataset, embedding_dim=128, hidden_dim=64)

The model reads ``dataset.input_schema``, ``dataset.output_schema``, and
``dataset.input_processors`` to auto-build embedding layers and the output
head. **Always pass the ``SampleDataset`` (result of ``set_task()``), not
the raw ``BaseDataset``.**

``model(**batch)`` where ``batch`` is a dict from the DataLoader. Must return
``{"loss", "y_prob", "y_true", "logit"}``.

**Choosing a model:**

- Mixed sequential + static features → ``MultimodalRNN``
- Purely sequential codes → ``RNN`` or ``Transformer``
- Static feature vector → ``MLP``
- Time-stamped vitals with irregular intervals → ``StageNet``
- Graph-structured features → ``GNN`` or ``GraphCare`` (see :doc:`graph`)


Stage 6: Training and Evaluation
----------------------------------

See :doc:`trainer` for the full reference.

.. code-block:: python

    from pyhealth.trainer import Trainer
    from pyhealth.datasets import get_dataloader

    train_dl = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_dl   = get_dataloader(val_ds,   batch_size=32, shuffle=False)
    test_dl  = get_dataloader(test_ds,  batch_size=32, shuffle=False)

    trainer = Trainer(model=model, metrics=["roc_auc_macro", "f1_macro"], device="cuda")
    trainer.train(train_dl, val_dl, test_dl, epochs=20,
                  monitor="roc_auc_macro", monitor_criterion="max", patience=5)

    scores = trainer.evaluate(test_dl)
    # → {"roc_auc_macro": 0.85, "loss": 0.3, ...}

Split by patient to avoid data leakage:

.. code-block:: python

    all_ids = list(sample_dataset.patient_to_index.keys())
    # ... split all_ids into train_ids / val_ids / test_ids ...
    train_indices = [i for pid in train_ids for i in sample_dataset.patient_to_index[pid]]
    train_ds = sample_dataset.subset(train_indices)


Common Pitfalls
----------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Mistake
     - Fix
   * - Missing ``if __name__ == '__main__':``
     - Wrap all ``set_task()`` / dataset loading code in this guard
   * - ``event.charttime`` instead of ``event.timestamp``
     - Always use ``event.timestamp``
   * - Task sample key doesn't match ``input_schema``
     - Keys in ``__call__`` return dict must exactly match schema keys
   * - ``dev=True`` during full training
     - Only use ``dev=True`` during exploration; set ``dev=False`` for final runs
   * - Passing ``BaseDataset`` to the model
     - Pass ``SampleDataset`` (result of ``set_task()``) to the model
   * - ``dataset.patients``
     - Does not exist; use ``dataset.unique_patient_ids`` + ``dataset.get_patient(id)``

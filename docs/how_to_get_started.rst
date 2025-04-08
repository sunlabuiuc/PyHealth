.. _how_to_get_started:

=====================
How to Get Started with PyHealth
=====================

Welcome to PyHealth! This guide will help you build machine learning pipelines for healthcare tasks using PyHealth.

Installing PyHealth
===================

You can install PyHealth using `pip`:

.. code-block:: bash

   pip install pyhealth

To install the latest development version from GitHub:

.. code-block:: bash

   git clone https://github.com/sunlabuiuc/PyHealth.git
   cd PyHealth
   pip install -e .

Ensure you have the required dependencies installed before proceeding.

Overview of ML Pipelines
=========================

All healthcare tasks in PyHealth follow a five-stage pipeline:

.. image:: figure/five-stage-pipeline.png
   :alt: Five-stage pipeline

Each stage is modular, allowing customization based on your needs.

Stage 1: Loading Datasets
=========================

`pyhealth.datasets` provides structured datasets independent of tasks. PyHealth supports MIMIC-III, MIMIC-IV, eICU, and more.

Example:

.. code-block:: python

   from pyhealth.datasets import MIMIC3Dataset

   mimic3base = MIMIC3Dataset(
       root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
       tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
       code_mapping={"NDC": "CCSCM"},
   )

Stage 2: Defining Tasks
========================

`pyhealth.tasks` processes patient data into task-specific samples.

Example:

.. code-block:: python

   from pyhealth.tasks import readmission_prediction_mimic3_fn

   mimic3sample = mimic3base.set_task(task_fn=readmission_prediction_mimic3_fn)

To split data and create DataLoaders:

.. code-block:: python

   from pyhealth.datasets import split_by_patient, get_dataloader

   train_ds, val_ds, test_ds = split_by_patient(mimic3sample, [0.8, 0.1, 0.1])
   train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
   val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
   test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)

Stage 3: Building ML Models
===========================

`pyhealth.models` provides various machine learning models.

Example:

.. code-block:: python

   from pyhealth.models import Transformer

   model = Transformer(
       dataset=mimic3sample,
       feature_keys=["conditions", "procedures", "drugs"],
       label_key="label",
       mode="binary",
   )

Stage 4: Training the Model
===========================

`pyhealth.trainer` allows specifying training parameters such as optimizer, epochs, and learning rate.

Example:

.. code-block:: python

   from pyhealth.trainer import Trainer

   trainer = Trainer(model=model)
   trainer.train(
       train_dataloader=train_loader,
       val_dataloader=val_loader,
       epochs=50,
       monitor="pr_auc_samples",
   )

Stage 5: Evaluating Model Performance
=====================================

`pyhealth.metrics` provides evaluation metrics.

Example:

.. code-block:: python

   trainer.evaluate(test_loader)

   from pyhealth.metrics.binary import binary_metrics_fn

   y_true, y_prob, loss = trainer.inference(test_loader)
   binary_metrics_fn(y_true, y_prob, metrics=["pr_auc", "roc_auc"])
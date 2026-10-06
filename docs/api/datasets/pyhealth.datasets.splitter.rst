pyhealth.datasets.splitter
===================================

Several data splitting function for `pyhealth.datasets` module to obtain training / validation / test sets.

Fitting processors on the training split
----------------------------------------

The ``split_by_*`` functions split an already-processed ``SampleDataset``: its
processors (code vocabularies, and any statistics a processor learns) were fitted
on all samples, validation and test patients included. To fit them on the
training patients only, let ``set_task`` do the split:

.. code-block:: python

    from pyhealth.datasets import PatientSplit

    train, val, test = dataset.set_task(
        task, split=PatientSplit(ratios=(0.7, 0.1, 0.2), seed=42)
    )
    train.fit_split   # {'kind': 'patient', 'ratios': [0.7, 0.1, 0.2], 'seed': 42}

- Every processor is fitted on the training patients; validation and test samples
  are processed with those frozen processors, so a code seen only in test maps to
  ``<unk>``, as it would at deployment.
- Patients are sorted by ID before the seeded shuffle, so a ``PatientSplit`` gives
  the same patients on every run and machine. Two ratios give ``(train, test)``.
- Samples are streamed while fitting: the training patients are selected from the
  index of samples per patient that processing already keeps, and each processor
  reads the same sample stream, skipping other patients. Memory does not grow with
  the size of the samples. ``examples/benchmark_perf/benchmark_split_set_task.py``
  measures it at scale on synthetic data.
- The result is cached per split: a rerun with the same ``PatientSplit`` reuses it,
  another seed or ratio builds its own.
- Processors that learn statistics from the data (``learns_statistics = True``)
  log a warning when fitted on all samples, since that leaks validation/test
  information. Vocabulary-only processors do not warn.

The default, ``set_task(task)`` without ``split``, is unchanged: it fits on all
samples and returns one dataset.


.. automodule:: pyhealth.datasets.splitter
    :members:
    :undoc-members:
    :show-inheritance:

   

   
   
   
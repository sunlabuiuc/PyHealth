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
  the same patients on every run and machine. It takes two or more ratios summing
  to 1, training part first: ``(train, test)``, ``(train, val, test)``,
  ``(train, val, cal, test)`` and so on, returned in that order.
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

Writing your own split
----------------------

``split=`` accepts any subclass of ``Split``. It implements two methods:

- ``split_indices(patient_to_index)`` returns one array of sample indices per part,
  training part first. ``patient_to_index`` maps each patient ID to the indices of
  its samples. ``set_task`` fits the processors on part 0 and returns every part as
  given: parts may overlap or leave samples out.
- ``to_dict()`` returns a JSON-serialisable description. It goes into the cache key,
  so it must change whenever the parts would, and is saved as ``fit_split``.

For example, holding out a fixed set of patients:

.. code-block:: python

    import numpy as np
    from pyhealth.datasets import Split

    class HoldoutSplit(Split):
        def __init__(self, test_patients):
            self.test_patients = sorted(test_patients)

        def split_indices(self, patient_to_index):
            test = set(self.test_patients)
            parts = ([], [])
            for pid, indices in patient_to_index.items():
                parts[pid in test].extend(indices)
            return [np.array(sorted(p), dtype=np.int64) for p in parts]

        def to_dict(self):
            return {"kind": "holdout", "test_patients": self.test_patients}

    train, test = dataset.set_task(task, split=HoldoutSplit(["patient-a", "patient-b"]))


.. automodule:: pyhealth.datasets.splitter
    :members:
    :undoc-members:
    :show-inheritance:

   

   
   
   
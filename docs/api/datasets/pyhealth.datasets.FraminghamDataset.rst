pyhealth.datasets.FraminghamDataset
===================================

The Framingham baseline cohort used in the Neural Fine-Gray experiments. The
prepared table contains one row for each of 4,434 participants, 18 baseline
covariates, follow-up duration, and competing event cause. Cardiovascular
disease is cause one; death without a prior cardiovascular event is cause two.

The optional downloader uses a commit-pinned public copy from the paper's
DeepSurvivalMachines dependency. No data are bundled with PyHealth.

.. autoclass:: pyhealth.datasets.FraminghamDataset
   :members:
   :show-inheritance:

.. autofunction:: pyhealth.datasets.framingham.prepare_framingham

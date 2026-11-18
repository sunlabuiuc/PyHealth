Time-to-event survival tasks
============================

The GBSG and SUPPORT tasks produce a dense feature vector and separate
``duration: [time]`` and ``event: [indicator]`` tensors. Event 1 is observed;
event 0 is censored. They work with PyHealth's existing
:class:`~pyhealth.models.SurvivalMDN` and standard Trainer.

GBSG retains seven ordered source covariates. SUPPORT uses the complete-case
cohort, eight numeric covariates and drop-first category encoding, giving 27
features. Missing race has its own category. Missing numeric SUPPORT values
exclude a record; invalid labels and unsupported categories raise errors.

Both tasks retain source time units by default. Feature scaling and time
conversion are explicit constructor options. Fit feature statistics on training
patients and reuse those same values for validation and test patients.

.. autoclass:: pyhealth.tasks.SurvivalTimeToEventGBSG
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.SurvivalTimeToEventSupport2
    :members:
    :show-inheritance:

See :doc:`../../survival_pipeline` for examples and manual acceptance checks.

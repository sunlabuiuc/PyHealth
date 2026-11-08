Frequently Asked Questions
==========================

----


Controlling PyHealth's log output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyHealth prints its progress messages (dataset loading, task processing,
training) to standard output through its own handler on the ``pyhealth``
logger, at level ``INFO``. That logger does not pass records on to the root
logger, so each line prints once even in Jupyter or Colab, or after
``logging.basicConfig``.

- Quieter output: ``logging.getLogger("pyhealth").setLevel(logging.WARNING)``.
- Route PyHealth's messages through your own logging configuration (for
  example to a file or a log collector) instead of standard output:

  .. code-block:: python

     import logging
     import pyhealth

     logging.basicConfig(filename="run.log", level=logging.INFO)
     logging.getLogger("pyhealth").removeHandler(pyhealth.handler)
     logging.getLogger("pyhealth").propagate = True

``Trainer`` also writes its messages to ``log.txt`` in the experiment folder
when ``enable_logging=True``; that is unaffected by these settings.

See ``examples/logging_setup.py``.


Blueprint & Development Plan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The long term goal of PyHealth is to become a comprehensive healthcare AI toolkit that supports
beyond EHR data, but also the images and clinical notes.

This is the central place to track important things to be fixed/added:

- The support of image datasets and clinical notes
- The compatibility and the support of OMOP format datasets
- Model persistence (save, load, and portability)
- The release of a benchmark paper with PyHealth
- Add contact channel with `Gitter <https://gitter.im>`_
- Support additional languages, see `Manage Translations <https://docs.readthedocs.io/en/latest/guides/manage-translations.html>`_

Feel free to open on issue report if needed.
See `Issues <https://github.com/yzhao062/combo/issues>`_.


Inclusion Criteria
^^^^^^^^^^^^^^^^^^

Similarly to Similarly to scikit-learn, We mainly consider well-established algorithms for inclusion.
A rule of thumb is at least two years since publication, 50+ citations, and usefulness.

However, we encourage the author(s) of newly proposed models to share and add your implementation into combo
for boosting ML accessibility and reproducibility.
This exception only applies if you could commit to the maintenance of your model for at least two year period.


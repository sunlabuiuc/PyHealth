PBC competing first events
==========================

.. automodule:: pyhealth.tasks.competing_risks_pbc
    :members:
    :show-inheritance:

The task retains baseline missingness, fixed clinical categories, and baseline
age. Labels are censoring=0, death=1, transplantation=2. Follow-up is in unshifted
years. A transplant ends first-event follow-up; it does not imply that later
death is impossible. Invalid delayed entry and inconsistent endpoints fail
explicitly.

Split patients before fitting
:class:`pyhealth.tasks.CompetingRisksPreprocessor`.

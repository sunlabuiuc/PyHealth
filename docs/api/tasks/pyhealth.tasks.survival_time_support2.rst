SUPPORT2 survival time and event task
========================================

The existing ``Support2Dataset`` loader expects ``support2.csv`` with an ID
column named ``sno``. This task retains both censored and observed outcomes.
Invalid/nonpositive durations and missing numerical predictors exclude rows;
unknown categories and invalid event codes raise. Missing race is retained
as a category. ``onehot22`` and ``benchmark14`` use distinct task/cache names.
Both encodings are fixed, not fitted across a cohort. Scale and impute only
on the eventual training partition. Some measurements are recorded through
day three; this task does not assert all features are admission-time values.

The output schema has separate numerical ``duration`` and ``event`` fields.
It does not turn time-to-event prediction into binary mortality prediction
or use supplied prognostic survival probabilities as targets. The benchmark
14-column order and category codes match the verified pycox SUPPORT route;
the default 22-column one-hot example is a different encoding.

Feature encodings
-----------------

``benchmark14`` uses the following order:

``age, sex, num.co, race, diabetes, dementia, ca, meanbp, hrt, resp, temp,
sod, wblc, crea``.

Its nominal codes are sex: male=0, female=1; race: other=0, white=1,
black=2, hispanic=3, asian=4, missing=5; ca: metastatic=0, no=1, yes=2.
These numeric codes match the processed benchmark but do not make the
categories ordinal biological measurements.

``onehot22`` starts with eleven numerical fields:

``age, num.co, diabetes, dementia, meanbp, hrt, resp, temp, wblc, sod, crea``.

It appends sex indicators (female, male), race indicators (asian, black,
hispanic, other, white, missing), and ca indicators (metastatic, no, yes),
in that order. The task's ``feature_names`` attribute gives the exact layout.
No prognosis, ID, death indicator or follow-up time enters the feature vector.

Usage
-----

.. code-block:: python

   from pyhealth.datasets import Support2Dataset
   from pyhealth.tasks import SurvivalTimeSupport2

   raw = Support2Dataset(root="/path/to/local/data", tables=["support2"])
   task = SurvivalTimeSupport2(encoding="onehot22")
   samples = raw.set_task(task, num_workers=1)

Some UCI files call the ID column ``id``; adapt that column to ``sno`` before
using the existing loader. Apply subject splits before fitting numerical
preprocessing or GRAFT. This task's encodings are fixed and fit no population
statistics. ``event=0`` denotes right-censoring, not permanent event absence.
Exclusions depend on missing clinical predictors and invalid durations, not
on whether an event was observed. A missing race is an explicit category.

Run ``examples/support2_survival_graft.py`` for a fabricated-data loader/task/
model pipeline. Unit tests use at most four fabricated patients and no files
from the real SUPPORT cohort.

Public API
----------

.. autoclass:: pyhealth.tasks.survival_time_support2.SurvivalTimeSupport2
   :members: __call__
   :show-inheritance:

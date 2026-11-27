pyhealth.tasks.PH2ArtifactClassification
========================================

Benign vs. malignant classification of PH2 dermoscopy images, with the model
input altered by one of twelve "views" (whole image, lesion only, background
only, lesion-hiding bounding boxes, high- and low-pass filters) and, optionally,
a hand-drawn artifact (dark corner, gel bubble, ink, patches, ruler). It follows
Jin and Ghassemi, `A Study of Artifacts on Melanoma Classification under
Diffusion-Based Perturbations <https://proceedings.mlr.press/v287/jin25b.html>`_
(CHIL 2025). See ``examples/derm/ph2_artifact_classification_resnet50.py``.

.. autoclass:: pyhealth.tasks.PH2ArtifactClassification
    :members:
    :undoc-members:
    :show-inheritance:

Manual test cases
-----------------

These checks need the real PH2 archive, so they are not part of the automated
tests (which use synthetic data). Each case lists its input and the expected
result. Run the code from a script guarded by ``if __name__ == "__main__":``,
because PyHealth starts worker processes.

.. code-block:: python

    import numpy as np

    from pyhealth.datasets import PH2Dataset
    from pyhealth.tasks import PH2ArtifactClassification
    from pyhealth.tasks.dermoscopy_artifact_views import make_view

    dataset = PH2Dataset(root="/path/to/PH2Dataset")
    (event,) = dataset.get_patient("IMD003").get_events(event_type="ph2")
    whole = PH2ArtifactClassification("whole", image_size=64)
    image, mask = whole._load(event.image_path, event.mask_path)  # float, bool

1. **Default samples.** Input: ``dataset.set_task(PH2ArtifactClassification())``.
   Expected: 200 samples; each ``sample["image"]`` is a ``float32`` tensor of
   shape ``(3, 224, 224)``; 40 samples have ``malignant == 1``.

2. **Lesion view.** Input: ``PH2ArtifactClassification("lesion", image_size=64)``
   rendered for ``IMD003``. Expected: every pixel outside the mask equals the
   normalised black value ``(-2.118, -2.036, -1.804)``, and every pixel inside
   the mask equals the ``whole`` view.

3. **Artifacts spare the lesion.** Input: ``PH2ArtifactClassification("whole",
   "ruler", image_size=64)`` rendered for ``IMD003``. Expected: pixels inside
   the mask are identical to the ``whole`` view, about 6% of pixels differ, and
   the bottom rows (where the ruler is drawn) differ.

4. **Repeatable artifacts.** Input: ``gel_bubble`` rendered twice with
   ``seed=0``, then once with ``seed=1``. Expected: the two ``seed=0`` outputs
   are equal and the ``seed=1`` output differs.

5. **Box coverage.** Input: ``make_view(image, mask, view)`` for ``bbox``,
   ``bbox70`` and ``bbox90``. Expected: the fraction of black pixels is about
   0.17, 0.71 and 0.91; the whole lesion is black in all three.

6. **Bad arguments.** Input: ``view="sepia"``, ``artifact="hair"`` and
   ``image_size=2``, one at a time. Expected: each raises ``ValueError``.

7. **End to end, no real data.** Input:

   .. code-block:: bash

       python examples/derm/ph2_artifact_classification_resnet50.py --quick

   Expected (about one minute on a CPU): the line ``Lesion check: 5 artifacts
   leave the lesion of 5 images unchanged.``, then ``whole original`` with
   AUROC about 1.00, and ``whole ruler`` clearly lower (about 0.7) with a
   ``delta CI95`` whose lower bound is above 0. This shows the test detects
   a planted ruler shortcut.

View and artifact helpers
-------------------------

.. automodule:: pyhealth.tasks.dermoscopy_artifact_views
    :members:
    :undoc-members:

pyhealth.datasets.PH2Dataset
============================

A dataset class for the **PH2 dermoscopy database** — 200 dermoscopic
images of melanocytic lesions labelled as Common Nevus (80), Atypical
Nevus (80), or Melanoma (40).

Two source formats are supported automatically:

* **Mirror format** (recommended for quick start): flat JPEG images and a
  ``PH2_simple_dataset.csv`` from the community mirror
  `vikaschouhan/PH2-dataset <https://github.com/vikaschouhan/PH2-dataset>`_.
  Pass ``download=True`` to fetch this automatically.

* **Original format**: nested BMP images and expert annotations from the
  `official ADDI project release <https://www.fc.up.pt/addi/ph2%20database.html>`_
  (requires registration).  Place the extracted ``PH2_Dataset_images/``
  directory inside *root* and the dataset will use it automatically.

**Data source**

Teresa Mendonça, Pedro M. Ferreira, Jorge Marques, Andre R.S. Marcal,
Jorge Rozeira. "PH² - A dermoscopic image database for research and
benchmarking", 35th International Conference on Engineering in Medicine
and Biology Society, EMBC'13, pp. 5437-5440, IEEE, 2013.

License: free for non-commercial research purposes.  See the
`ADDI project page <https://www.fc.up.pt/addi/ph2%20database.html>`_ for
full terms.

**Quick start**

.. code-block:: python

    from pyhealth.datasets import PH2Dataset
    from pyhealth.tasks import PH2MelanomaClassification

    # Download mirror automatically
    dataset = PH2Dataset(root="~/ph2", download=True)

    # Apply 3-class melanoma classification task
    samples = dataset.set_task(PH2MelanomaClassification())

.. autoclass:: pyhealth.datasets.PH2Dataset
    :members:
    :undoc-members:
    :show-inheritance:

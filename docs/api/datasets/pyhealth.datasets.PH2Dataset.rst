pyhealth.datasets.PH2Dataset
============================

Overview
--------

PH2 dermoscopy dataset: 200 images (80 common nevi, 80 atypical nevi, 40
melanomas) from Hospital Pedro Hispano, Matosinhos, Portugal, each with an
expert binary lesion mask. The masks let
:class:`~pyhealth.tasks.PH2ArtifactClassification` build lesion-only,
background-only and bounding-box views of every image. Each image is its own
PyHealth patient (``IMDxxx``).

Getting the data
----------------

- **Source:** `PH2 database <https://www.fc.up.pt/addi/ph2%20database.html>`_
- **Paper:** Mendonça, T., Ferreira, P. M., Marques, J. S., Marcal, A. R. S.,
  Rozeira, J. PH2 - A dermoscopic image database for research and benchmarking.
  35th Annual International Conference of the IEEE EMBC, 2013.
  https://doi.org/10.1109/EMBC.2013.6610779
- **Access:** released for research and educational use; cite the paper. Get the
  archive from the page above (it may ask for a short form) and unpack it.
  PyHealth cannot download it for you.
- **Annotations:** the clinical diagnosis (0 common nevus, 1 atypical nevus,
  2 melanoma) is read from the ``Clinical Diagnosis`` column of the archive's
  ``PH2_dataset.txt``, or of ``PH2_dataset.csv`` or ``PH2_dataset.xlsx`` (needs
  ``openpyxl``).
- **root layout** (folders are searched recursively):

  .. code-block:: text

      PH2Dataset/                                    <- root
          PH2_dataset.txt    (or .xlsx / .csv)
          PH2 Dataset images/
              IMD002/
                  IMD002_Dermoscopic_Image/IMD002.bmp
                  IMD002_lesion/IMD002_lesion.bmp
              IMD003/ ...

``root`` is only read. The derived metadata CSV goes to ``metadata_dir``.

Manual test cases
-----------------

These checks need the real PH2 archive, so they are not part of the automated
tests (which use synthetic data). Set ``root`` to the folder that holds
``PH2_dataset.txt`` and ``PH2 Dataset images``. Each case lists its input and
the expected result.

.. code-block:: python

    from pathlib import Path

    import pandas as pd
    from PIL import Image

    from pyhealth.datasets import PH2Dataset

    root = "/path/to/PH2Dataset"
    dataset = PH2Dataset(root=root)
    meta = pd.read_csv(dataset.metadata_path)

1. **Load and count.** Input: the full archive.
   ``len(dataset.unique_patient_ids)`` is ``200`` and
   ``meta["diagnosis"].value_counts().to_dict()`` is
   ``{"atypical_nevus": 80, "common_nevus": 80, "melanoma": 40}``.

2. **One image.** Input: patient ``"IMD003"``.

   .. code-block:: python

       (event,) = dataset.get_patient("IMD003").get_events(event_type="ph2")
       print(event.diagnosis, Path(event.image_path).exists(),
             Path(event.mask_path).exists())

   Expected: ``common_nevus True True``.

3. **Image and mask match.** Input: all 200 rows of ``meta``. Expected: for
   every row, ``Image.open(image_path).size == Image.open(mask_path).size``.

4. **Missing annotation file.** Input: ``PH2Dataset(root=<an empty folder>)``.
   Expected: ``FileNotFoundError`` saying that no ``PH2_dataset.txt``, ``.csv``
   or ``.xlsx`` was found.

5. **Partial copy.** Input: a folder with the full ``PH2_dataset.txt`` but only
   the ``IMD002``, ``IMD003`` and ``IMD004`` image folders. Expected: the
   dataset loads, ``sorted(dataset.unique_patient_ids)`` is
   ``["IMD002", "IMD003", "IMD004"]``, and a warning reports that 197 annotated
   images were dropped for lacking an image or mask.

API Reference
-------------

.. autoclass:: pyhealth.datasets.PH2Dataset
    :members:
    :undoc-members:
    :show-inheritance:

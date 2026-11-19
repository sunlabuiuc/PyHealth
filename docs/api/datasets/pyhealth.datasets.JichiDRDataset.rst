pyhealth.datasets.JichiDRDataset
================================

Overview
--------

Jichi Medical University diabetic retinopathy dataset: 9,939 fundus
photographs of 2,740 patients, each graded on the modified Davis scale (no DR,
simple DR, pre-proliferative DR, proliferative DR), with a single-image grade
and a grade read with the patient's other images. Open access under CC BY 4.0.
``root`` is the extracted ``dmr/`` folder (``list.csv`` plus the JPEGs) or its
parent. The patient id is taken from the file-name prefix.

Getting the data
----------------

- **Source page:** `Davis Grading of One and Concatenated Figures (figshare) <https://figshare.com/articles/dataset/4879853>`_,
  DOI `10.6084/m9.figshare.4879853.v1 <https://doi.org/10.6084/m9.figshare.4879853.v1>`_.
- **Paper:** Takahashi H, Tampo H, Arai Y, Inoue Y, Kawashima H. Applying
  artificial intelligence to disease staging: Deep learning for improved
  staging of diabetic retinopathy. *PLoS ONE* 12(6): e0179790 (2017).
  https://doi.org/10.1371/journal.pone.0179790
- **Access:** open download. No account, registration or data-use agreement
  is needed. Click "Download" on the figshare page, or use the direct link.
- **Download:** the single file ``dmr.tar`` (886,820,858 bytes, about 887 MB):

  .. code-block:: bash

      wget -O dmr.tar https://ndownloader.figshare.com/files/8169134

- **Unpack:** despite the ``.tar`` name the archive is gzip-compressed:

  .. code-block:: bash

      tar -xzf dmr.tar    # creates dmr/

- **Layout:** pass the extracted ``dmr/`` folder (or its parent) as ``root``:

  .. code-block:: text

      <root>/                 # or <root>/dmr/
          list.csv            # Image, Davis_grading_of_one_figure,
                              # Davis_grading_of_concatenated_figures
          1_1_R.jpg           # <patient>_<index>_<eye>.jpg
          1_2_L.jpg
          ...                 # 9,939 JPEGs listed in list.csv

  The derived metadata CSV goes to the PyHealth cache (or ``metadata_dir``),
  never into ``root``.
- **Licence:** `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_.
  Reuse, including commercial use and derivatives, is allowed with
  attribution; there are no NC or ND terms.

API Reference
-------------

.. autoclass:: pyhealth.datasets.JichiDRDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.jichi.parse_jichi_filename

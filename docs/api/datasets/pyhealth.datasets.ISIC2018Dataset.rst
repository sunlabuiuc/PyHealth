pyhealth.datasets.ISIC2018Dataset
===================================

Unified dataset class for the ISIC 2018 challenge, supporting both
**Task 1/2** (lesion segmentation & attribute detection) and **Task 3**
(7-class skin lesion classification) via the ``task`` argument.

For more information see `the ISIC 2018 challenge page <https://challenge.isic-archive.com/data/#2018>`_.

.. note::
   **Licenses differ by task:**

   * ``task="task1_2"`` — `CC-0 (Public Domain) <https://creativecommons.org/share-your-work/public-domain/cc0/>`_.
     No attribution required.
   * ``task="task3"`` — `CC-BY-NC 4.0 <https://creativecommons.org/licenses/by-nc/4.0/>`_.
     Attribution is required; commercial use is not permitted.

.. autoclass:: pyhealth.datasets.ISIC2018Dataset
    :members:
    :undoc-members:
    :show-inheritance:

Data
===============

**pyhealth.data** defines the atomic data structures of this package.

Getting Started
---------------

New to PyHealth's data structures? Start here:

- **Tutorial**: `Introduction to pyhealth.data <https://colab.research.google.com/drive/1y9PawgSbyMbSSMw1dpfwtooH7qzOEYdN?usp=sharing>`_ | `Video <https://www.youtube.com/watch?v=Nk1itBoLOX8&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=2>`_

This tutorial introduces the core data structures in PyHealth:

- **Event**: Represents individual clinical events (diagnoses, procedures, medications, lab results, etc.)
- **Patient**: Contains all events and visits for a single patient, forming the foundation of healthcare data organization

Understanding these structures is essential for working with PyHealth, as they provide the standardized format for representing electronic health records.

API Reference
-------------

.. toctree::
    :maxdepth: 3

    data/pyhealth.data.Event
    data/pyhealth.data.Patient


pyhealth.datasets.OMOPDataset
===================================

The OMOP Common Data Model (CDM) is an open community data standard designed to standardize the structure and content of observational health data. The OMOPDataset class provides a convenient interface for loading and working with OMOP CDM formatted databases.

We can process any OMOP-CDM formatted database. The raw data is processed into a well-structured dataset object providing **flexibility and convenience** for supporting modeling and analysis.

Key Features:
    - Supports OMOP CDM version 5.x
    - Uses Polars for efficient data loading and processing
    - Automatic table loading with YAML configuration
    - Person-centric data organization
    - Links clinical events to visits via visit_occurrence_id
    - Compatible with standard OMOP vocabularies and concept IDs

Refer to the `OMOP CDM documentation <https://www.ohdsi.org/data-standardization/the-common-data-model/>`_ for more information about the data model.

.. autoclass:: pyhealth.datasets.OMOPDataset
    :members:
    :undoc-members:
    :show-inheritance:

   

   
   
   
   

   
   
   
.. _medical_standards:

===============================
Translating Between Medical Standards
===============================

Healthcare data comes in many different coding systems and standards. PyHealth's medical code mapping functionality helps you seamlessly translate between different medical coding systems and look up detailed information about medical codes.

.. note::
   **This module can be used independently** of the main PyHealth pipeline for any medical coding research or applications.

Why Medical Code Translation Matters
====================================

Healthcare systems worldwide use different medical coding standards. In PyHealth, the following are currently supported:

- **ICD-9-CM/ICD-10-CM**: International Classification of Diseases for diagnoses
- **ICD-9-PROC/ICD-10-PCS**: Procedure coding systems
- **CCS/CCS-PROC**: Clinical Classifications Software (grouping categories)
- **NDC**: National Drug Code for medications
- **ATC**: Anatomical Therapeutic Chemical Classification for drugs

Converting between these systems is crucial for:

- **Data Integration**: Combining datasets from different healthcare systems
- **Research Standardization**: Ensuring consistent coding across studies
- **Clinical Decision Support**: Providing unified views of patient data
- **Regulatory Compliance**: Meeting different reporting requirements

Cross-System Code Mapping
=========================

The ``CrossMap`` class enables mapping codes between different coding systems.

Diagnosis Code Mapping
----------------------

Map ICD-9-CM diagnosis codes to Clinical Classifications Software (CCS) categories:

.. code-block:: python

    from pyhealth.medcode import CrossMap

    # Load ICD-9-CM to CCS mapping
    icd9_to_ccs = CrossMap.load("ICD9CM", "CCSCM")
    
    # Map a specific diagnosis code
    ccs_code = icd9_to_ccs.map("82101")  # Closed fracture of base of skull
    print(f"ICD-9-CM 82101 maps to CCS: {ccs_code}")
    
    # Use it like a dictionary
    multiple_codes = ["25000", "4280", "41401"]  # Diabetes, Heart failure, CAD
    for code in multiple_codes:
        mapped = icd9_to_ccs.map(code)
        print(f"{code} → {mapped}")

Drug Code Mapping
-----------------

Convert National Drug Codes (NDC) to Anatomical Therapeutic Chemical (ATC) codes:

.. code-block:: python

    from pyhealth.medcode import CrossMap

    # Load NDC to ATC mapping
    ndc_to_atc = CrossMap.load("NDC", "ATC")
    
    # Map drug codes
    atc_code = ndc_to_atc.map("00527051210")  # Specific NDC number
    print(f"NDC 00527051210 maps to ATC: {atc_code}")
    
    # Batch mapping for efficiency
    ndc_codes = ["00527051210", "00093310701", "00781158530"]
    atc_codes = [ndc_to_atc.map(code) for code in ndc_codes]
    
    for ndc, atc in zip(ndc_codes, atc_codes):
        print(f"NDC {ndc} → ATC {atc}")

Available Mapping Pairs
-----------------------

PyHealth supports numerous cross-system mappings:

.. list-table:: Supported Code Mappings
   :widths: 25 25 50
   :header-rows: 1

   * - Source System
     - Target System
     - Description
   * - ICD9CM
     - CCSCM
     - ICD-9-CM diagnoses to CCS categories
   * - ICD10CM
     - CCSCM
     - ICD-10-CM diagnoses to CCS categories
   * - NDC
     - ATC
     - National Drug Code to ATC classification
   * - ICD9PROC
     - CCSPROC
     - ICD-9-CM procedures to CCS procedure categories
   * - ICD10PROC
     - CCSPROC
     - ICD-10-PCS procedures to CCS procedure categories

Within-System Code Lookup
=========================

The ``InnerMap`` class provides detailed information lookup within a single coding system.

ICD-9-CM Code Lookup
--------------------

Get detailed information about diagnosis codes:

.. code-block:: python

    from pyhealth.medcode import InnerMap

    # Load ICD-9-CM system
    icd9cm = InnerMap.load("ICD9CM")
    
    # Look up detailed information
    code_info = icd9cm.lookup("428.0")
    print(f"Code 428.0: {code_info}")
    
    # Get hierarchical relationships
    ancestors = icd9cm.get_ancestors("428.0")
    print(f"Parent codes: {ancestors}")
    
    descendants = icd9cm.get_descendants("428")
    print(f"Child codes: {descendants}")

ATC Drug Classification
----------------------

Explore the hierarchical structure of drug classifications:

.. code-block:: python

    from pyhealth.medcode import InnerMap

    # Load ATC system
    atc = InnerMap.load("ATC")
    
    # Look up drug information
    drug_info = atc.lookup("C09AA01")  # Captopril
    print(f"ATC C09AA01: {drug_info}")
    
    # Explore hierarchy
    # ATC has 5 levels: Anatomical group → Therapeutic group → 
    # Pharmacological group → Chemical group → Chemical substance
    ancestors = atc.get_ancestors("C09AA01")
    print(f"Drug hierarchy: {ancestors}")

Supported Systems
-----------------

.. list-table:: Available InnerMap Systems
   :widths: 20 30 50
   :header-rows: 1

   * - System
     - Full Name
     - Description
   * - ICD9CM
     - ICD-9-CM
     - International Classification of Diseases, 9th Revision, Clinical Modification
   * - ICD10CM
     - ICD-10-CM
     - International Classification of Diseases, 10th Revision, Clinical Modification
   * - ATC
     - ATC Classification
     - Anatomical Therapeutic Chemical Classification System
   * - NDC
     - National Drug Code
     - FDA's National Drug Code Directory
   * - CCSCM
     - CCS Categories
     - Clinical Classifications Software for diagnoses

Practical Examples
==================

Real-World Use Case: Multi-Site Study
-------------------------------------

Imagine you're conducting a study using data from multiple hospitals that use different coding systems:

.. code-block:: python

    from pyhealth.medcode import CrossMap, InnerMap

    # Hospital A uses ICD-9-CM, Hospital B uses ICD-10-CM
    # Standardize both to CCS categories for analysis
    
    icd9_to_ccs = CrossMap.load("ICD9CM", "CCSCM")
    icd10_to_ccs = CrossMap.load("ICD10CM", "CCSCM")
    
    # Hospital A data (ICD-9-CM codes)
    hospital_a_codes = ["25000", "4280", "41401"]
    standardized_a = [icd9_to_ccs.map(code) for code in hospital_a_codes]
    
    # Hospital B data (ICD-10-CM codes)  
    hospital_b_codes = ["E11.9", "I50.9", "I25.10"]
    standardized_b = [icd10_to_ccs.map(code) for code in hospital_b_codes]
    
    print("Standardized codes from Hospital A:", standardized_a)
    print("Standardized codes from Hospital B:", standardized_b)

Drug Safety Analysis
-------------------

Analyze drug interactions by converting to ATC codes:

.. code-block:: python

    from pyhealth.medcode import CrossMap, InnerMap
    
    # Convert patient's medications from NDC to ATC for analysis
    ndc_to_atc = CrossMap.load("NDC", "ATC")
    atc_system = InnerMap.load("ATC")
    
    patient_medications = ["00527051210", "00093310701", "00781158530"]
    
    for ndc in patient_medications:
        atc = ndc_to_atc.map(ndc)
        if atc:
            drug_info = atc_system.lookup(atc)
            therapeutic_class = atc[:3]  # First 3 characters = therapeutic subgroup
            print(f"NDC {ndc} → ATC {atc} ({drug_info}) - Class: {therapeutic_class}")

Integration with PyHealth Pipeline
=================================

Medical code mapping integrates seamlessly with the main PyHealth pipeline:

.. code-block:: python

    from pyhealth.datasets import MIMIC3Dataset
    from pyhealth.medcode import CrossMap
    
    # Load dataset with automatic code mapping
    mimic3base = MIMIC3Dataset(
        root="path/to/mimic3/",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={
            "ICD9CM": "CCSCM",      # Map diagnoses to CCS categories
            "ICD9PROC": "CCSPROC",  # Map procedures to CCS categories  
            "NDC": "ATC"            # Map drugs to ATC codes
        }
    )

This automatically standardizes all medical codes during dataset loading, making your downstream analysis more consistent and interpretable.

Next Steps
==========

- **Explore Available Mappings**: Check what coding systems are available for your data
- **Custom Mappings**: Learn how to add custom mapping files for proprietary coding systems
- **Integration**: Use medical code mapping within your PyHealth ML pipelines
- **Validation**: Verify mapping quality and coverage for your specific use case

.. seealso::
   
   - :doc:`api/medcode` - Complete medical code API documentation
   - :doc:`tutorials` - Interactive tutorials with real examples

.. _how_to_contribute:

=====================
How to Contribute
=====================

We welcome contributions to PyHealth! This guide will help you get started with contributing datasets, tasks, models, bug fixes, or other improvements to the project.

Getting Started
===============

Prerequisites
-------------

PyHealth uses GitHub for development, so you'll need a GitHub account to contribute.

Setting Up Your Development Environment
---------------------------------------

To start contributing to PyHealth:

1. **Fork the PyHealth repository** on GitHub
2. **Clone your forked repository** to your local machine:

   .. code-block:: bash

      git clone https://github.com/your_username/PyHealth.git
      cd PyHealth

3. **Install dependencies**:

   .. code-block:: bash

      pip install -e . 

4. **Implement your code** with proper test cases
5. **Push changes** to your forked repository
6. **Create a pull request** to the main PyHealth repository

   - Target the ``main`` branch
   - Enable edits by maintainers
   - Rebase with the remote ``sunlabuiuc`` main branch before creating the PR

Implementation Requirements
===========================

Code File Headers
-----------------

For new contributors, include the following information at the top of your code files:

- Your name(s)
- Your NetID(s) (if applicable for UIUC students)
- Paper title (if applicable to a reproducibility contribution)
- Paper link (if applicable)
- Description of the task/dataset/model you're implementing

Code Style and Documentation
-----------------------------

**General Guidelines:**

- Use object-oriented programming with well-defined and typed functions
- Follow snake_case naming for variables and functions (e.g., ``this_variable``)
- Use PascalCase for class names (e.g., ``ThisClass``)
- Follow PEP8 style with 88 character line length
- Use Google style for docstrings

**Function Documentation Requirements:**

Each function must document:

- **Input arguments**: Define variable types and descriptions
- **Output arguments**: Define variable types and descriptions  
- **High-level description** of what the function does
- **Example use case** or where it will be called

**Example Well-Documented Function:**

.. code-block:: python

   def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
       """Helper functions which parses patients and admissions tables.

       Will be called in `self.parse_tables()`

       Docs:
           - patients: https://mimic.mit.edu/docs/iv/modules/hosp/patients/
           - admissions: https://mimic.mit.edu/docs/iv/modules/hosp/admissions/

       Args:
           patients: a dict of `Patient` objects indexed by patient_id.

       Returns:
           The updated patients dict.
       """

Types of Contributions
======================

Contributing a Dataset
----------------------

All datasets must follow these guidelines:

- **Inherit from BaseDataset**: All datasets must inherit from the appropriate BaseDataset class
- **Follow established patterns**: 
  
  - For EHR datasets: See the `MIMIC4 dataset example <https://github.com/sunlabuiuc/PyHealth/blob/main/pyhealth/datasets/mimic4.py>`_
  - For image datasets: See the `CovidCXR dataset example <https://github.com/sunlabuiuc/PyHealth/blob/main/pyhealth/datasets/covidcxr.py>`_ where each folder represents a sample

- **Include a test task**: Datasets should ideally have an associated task for testing purposes

**Key Requirements:**

- Define all required variables outlined in the BaseDataset documentation
- Provide clear data loading and processing methods
- Include proper error handling and validation

Contributing a Task
-------------------

Tasks must follow the established task class structure:

- **Inherit from base task class**: Follow the pattern defined in existing tasks
- **Examples to reference**:
  
  - `Mortality prediction task <https://github.com/sunlabuiuc/PyHealth/blob/main/pyhealth/tasks/mortality_prediction.py>`_
  - `X-ray classification task <https://github.com/sunlabuiuc/PyHealth/blob/main/pyhealth/tasks/chest_xray_classification.py>`_

- **Flexibility**: Tasks can include various implementation details but must have clear inputs/outputs
- **Test cases**: Include example test cases with defined inputs and expected outputs

Contributing a Model
--------------------

Models must follow the model base class structure:

- **Inherit from BaseModel**: All models must inherit from the appropriate base model class
- **Reference implementation**: See the `RNN model example <https://github.com/sunlabuiuc/PyHealth/blob/main/pyhealth/models/rnn.py>`_
- **Test cases**: Include example test cases with dummy inputs and expected outputs

**Key Requirements:**

- Implement required abstract methods from the base class
- Provide clear forward pass implementation
- Include proper initialization and configuration methods

Test Case Requirements
======================

Every contribution must include two types of test cases:

1. **Automated tests**: These will be run by our continuous integration system
2. **Manual test cases**: You must define these yourself with:

   - Clear input specifications
   - Expected output formats
   - Example usage demonstrating functionality

**Note**: You can use frontier LLMs to help generate basic test cases, which we consider valid as long as they are reasonable and comprehensive.

All unit tests should be placed in the `tests/` directory following the existing structure, with 'tests/core/' for core functionality tests.

Writing Fast and Performant Tests
---------------------------------

Guidelines for authoring test cases:

- Keep tests fast: avoid large data processing; prefer tiny, synthetic fixtures.
- Place core unit tests in ``tests/core/`` and name files ``test_*.py``.
- Avoid network access and external services; use local, in-memory data.
- Use small configurations: minimal sample sizes, tiny models, small batch sizes, and ``epochs=1``.
- Make tests deterministic: set random seeds and avoid time-based randomness.
- Stub or monkeypatch heavy components (I/O, model training) when the logic under test allows.
- Skip or gate any heavyweight tests behind explicit markers; by default, all tests must run quickly in CI.

Pull Request Guidelines
=======================

Formatting Your Pull Request
----------------------------

Every pull request must include the following information in the comment:

1. **Who you are** (include NetID if you're an Illinois student)
2. **Type of contribution** (dataset, task, model, bug fix, etc.)
3. **High-level description** of what you've implemented
4. **File guide**: Quick rundown of which files to examine to test your implementation

**Example PR Description:**

.. code-block:: text

   **Contributor:** Jane Doe (jdoe2@illinois.edu)
   
   **Contribution Type:** New Dataset
   
   **Description:** Added support for the XYZ Hospital dataset with patient 
   admission records and diagnostic codes. Includes data preprocessing and 
   sample task for mortality prediction.
   
   **Files to Review:**
   - `pyhealth/datasets/xyz_hospital.py` - Main dataset implementation
   - `pyhealth/tasks/xyz_mortality.py` - Example task
   - `tests/core/test_xyz_dataset.py` - Test cases

Review Process
--------------

After submitting your pull request:

1. Maintainers will review your code for style, functionality, and completeness
2. Automated tests will be run to ensure compatibility
3. You may be asked to make revisions based on feedback
4. Once approved, your contribution will be merged into the main branch

Getting Help
============

If you need assistance:

- Check existing issues and discussions on GitHub
- Review similar implementations in the codebase
- Reach out to maintainers through GitHub issues
- Consider using LLMs to help with code formatting and documentation

We appreciate your contributions to making PyHealth better for the healthcare AI community!
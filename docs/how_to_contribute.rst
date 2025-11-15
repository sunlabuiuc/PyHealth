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

Data Access for Testing
========================

If you're new to working with healthcare datasets and need access to data for testing your contributions, here are some helpful resources:

Getting MIMIC Access
--------------------

For full access to MIMIC datasets, you'll need to complete credentialing through PhysioNet. Detailed instructions are available in our `Getting MIMIC Access Guide <https://docs.google.com/document/d/1NHgXzSPINafSg8Cd_whdfSauFXgh-ZflZIw5lu6k2T0/edit?usp=sharing>`_.

Using Demo Datasets
-------------------

While completing the credentialing process, you can use publicly available demo datasets to develop and test your code:

- **MIMIC-IV Demo**: A subset of 100 patients from MIMIC-IV, available at https://physionet.org/content/mimic-iv-demo/2.2/
- **MIMIC-III Demo**: A subset of 100 patients from MIMIC-III, available at https://physionet.org/content/mimiciii-demo/1.4/

These demo datasets are open access and do not require credentialing. They're perfect for:

- Testing your dataset implementations
- Developing new tasks
- Creating reproducible examples
- Verifying your code works before requesting full data access

**Important Note on Test Data**: While demo datasets are useful for development, your test cases should use **small synthetic/pseudo data** to ensure fast execution in continuous integration. Demo datasets (even with 100 patients) are too large for unit tests. See the "Writing Fast and Performant Tests" section below for guidance on creating minimal test fixtures.

Examples and Tutorials
----------------------

Before implementing new features, review our tutorials page for examples. The tutorials include Jupyter notebooks demonstrating:

- How to load and process datasets
- How to define and run tasks
- How to train and evaluate models
- Best practices for working with PyHealth

Visit the :ref:`tutorials` page in our documentation to explore these examples.

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

Required File Updates
=====================

When contributing new features to PyHealth, you'll need to create and update several files to ensure your contribution is properly integrated into the library. Below are detailed examples of what files should be included for different types of contributions.

Contributing a New Dataset: Complete File Checklist
----------------------------------------------------

When adding a new dataset (e.g., ``NewDataset``), you should create and update the following files:

**1. Core Implementation File**

``pyhealth/datasets/new_dataset.py``

- Inherits from appropriate BaseDataset class
- Implements all required data loading and parsing methods
- Includes comprehensive docstrings following Google style
- Contains proper type hints for all methods

**2. Documentation File**

``docs/api/datasets/pyhealth.datasets.new_dataset.rst``

Create a new reStructuredText file documenting your dataset:

.. code-block:: rst

   pyhealth.datasets.new_dataset
   =============================

   Overview
   --------

   Brief description of the dataset, its source, and key characteristics.

   API Reference
   -------------

   .. autoclass:: pyhealth.datasets.NewDataset
       :members:
       :undoc-members:
       :show-inheritance:

**3. Update Dataset Index**

``docs/api/datasets.rst``

Add your new dataset to the table of contents:

.. code-block:: rst

   .. toctree::
       :maxdepth: 4

       datasets/pyhealth.datasets.existing_dataset1
       datasets/pyhealth.datasets.existing_dataset2
       datasets/pyhealth.datasets.new_dataset
       ...

**4. Test Case File**

``tests/core/test_new_dataset.py``

Create comprehensive test cases that verify:

- Dataset can be instantiated correctly
- Data loading works with demo/synthetic data
- All parsing methods execute without errors
- Output formats match expected schemas
- Edge cases are handled appropriately

**Critical: Use Synthetic/Pseudo Data for Tests**

Your test cases must use **small, synthetic data** rather than real datasets. Create minimal CSV files or in-memory data structures with just enough records to verify functionality:

- Use 2-5 patients maximum (not 100!)
- Create a few events per patient (5-10 records total)
- Generate this data programmatically or as tiny fixture files in ``test-resources/``
- Ensure tests complete in milliseconds, not seconds

Example test structure with synthetic data:

.. code-block:: python

   import unittest
   import tempfile
   import pandas as pd
   from pathlib import Path
   from pyhealth.datasets import NewDataset

   class TestNewDataset(unittest.TestCase):
       def setUp(self):
           # Create temporary directory with synthetic data
           self.test_dir = tempfile.mkdtemp()
           
           # Create minimal synthetic patient data
           patients_df = pd.DataFrame({
               'patient_id': [1, 2, 3],
               'birth_date': ['1980-01-01', '1975-05-15', '1990-03-20'],
               'gender': ['M', 'F', 'M']
           })
           patients_df.to_csv(Path(self.test_dir) / 'patients.csv', index=False)
           
           # Create minimal synthetic events
           events_df = pd.DataFrame({
               'patient_id': [1, 1, 2, 3],
               'event_time': ['2020-01-01', '2020-01-05', '2020-02-01', '2020-03-01'],
               'event_code': ['D001', 'P002', 'D003', 'D001']
           })
           events_df.to_csv(Path(self.test_dir) / 'events.csv', index=False)
           
           # Instantiate dataset with synthetic data
           self.dataset = NewDataset(root=self.test_dir)
       
       def test_load_data(self):
           # Test that data loads correctly
           self.assertEqual(len(self.dataset.patients), 3)
       
       def test_parse_tables(self):
           # Test that parsing produces expected format
           patient = self.dataset.patients[1]
           self.assertIsNotNone(patient)
           self.assertTrue(hasattr(patient, 'visits'))

**5. Associated Task (Optional but Recommended)**

``pyhealth/tasks/new_dataset_task.py``

If your dataset enables a specific machine learning task, create a task file that:

- Defines the task's objective (e.g., mortality prediction, disease classification)
- Specifies input features and label generation
- Implements data preprocessing specific to the task
- Includes example usage in docstrings

**What Should Be Tested:**

For each new dataset contribution, your test cases should verify:

- **Data Loading**: The dataset can locate and load files from the specified directory
- **Patient Parsing**: Patient-level information is correctly extracted and structured
- **Event Parsing**: Clinical events (diagnoses, procedures, medications, etc.) are properly parsed
- **Data Integrity**: No missing critical fields, appropriate data types, valid value ranges
- **Sample Output**: At least one complete example showing input data → processed output
- **Performance**: Tests run quickly (**critical**: always use tiny synthetic data, never real datasets)

**Why Synthetic Data Matters:**

- **Speed**: CI/CD pipelines must complete quickly; tests with real data can take minutes or hours
- **Reproducibility**: Synthetic data ensures tests work without credentials or external dependencies
- **Maintainability**: Small fixtures are easy to understand, modify, and debug
- **Coverage**: You can create edge cases and corner cases easily with synthetic data

Contributing a New Task: File Checklist
----------------------------------------

**Files Required:**

1. ``pyhealth/tasks/new_task.py`` - Core task implementation
2. ``docs/api/tasks/pyhealth.tasks.new_task.rst`` - Documentation file
3. ``docs/api/tasks.rst`` - Update the task index
4. ``tests/core/test_new_task.py`` - Test cases with synthetic data
5. Example usage in ``examples/`` directory (optional but encouraged)

**Task Test Requirements:**

Task tests should verify:

- Task can process synthetic dataset samples correctly
- Label generation works as expected
- Feature extraction produces correct output format
- Edge cases (missing data, empty visits) are handled gracefully

Use synthetic ``Patient`` objects or minimal datasets (2-5 patients) to test task logic quickly.

Contributing a New Model: File Checklist
-----------------------------------------

**Files Required:**

1. ``pyhealth/models/new_model.py`` - Core model implementation
2. ``docs/api/models/pyhealth.models.new_model.rst`` - Documentation file
3. ``docs/api/models.rst`` - Update the model index
4. ``tests/core/test_new_model.py`` - Test cases with dummy data
5. Example usage in ``examples/`` directory (optional but encouraged)

**Model Test Requirements:**

Model tests should verify:

- Model can be instantiated with various configurations
- Forward pass executes correctly with dummy inputs
- Output shapes match expected dimensions
- Gradient computation works (for trainable models)
- Model can save and load state correctly

**Use minimal synthetic tensors for model tests:**

.. code-block:: python

   import torch
   import unittest
   from pyhealth.models import NewModel

   class TestNewModel(unittest.TestCase):
       def test_forward_pass(self):
           # Use tiny dimensions for fast testing
           model = NewModel(
               input_dim=10,
               hidden_dim=8,  # Keep small!
               output_dim=2
           )
           
           # Create minimal synthetic input (batch_size=2)
           x = torch.randn(2, 10)
           
           # Test forward pass
           output = model(x)
           self.assertEqual(output.shape, (2, 2))

General Documentation Guidelines
---------------------------------

For all contributions:

- **Keep documentation consistent** with existing files in the same category
- **Include working examples** in docstrings whenever possible
- **Reference related classes/methods** using Sphinx cross-references
- **Update index files** so your contribution appears in the documentation
- **Use small, reproducible examples** in your documentation that others can easily run

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

**Test cases must run quickly in CI/CD.** Follow these guidelines to ensure your tests are fast and efficient:

**Creating Synthetic/Pseudo Data:**

The most important rule: **never use real datasets in test cases.** Instead, create minimal synthetic data:

1. **Generate data programmatically**: Use pandas, numpy, or Python dictionaries to create tiny datasets in-memory
2. **Use fixture files**: Place small synthetic CSV/JSON files in ``test-resources/`` (max a few KB each)
3. **Keep it minimal**: 2-5 patients, 5-20 events total - just enough to test logic
4. **Use temporary directories**: Create data in ``tempfile.mkdtemp()`` and clean up after tests

**Example of creating synthetic data in a test:**

.. code-block:: python

   import tempfile
   import pandas as pd
   from pathlib import Path

   def setUp(self):
       # Create temporary directory
       self.test_dir = tempfile.mkdtemp()
       
       # Generate minimal synthetic CSV files
       pd.DataFrame({
           'id': [1, 2, 3],
           'value': ['A', 'B', 'C']
       }).to_csv(Path(self.test_dir) / 'data.csv', index=False)
       
       # Now use self.test_dir as your dataset root

**Additional Performance Guidelines:**

- **Keep tests fast**: Each test should complete in milliseconds; entire suite in seconds
- **Place core unit tests** in ``tests/core/`` and name files ``test_*.py``
- **Avoid network access**: No external APIs, downloads, or database connections
- **Use small configurations**: Minimal sample sizes, tiny models (e.g., ``hidden_dim=4``), ``batch_size=2``, ``epochs=1``
- **Make tests deterministic**: Set random seeds (``np.random.seed(42)``, ``torch.manual_seed(42)``)
- **Mock heavy operations**: Stub or monkeypatch I/O, model training, or expensive computations when testing logic
- **Skip heavyweight tests**: Use pytest markers for any tests that can't be made fast; default tests must run quickly

**What NOT to do in tests:**

- ❌ Load MIMIC-III/IV demo datasets (too large)
- ❌ Download data from the internet
- ❌ Train models for multiple epochs
- ❌ Use real medical images or large files
- ❌ Require credentials or external databases
- ❌ Run tests that take > 1 second each

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
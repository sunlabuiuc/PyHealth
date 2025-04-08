.. _how_to_contribute:

=====================
How to Contribute
=====================

Thank you for your interest in contributing to PyHealth! We welcome contributions of all types, including bug fixes, new features, documentation improvements, and more.

Contribution Guidelines
=======================

- For **bug fixes** and **documentation improvements**, you can get started on your own.
- For **major framework updates, new datasets, or new models**, it's best to **open an issue and discuss with the PyHealth team first** before proceeding.

Getting Started
===============

1. **Fork the Repository**: Navigate to the PyHealth GitHub repository and click the **Fork** button.
2. **Clone Your Fork**:
   .. code-block:: bash

      git clone https://github.com/your-username/PyHealth.git
      cd PyHealth
3. **Create a New Branch**:
   .. code-block:: bash

      git checkout -b feature-branch

Setting Up the Development Environment
======================================

To avoid interfering with an existing PyHealth installation via `pip`, it is recommended to create a new Python environment specifically for development.
You can use `conda` or `virtualenv` to do so.

Making Changes
==============

1. **Write Code**: Implement your changes in a clean and modular way.
2. **Follow Code Style**: Ensure your code follows `black` formatting:
   .. code-block:: bash

      black .
3. **Write Tests**: Add unit tests for your changes in `unittests/`.
4. **Run Tests**: Before submitting, make sure all tests pass:
   .. code-block:: bash

      pytest

Submitting Your Changes
=======================

1. **Commit Changes**:
   .. code-block:: bash

      git add .
      git commit -m "Description of your changes"
2. **Push to Your Fork**:
   .. code-block:: bash

      git push origin feature-branch
3. **Open a Pull Request (PR)**:
   - Go to the original PyHealth repository.
   - Click **New Pull Request**.
   - Select your fork and branch.
   - Provide a clear description of your changes.
   - Submit the PR.

Review Process
==============

- The PyHealth maintainers will review your PR and provide feedback.
- Make any requested changes and push updates to your PR.
- Once approved, your changes will be merged into the main branch!

Need Help?
==========

- Check the `Issues` tab on GitHub.
- Reach out via `email` if applicable.

We appreciate your contributions and look forward to working with you!
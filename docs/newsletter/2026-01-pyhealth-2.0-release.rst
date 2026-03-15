PyHealth 2.0 Early Release Announcement
=====================================

**January 2026**

Hey everyone,

The PyHealth team has been really hard at work, trying to modernize much of the codebase and models to leverage our new and improved backend. Seeing as how the majority of our bounties are completed (check out the progress at our `bounties spreadsheet <https://docs.google.com/spreadsheets/d/1ruzKu-sTUnGZ3i9JPXzvjVCvxRFn6QJv2YLtVdXsmuE/edit?usp=sharing>`_), we've decided to do an early full release of PyHealth 2.0!

What's New
----------

The full technical report with comparisons against other frameworks is now available: https://arxiv.org/abs/2601.16414

If you use PyHealth 2.0 in your research, please cite:

.. code-block:: bibtex

    @misc{wu2026pyhealth20comprehensiveopensource,
        title={PyHealth 2.0: A Comprehensive Open-Source Toolkit for Accessible and Reproducible Clinical Deep Learning}, 
        author={John Wu and Yongda Fan and Zhenbang Wu and Paul Landes and Eric Schrock and Sayeed Sajjad Razin and Arjun Chatterjee and Naveen Baskaran and Joshua Steier and Andrea Fitzpatrick and Bilal Arif and Rian Atri and Jathurshan Pradeepkumar and Siddhartha Laghuvarapu and Junyi Gao and Adam R. Cross and Jimeng Sun},
        year={2026},
        eprint={2601.16414},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
        url={https://arxiv.org/abs/2601.16414}
    }

**This means when you use** ``pip install pyhealth``, **it no longer installs the legacy version PyHealth 1.16, but rather the PyHealth 2.0.1 release!**

Key Features in PyHealth 2.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Dynamic memory support**: We've finally been able to train a clinical predictive model on MIMIC-IV on a desktop.
* **Parallelized worker usage**: See up to 39x faster ML task processing performance compared to a naive single-threaded pandas solution.
* **Refined API changes**: Enabling streamlined patient data exploration and task customization.
* **Multimodal dataloaders**: Process different data modalities seamlessly.
* **New models**: Expanded model library for various clinical tasks.
* **New datasets**: Support for additional healthcare datasets.
* **New interpretability module**: Better understand your model predictions.

Future Roadmap
--------------

We're still a long way from a fully-mature production-grade research toolkit. There's still a variety of other problems that the PyHealth Research Initiative aims to address through peer-reviewed research and additional contributions to the package.

Finishing Up Remaining Bounties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We are still actively looking for those who are interested in contributing and joining the community. Please see our `Research Initiative Call <https://pyhealth.readthedocs.io/en/latest/newsletter/2025-12-research-initiative-call.html>`_ for more information.

Multimodality and Missing Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One key new benefit of PyHealth 2.0's dataloaders is that they enable processing of any type of modality within a single task given a dataset. However:

* The number of processors in ``pyhealth.processors`` that directly handle the challenge of missing values is still small.
* The number of models that can take in a wide range of features (time-series, structured EHR, text, etc.) is also limited.

We're working on expanding both of these areas to better support real-world clinical data challenges.

Improving the Interpretability Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We'd like to resolve technical challenges like standardizing the procedure to make PyHealth models compatible with the interpretability module. Currently, it only supports models with access to gradient hooks and input embeddings for continuous values.

This isn't an easy challenge, but we felt it was worthwhile in looking into better ways of improving the state of interpretability in clinical AI. Ultimately, our goal is to better answer: **What is the best interpretability approach for clinical predictive models?** We aim to do this by enabling interpretability across a wider range of models and methods.

Robust Distribution Shift and Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most ML pipelines assume i.i.d. data. Most clinical data is not.

We're actively building out our conformal prediction approaches towards enabling more robust uncertainty quantification when your calibration set is not necessarily aligned with your test set. The goal here is to:

* Improve coverage guarantees
* Explore if it's feasible to truly make your models trustworthy to deploy even when model performance is imperfect and prediction sets are necessary

Get Involved
------------

Interested in contributing to PyHealth? Check out:

* `Research Initiative <https://pyhealth.readthedocs.io/en/latest/newsletter/2025-12-research-initiative-call.html>`_
* `How to Contribute <https://pyhealth.readthedocs.io/en/latest/how_to_contribute.html>`_
* `GitHub Repository <https://github.com/sunlabuiuc/PyHealth>`_

Thanks for being part of the PyHealth community!

----

**The PyHealth Team**


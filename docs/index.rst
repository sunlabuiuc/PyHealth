.. PyHealth documentation master file, created by
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyHealth
====================================

**The Python Library for Healthcare AI**

Build, test, and deploy healthcare machine learning models with ease. PyHealth is designed for both **ML researchers and medical practitioners**. We can make your **healthcare AI applications** easier to develop, test and validate. Your development process becomes more flexible and more customizable. `[GitHub] <https://github.com/sunlabuiuc/PyHealth>`_ 

**Key Features**

- **Dramatically simpler**: Build any healthcare AI model in ~7 lines of code
- **Blazing fast**: Up to 39× faster than pandas
- **Memory efficient**: Runs on 16GB laptops 
- **True multimodal**: Unified API for EHR, medical images, biosignals, clinical text, and genomics
- **Production-ready**: 25+ pre-built models, 20+ tasks, 12+ datasets with comprehensive evaluation tools
- **Healthcare-first**: Built-in medical coding standards (ICD, CPT, NDC, ATC) and clinical datasets (MIMIC, eICU, OMOP)


.. image:: https://img.shields.io/readthedocs/pyhealth?logo=readthedocs&label=docs&version=latest
   :target: https://pyhealth.readthedocs.io/en/latest/
   :alt: Docs

.. image:: https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white
   :target: https://discord.gg/mpb835EHaX
   :alt: Discord

.. image:: https://img.shields.io/badge/Mailing%20List-Subscribe-blue?logo=gmail&logoColor=white
   :target: https://docs.google.com/forms/d/e/1FAIpQLSfpJB5tdkI7BccTCReoszV9cyyX2rF99SgznzwlOepi5v-xLw/viewform?usp=header
   :alt: Mailing list

.. image:: https://img.shields.io/pypi/v/pyhealth.svg?color=brightgreen
   :target: https://pypi.org/project/pyhealth/
   :alt: PyPI version

.. image:: https://img.shields.io/github/stars/yzhao062/pyhealth.svg
   :target: https://github.com/sunlabuiuc/pyhealth/stargazers
   :alt: GitHub stars


.. image:: https://img.shields.io/github/forks/yzhao062/pyhealth.svg?color=blue
   :target: https://github.com/sunlabuiuc/pyhealth/network
   :alt: GitHub forks


.. image:: https://static.pepy.tech/badge/pyhealth
   :target: https://pepy.tech/project/pyhealth
   :alt: Downloads


.. image:: https://img.shields.io/badge/Tutorials-Google%20Colab-red
   :target: https://pyhealth.readthedocs.io/en/latest/tutorials.html
   :alt: Tutorials


.. image:: https://img.shields.io/badge/YouTube-16%20Videos-red
   :target: https://www.youtube.com/playlist?list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV
   :alt: YouTube


----------

 **[News!]** Join us for **PyHealth Casual Chats** – informal sessions where you can ask questions, discuss research ideas, or talk about PyHealth developments! Everyone is welcome. `Join Zoom → <https://illinois.zoom.us/j/83607767000?pwd=xXKdKKs2YBH8d0UMWeUiYNEl0lhDoU.1>`_ | `Add to Calendar → <https://calendar.app.google/mo9G9RNY5XRLo6hu5>`_

 **[News!]** We are continuously implementing good papers and benchmarks into PyHealth, checkout the `[Planned List] <https://docs.google.com/spreadsheets/d/1PNMgDe-llOm1SM5ZyGLkmPysjC4wwaVblPLAHLxejTw/edit#gid=159213380>`_. Welcome to pick one from the list and send us a PR or add more influential and new papers into the plan list.

----------


.. -----


.. **Build Status & Coverage & Maintainability & License**

.. .. image:: https://travis-ci.org/yzhao062/pyhealth.svg?branch=master
..    :target: https://travis-ci.org/yzhao062/pyhealth
..    :alt: Build Status


.. .. image:: https://ci.appveyor.com/api/projects/status/1kupdy87etks5n3r/branch/master?svg=true
..    :target: https://ci.appveyor.com/project/yzhao062/pyhealth/branch/master
..    :alt: Build status


.. .. image:: https://api.codeclimate.com/v1/badges/bdc3d8d0454274c753c4/maintainability
..    :target: https://codeclimate.com/github/yzhao062/pyhealth/maintainability
..    :alt: Maintainability


.. .. image:: https://img.shields.io/github/license/yzhao062/pyhealth
..    :target: https://github.com/yzhao062/pyhealth/blob/master/LICENSE
..    :alt: License



Get Started in Minutes
=============================

PyHealth makes healthcare AI development simple and powerful. Build production-ready models with just a few lines of code.

.. code-block:: python

   from pyhealth.datasets import MIMIC3Dataset
   from pyhealth.tasks import MortalityPredictionMIMIC3
   from pyhealth.models import RNN
   from pyhealth.trainer import Trainer

   # Load healthcare data
   dataset = MIMIC3Dataset(root="data/", tables=["diagnoses_icd", "procedures"])
   samples = dataset.set_task(MortalityPredictionMIMIC3())

   # Train model
   model = RNN(dataset=samples)
   trainer = Trainer(model=model)
   trainer.train(train_dataloader, val_dataloader, epochs=50)

**That's it!** You now have a trained healthcare AI model ready for deployment.

 

Quick Navigation
======================

.. list-table::
   :widths: 50 50
   :class: borderless

   * - **Getting Started**

       Build your first healthcare AI model in 5 minutes

       :doc:`Read Guide → <how_to_get_started>`

     - **Why PyHealth?**

       Discover the comprehensive benefits and capabilities

       :doc:`Learn More → <why_pyhealth>`

   * - **Installation**

       Install PyHealth and set up your environment

       :doc:`Install Now → <install>`

     - **Research Initiative**

       Join our year round research program and contribute

       :doc:`View Projects → <research_initiative>`

   * - **Tutorials**

       Hands-on notebooks and step-by-step guides

       :doc:`Open Tutorials → <tutorials>`

     - **Medical Standards**

       Translate between medical coding systems (ICD, NDC, ATC, CCS)

       :doc:`Explore → <api/medcode>`

   * - **Community**

       Join our Discord server and contribute to PyHealth

       `Discord → <https://discord.gg/mpb835EHaX>`_ | :doc:`Contribute → <how_to_contribute>`

     - **Newsletter**

       Stay updated with the latest PyHealth developments

       :doc:`Read Newsletter → <newsletter>`



.. **Citing PyHealth**\ :

.. `PyHealth paper <https://arxiv.org/abs/2101.04209>`_ is under review at
.. `JMLR <http://www.jmlr.org/>`_ (machine learning open-source software track).
.. If you use PyHealth in a scientific publication, we would appreciate
.. citations to the following paper::

..     @article{
..     }



.. **Key Links and Resources**\ :


.. * `View the latest codes on Github <https://github.com/ycq091044/PyHealth-OMOP>`_
.. * `Execute Interactive Jupyter Notebooks <https://mybinder.org/v2/gh/yzhao062/pyhealth/master>`_
.. * `Check out the PyHealth paper <https://github.com/yzhao062/pyhealth>`_



----




.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Getting Started

   
   why_pyhealth
   how_to_get_started
   install
   tutorials
   .. advance_tutorials


.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Documentation

   api/data
   api/datasets
   api/graph
   api/tasks
   api/models
   api/processors
   api/interpret
   api/trainer
   api/tokenizer
   api/metrics
   api/medcode
   api/calib


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   how_to_contribute
   research_initiative
   newsletter
   live
   log
   about



.. .. bibliography:: references.bib
..    :cited:
..    :labelprefix: A
..    :keyprefix: a-


.. .. rubric:: References

..    Indices and tables
..    ==================

..    * :ref:`genindex`
..    * :ref:`modindex`
..    * :ref:`search`

Memory Optimization: PyHealth Now Runs on 16GB RAM
==================================================

*Published: December 18, 2025*

Hey everyone,

Over the past month, we've been working hard to reduce PyHealth's memory footprint when working with large public EHR datasets like MIMIC-IV. The result? **A major backend overhaul that lets you run PyHealth on machines with as little as 16GB of RAM—even when processing MIMIC-IV's 300,000+ patients with both medical codes and lab events spanning millions of records.** This makes clinical predictive modeling substantially more accessible, especially as memory prices are expected to keep rising [1]_.


----


The Scale We're Dealing With
----------------------------

To construct a multimodal mortality task in MIMIC-IV, we parse and scan:

* 315,460 patients
* 454,324 admissions
* 5,006,884 diagnosis codes
* 704,124 procedure codes
* 124,342,638 lab events

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 25

   * - Metric
     - Pandas
     - PyHealth 2.0a10
     - PyHealth 2.0a13
   * - Memory Required
     - 49.23GB
     - 385GB
     - **16GB** ⚡
   * - Run Time (h)
     - 26.03
     - 2.25
     - 3.97

\*Results benchmarked on an AMD EPYC 7513 32-Core Processor Workstation.


----


What Changed?
-------------

Previously, PyHealth loaded all patient data into memory whenever any computation was needed—meaning you'd load every patient's records just to analyze one patient's EHR. Fast with Polars, but wildly wasteful.

In PyHealth 2.0a13+, we've switched to lazy-loading patient data, pulling records into memory only when actually needed.


----


We Need Your Help!
------------------

Please test the new approach and report any issues here: `[Tracking] Tracking issue for the new memory efficient dataset · Issue #740 <https://github.com/REDACTED_ORG/PyHealth/issues/740>`_

Don't hesitate to share comments or suggestions.


----


Looking Ahead
-------------

This memory optimization is just the first step in our mission to make healthcare AI truly accessible. By dramatically lowering the computational barriers to entry, we're opening the door for researchers and clinicians with limited resources—whether that's a REDACTED_ROLE with a laptop or a hospital in a low-resource setting—to build and deploy meaningful healthcare AI solutions.

We believe democratizing these tools is essential for ensuring healthcare AI benefits everyone, not just those with access to massive compute infrastructure.


----


References
----------

.. [1] https://www.tomsguide.com/news/live/ram-price-crisis-updates

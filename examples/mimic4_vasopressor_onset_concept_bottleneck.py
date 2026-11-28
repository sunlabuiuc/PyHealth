# Authors: Shashank Bhagavatula (sbhag3), Aarushi Sharma (sharma90)
# Paper: Learning Optimal Summaries of Clinical Time-series with Concept
#   Bottleneck Models (Wu et al., MLHC 2022)
# Paper link: https://proceedings.mlr.press/v182/wu22a/wu22a.pdf
# Description: Full pipeline and ablation study on MIMIC-IV (synthetic by
#   default).

"""Concept bottleneck model for vasopressor onset on MIMIC-IV.

Paper: Wu, Parbhoo, Havasi, Doshi-Velez. "Learning Optimal Summaries of
Clinical Time-series with Concept Bottleneck Models." MLHC 2022.
https://proceedings.mlr.press/v182/wu22a.html

The paper uses MIMIC-III; this is the same pipeline and ablation study on
MIMIC-IV (``MIMIC4EHRDataset`` -> ``VasopressorOnsetMIMIC4`` ->
``ConceptBottleneckModel`` and its baselines), an extension of this port.
It runs ``examples/mimic3_vasopressor_onset_concept_bottleneck.py`` with
``--dataset mimic4`` as the default; the experimental setup, the
ablations and all options are documented there.

No data needed (default): writes a small synthetic MIMIC-IV-format
dataset to a temporary directory and runs everything on it::

    python examples/mimic4_vasopressor_onset_concept_bottleneck.py

Real data: point ``--root`` at MIMIC-IV (credentialed) or at its open
100-patient demo (https://physionet.org/content/mimic-iv-demo/2.2/)::

    python examples/mimic4_vasopressor_onset_concept_bottleneck.py --root DIR

Demo run (Nov 24, 2026, demo v2.2, seeds 1-3): 59 of 100 patients pass the
cohort rules and 6 are positive, so each test split holds about one
positive. The script runs end to end, but the metrics it prints on the
demo are noise and are not comparable to the paper.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mimic3_vasopressor_onset_concept_bottleneck import main

if __name__ == "__main__":
    main(default_dataset="mimic4")

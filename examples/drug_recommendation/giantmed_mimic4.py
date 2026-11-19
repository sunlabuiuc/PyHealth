"""Run GiantMed on the public MIMIC-IV demo; see giantmed_README.md.

Without a model option, inference stays local and leaves SafeDrug scores intact.
Use --demo-refinement to exercise updates with a deterministic fake responder.
"""

from giantmed_mimic3 import main

if __name__ == "__main__":
    main(mimic_version=4)

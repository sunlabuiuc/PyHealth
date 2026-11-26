"""TaskAug augmentation policy for ECG signals.

Adapted from https://github.com/aniruddhraghu/ecg_aug (Apache License 2.0;
the license is in the LICENSE file of this folder). The code was modified to
use package-relative imports, to run on GPU and to pass PyHealth's
lint rules. The unused ``spec_aug`` function was removed.

Paper:
    Aniruddh Raghu, Divya Shanmugam, Eugene Pomerantsev, John Guttag, and
    Collin M. Stultz. "Data Augmentation for Electrocardiograms."
    CHIL 2022, PMLR 174:282-310.
"""

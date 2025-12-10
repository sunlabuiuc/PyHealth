pyhealth.datasets.NoisyMRCICUMortalityDataset
=============================================

The ICU Mortality dataset used in the *Minimax Risk Classifiers for Mislabeled Data* paper is derived from the
MIT GOSSIS WiDS 2020 Datathon ICU cohort. We use the **preprocessed versions released by the MRC authors**, which
apply missingness filtering, feature selection, median imputation, normalization, and one-hot encoding of
categorical variables. The dataset is provided in two variants:

- ``mortality_alsocat``: includes all processed features, with categorical variables one-hot encoded;
- ``mortality_nocat``: includes only continuous features, with categorical variables removed.

These variants are wrapped into a single PyHealth dataset class to give users convenient access to both feature
configurations for ICU mortality prediction tasks under label noise.

.. autoclass:: pyhealth.datasets.NoisyMRCICUMortalityDataset
    :members:
    :undoc-members:
    :show-inheritance:

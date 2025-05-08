# cxr_adv
Repository for the paper "An Adversarial Approach for the Robust Classification of Pneumonia from Chest Radiographs"

## Basic usage:

Before using this repository, be sure to set up a `./data` directory containing the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) and [MIMIC](https://physionet.org/content/mimic-cxr/1.0.0/) datasets.

### Training a model | Command line interface

To train a model, run `python train.py dataset training` from the command line. The argument 'dataset' specificies which dataset to use for training, and can be either 'MIMIC' or 'CheXpert'. The argument 'training' indicates whether to follow the standard training procedure or to train the adversarial view-invariant model. This argument can be either 'Standard' or 'Adversarial'. So, for example, to train the adversarial model on the CheXpert dataset, run `python train.py CheXpert Adversarial`.

### Testing a model | Command line interface

To test a model, simply run `python test.py model_path training`, where 'model_path' is the path to the saved model you would like to test, and 'training' specifies whether the model was trained as a 'Standard' or 'Adversarial' model. So, for example, to test the adversarial model on the MIMIC dataset, run `python test.py chexpert_adversarial_model.pkl Adversarial`. 

import unittest
from pyhealth.datasets import *
from pyhealth.models import *
from pyhealth.tasks import *

class TestModules(unittest.TestCase):

    test_datasets = [MIMIC3Dataset, MIMIC4Dataset, OMOPDataset, eICUDataset]
    test_tasks = ["drug_rec", "len_of_stay", "mortality", "readmission"]
    test_models = [CNN, RNN, Transformer, RETAIN, MICRON, GAMENet, SafeDrug]
    test_layers = [CNNLayer, RNNLayer, TransformerLayer, RETAINLayer, MICRONLayer, GAMENetLayer, SafeDrugLayer]

    





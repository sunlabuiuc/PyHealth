from pyhealth.tasks import BaseTask
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

class HeartDiseasePrediction(BaseTask):
    """
    This function aims to predict heart problems in patients. We take in a csv from HeartDiseaseDataset
    """
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    def split(self, test_size=0.2, random_state=42):
        """
        
        data training occurs here. we split into a training and a testing subset from the original data
        this is because we do not have multiple datasets to train/test off of

        """
        indices = list(range(len(self.dataset)))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )

        """

        outputs a subset of both the train and test.

        """
        return Subset(self.dataset, train_idx), Subset(self.dataset, test_idx)
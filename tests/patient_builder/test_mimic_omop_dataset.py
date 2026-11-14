from pyhealth.datasets.mimic_omop_dataset import (
    MIMICOMOPDataset,
)


def test_dataset_can_be_imported():
    assert MIMICOMOPDataset is not None


def test_dataset_is_subclass_of_omop():
    from pyhealth.datasets import OMOPDataset

    assert issubclass(
        MIMICOMOPDataset,
        OMOPDataset,
    )
	
def test_load_dataset(sample_omop_path):

    dataset = MIMICOMOPDataset(
        root=sample_omop_path
    )

    assert len(dataset.patients) > 0
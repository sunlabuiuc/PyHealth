class BaseEHRDataset:
    """This class is deprecated and should not be used."""
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn("The BaseEHRDataset class is deprecated and will be removed in a future version.", DeprecationWarning)

class BaseSignalDataset:
    """This class is deprecated and should not be used."""
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn("The BaseSignalDataset class is deprecated and will be removed in a future version.", DeprecationWarning)


class SampleEHRDataset:
    """This class is deprecated and should not be used."""
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn("The SampleEHRDataset class is deprecated and will be removed in a future version.", DeprecationWarning)


class SampleSignalDataset:
    """This class is deprecated and should not be used."""
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn("The SampleSignalDataset class is deprecated and will be removed in a future version.", DeprecationWarning)


from .base_dataset import BaseDataset
from .cardiology import CardiologyDataset
from .covid19_cxr import COVID19CXRDataset
from .ehrshot import EHRShotDataset
from .eicu import eICUDataset
from .isruc import ISRUCDataset
from .medical_transcriptions import MedicalTranscriptionsDataset
from .mimic3 import MIMIC3Dataset
from .mimic4 import MIMIC4CXRDataset, MIMIC4Dataset, MIMIC4EHRDataset, MIMIC4NoteDataset
from .mimicextract import MIMICExtractDataset
from .omop import OMOPDataset
from .sample_dataset import SampleDataset
from .shhs import SHHSDataset
from .sleepedf import SleepEDFDataset
from .splitter import split_by_patient, split_by_sample, split_by_visit
from .tuab import TUABDataset
from .tuev import TUEVDataset
from .utils import collate_fn_dict, collate_fn_dict_with_padding, get_dataloader

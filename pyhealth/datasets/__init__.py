class BaseEHRDataset:
    """This class is deprecated and should not be used."""

    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "The BaseEHRDataset class is deprecated and will be removed in a future version.",
            DeprecationWarning,
        )


class BaseSignalDataset:
    """This class is deprecated and should not be used."""

    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "The BaseSignalDataset class is deprecated and will be removed in a future version.",
            DeprecationWarning,
        )


class SampleEHRDataset:
    """This class is deprecated and should not be used."""

    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "The SampleEHRDataset class is deprecated and will be removed in a future version.",
            DeprecationWarning,
        )


class SampleSignalDataset:
    """This class is deprecated and should not be used."""

    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "The SampleSignalDataset class is deprecated and will be removed in a future version.",
            DeprecationWarning,
        )


from .base_dataset import BaseDataset
from .cardiology import CardiologyDataset
try:
    from .chestxray14 import ChestXray14Dataset
except ImportError:
    pass  # PIL/torchvision unavailable
from .clinvar import ClinVarDataset
from .cosmic import COSMICDataset
try:
    from .covid19_cxr import COVID19CXRDataset
except ImportError:
    pass  # PIL/torchvision unavailable
from .dreamt import DREAMTDataset
from .ehrshot import EHRShotDataset
from .eicu import eICUDataset
from .isruc import ISRUCDataset
from .medical_transcriptions import MedicalTranscriptionsDataset
from .mimic3 import MIMIC3Dataset
from .mimic4 import MIMIC4CXRDataset, MIMIC4Dataset, MIMIC4EHRDataset, MIMIC4NoteDataset
from .mimicextract import MIMICExtractDataset
from .omop import OMOPDataset
from .sample_dataset import SampleBuilder, SampleDataset, create_sample_dataset
from .shhs import SHHSDataset
try:
    from .sleepedf import SleepEDFDataset
except ImportError:
    pass  # mne unavailable
from .bmd_hs import BMDHSDataset
from .support2 import Support2Dataset
from .tcga_prad import TCGAPRADDataset
from .splitter import (
    sample_balanced,
    split_by_patient,
    split_by_patient_conformal,
    split_by_sample,
    split_by_sample_conformal,
    split_by_visit,
    split_by_visit_conformal,
)
try:
    from .tuab import TUABDataset
except ImportError:
    pass  # mne unavailable; TUABDataset not registered
try:
    from .tuev import TUEVDataset
except ImportError:
    pass  # mne unavailable; TUEVDataset not registered
from .utils import (
    collate_fn_dict,
    collate_fn_dict_with_padding,
    get_dataloader,
    load_processors,
    save_processors,
)
from .collate import collate_temporal

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
from .chestxray14 import ChestXray14Dataset
from .clinvar import ClinVarDataset
from .cosmic import COSMICDataset
from .covid19_cxr import COVID19CXRDataset
from .ptbxl import PTBXLDataset as PTBXLDataset
from .physionet2012 import PhysioNet2012Dataset as PhysioNet2012Dataset
from .dreamt import DREAMTDataset
from .ehrshot import EHRShotDataset
from .eicu import eICUDataset
from .isruc import ISRUCDataset
from .medical_transcriptions import MedicalTranscriptionsDataset
from .meds import MEDSDataset as MEDSDataset
from .metabric import MetabricDataset as MetabricDataset
from .mimic3 import MIMIC3Dataset
from .mimic4 import MIMIC4CXRDataset, MIMIC4Dataset, MIMIC4EHRDataset, MIMIC4NoteDataset
from .mimic4 import MIMIC4EDDataset as MIMIC4EDDataset
from .fhir import FHIRDataset, MIMIC4FHIR
from .mimicextract import MIMICExtractDataset
from .omop import OMOPDataset
from .physionet_deid import PhysioNetDeIDDataset
from .sample_dataset import SampleBuilder, SampleDataset, create_sample_dataset
from .shhs import SHHSDataset
from .sleepedf import SleepEDFDataset
from .bmd_hs import BMDHSDataset
from .support2 import Support2Dataset
from .synthea_csv import SyntheaCSVDataset as SyntheaCSVDataset
from .tcga_prad import TCGAPRADDataset
from .splitter import (
    PatientSplit as PatientSplit,
    Split as Split,
    SubjectSplit as SubjectSplit,
    sample_balanced,
    split_by_patient,
    split_by_patient_conformal,
    split_by_patient_conformal_tuh,
    split_by_patient_tuh,
    split_by_sample,
    split_by_sample_conformal,
    split_by_sample_conformal_tuh,
    split_by_sample_tuh,
    split_by_strat_fold as split_by_strat_fold,
    split_by_subject as split_by_subject,
    split_by_visit,
    split_by_visit_conformal,
)
from .eegbci import EEGBCIDataset as EEGBCIDataset  # noqa: E402
from .iiic import IIICDataset as IIICDataset
from .tuab import TUABDataset
from .tuev import TUEVDataset
from .chbmit import CHBMITDataset as CHBMITDataset
from .eesm23 import EESM23Dataset as EESM23Dataset
from .utils import (
    collate_fn_dict,
    collate_fn_dict_with_padding,
    get_dataloader,
    load_processors,
    save_processors,
)
from .collate import collate_temporal

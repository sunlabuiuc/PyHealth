def _removed_1x_class(name: str, replacement: str) -> type:
    """Builds a placeholder for a PyHealth 1.x class removed in 2.0.

    The name stays importable so old imports load, but creating an instance
    (or an instance of a subclass) raises at once with the 2.x replacement,
    instead of returning an object with none of its attributes set.

    Example:
        >>> from pyhealth.datasets import SampleEHRDataset
        >>> try:
        ...     SampleEHRDataset()
        ... except NotImplementedError as err:
        ...     print(str(err).split(";")[0])
        SampleEHRDataset was removed in PyHealth 2.0. Use SampleDataset \
(returned by BaseDataset.set_task) instead
    """

    def __init__(self, *args, **kwargs):
        docs = "https://pyhealth.readthedocs.io/en/latest/api/datasets.html"
        cls = type(self).__name__
        if cls != name:
            # An in-tree or user dataset still built on the 1.x base class
            # (e.g. CardiologyDataset, ISRUCDataset, SHHSDataset,
            # MIMICExtractDataset). It never worked on 2.x: the stub set no
            # attributes and provided no set_task.
            raise NotImplementedError(
                f"{cls} is built on {name}, which was removed in PyHealth "
                f"2.0, so {cls} does not work yet. It needs porting to "
                f"{replacement}; see {docs}"
            )
        raise NotImplementedError(
            f"{name} was removed in PyHealth 2.0. Use {replacement} instead; "
            f"see {docs}"
        )

    return type(
        name,
        (),
        {
            "__init__": __init__,
            "__doc__": f"Removed in PyHealth 2.0; use {replacement}.",
        },
    )


BaseEHRDataset = _removed_1x_class("BaseEHRDataset", "BaseDataset")
BaseSignalDataset = _removed_1x_class("BaseSignalDataset", "BaseDataset")
SampleEHRDataset = _removed_1x_class(
    "SampleEHRDataset", "SampleDataset (returned by BaseDataset.set_task)"
)
SampleSignalDataset = _removed_1x_class(
    "SampleSignalDataset", "SampleDataset (returned by BaseDataset.set_task)"
)


from .base_dataset import BaseDataset
from .imaging.patches import PatchDataset as PatchDataset
from .imaging.stats import (
    compute_foreground_intensity_stats as compute_foreground_intensity_stats,
)
from .cardiology import CardiologyDataset
from .chaos import CHAOSDataset as CHAOSDataset
from .chest_imagenome import ChestImaGenomeDataset as ChestImaGenomeDataset
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
from .msd import (
    MSDHippocampusDataset as MSDHippocampusDataset,
    MSDProstateDataset as MSDProstateDataset,
    MSDSegmentationDataset as MSDSegmentationDataset,
    MSDSpleenDataset as MSDSpleenDataset,
)
from .meds import MEDSDataset as MEDSDataset
from .metabric import MetabricDataset as MetabricDataset
from .mimic3 import MIMIC3Dataset
from .mimic4 import MIMIC4CXRDataset, MIMIC4Dataset, MIMIC4EHRDataset, MIMIC4NoteDataset
from .mimic4 import MIMIC4EDDataset as MIMIC4EDDataset
from .fhir import FHIRDataset, MIMIC4FHIR
from .mimicextract import MIMICExtractDataset
from .omop import OMOPDataset
from .physionet_deid import PhysioNetDeIDDataset
from .rsna_pneumonia import RSNAPneumoniaDataset as RSNAPneumoniaDataset
from .sample_dataset import SampleBuilder, SampleDataset, create_sample_dataset
from .shhs import SHHSDataset
from .sleepedf import SleepEDFDataset
from .bmd_hs import BMDHSDataset
from .support2 import Support2Dataset
from .synthea_csv import SyntheaCSVDataset as SyntheaCSVDataset
from .tcga_prad import TCGAPRADDataset
from .vsac import VSACDataset as VSACDataset
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
from .gold_test_split import GoldTestSplit as GoldTestSplit
from .mimic4_imagenome import MIMIC4ImaGenomeDataset as MIMIC4ImaGenomeDataset
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

from .gbsg import (
    GBSGDataset as GBSGDataset,
    prepare_gbsg_csv as prepare_gbsg_csv,
    gbsg_paper_time_transform as gbsg_paper_time_transform,
    gbsg_released_split_ids as gbsg_released_split_ids,
)
from .support2 import (
    support2_paper_time_transform as support2_paper_time_transform,
    support2_released_split_ids as support2_released_split_ids,
)

from .base_task import BaseTask
from .task_card import (
    DIRECTIONS as DIRECTIONS,
    FeatureInfo as FeatureInfo,
    TaskCard as TaskCard,
)
from .competing_risks import CompetingRisksPreprocessor as CompetingRisksPreprocessor
from .competing_risks_pbc import CompetingRisksPBC as CompetingRisksPBC
from .benchmark_ehrshot import BenchmarkEHRShot
from .cancer_survival import CancerMutationBurden, CancerSurvivalPrediction
from .bmd_hs_disease_classification import BMDHSDiseaseClassification
from .cardiology_detect import (
    CardiologyDetection as CardiologyDetection,
    cardiology_isAD_fn,
    cardiology_isAR_fn,
    cardiology_isBBBFB_fn,
    cardiology_isCD_fn,
    cardiology_isWA_fn,
)
from .chestxray14_binary_classification import ChestXray14BinaryClassification
from .isic2019_classification import (
    ISIC2019Classification as ISIC2019Classification,
)
from .chestxray14_multilabel_classification import ChestXray14MultilabelClassification
from .ham10000_classification import (
    HAM10000Classification as HAM10000Classification,
)
from .chaos_segmentation import (
    CHAOSAbdominalSegmentationCTMRI as CHAOSAbdominalSegmentationCTMRI,
    CHAOSAbdominalSegmentationMRIonly as CHAOSAbdominalSegmentationMRIonly,
    CHAOSLiverSegmentationCTMRI as CHAOSLiverSegmentationCTMRI,
    CHAOSLiverSegmentationCTonly as CHAOSLiverSegmentationCTonly,
    CHAOSLiverSegmentationMRIonly as CHAOSLiverSegmentationMRIonly,
)
from .brain_tumor_mri_classification import (
    BrainTumorMRIClassification as BrainTumorMRIClassification,
)
from .covid19_cxr_classification import COVID19CXRClassification
from .covid19_cxr_generation import (
    COVID19CXRImageGeneration as COVID19CXRImageGeneration,
)
from .ptbxl_multilabel_classification import PTBXLMultilabelClassification as PTBXLMultilabelClassification
from .ptbxl_heartlang import PTBXLHeartLangClassification as PTBXLHeartLangClassification
from .ptbxl_heartbeat import PTBXLHeartbeatMI as PTBXLHeartbeatMI
from .chapman_shaoxing_classification import (
    ChapmanShaoxingSNOMEDClassification as ChapmanShaoxingSNOMEDClassification,
    ChapmanShaoxingSuperclassClassification as ChapmanShaoxingSuperclassClassification,
)
from .physionet2012_mortality import (
    PhysioNet2012Mortality as PhysioNet2012Mortality,
    physionet2012_mortality_statistics as physionet2012_mortality_statistics,
    split_by_record_ids as split_by_record_ids,
)
from .physionet2012_forecasting import (
    PhysioNet2012Forecasting as PhysioNet2012Forecasting,
    physionet2012_benchmark_split as physionet2012_benchmark_split,
    physionet2012_statistics as physionet2012_statistics,
)
from .deid_ner import DeIDNERTask
from .dka import DKAPredictionMIMIC4, T1DDKAPredictionMIMIC4
# New exports must use the redundant `X as X` form: this module has no
# __all__, and the PR lint gate flags F401 on newly added import lines.
from .disease_progression_mimic4 import (
    DiseaseProgressionMIMIC4 as DiseaseProgressionMIMIC4,
)
from .delphi_trajectory import (
    DelphiTrajectoryMEDS as DelphiTrajectoryMEDS,
    DelphiTrajectoryMIMIC4 as DelphiTrajectoryMIMIC4,
    DelphiTrajectoryUKB as DelphiTrajectoryUKB,
)
from .drug_recommendation import (
    DrugRecommendationEICU,
    DrugRecommendationMIMIC3,
    DrugRecommendationMIMIC4,
    DrugRecommendationOMOP as DrugRecommendationOMOP,
    drug_recommendation_mimic3_fn,
    drug_recommendation_mimic4_fn,
    drug_recommendation_omop_fn,
)
from .ed_disposition_mimic4 import (
    ED_NOTE_DOMAINS as ED_NOTE_DOMAINS,
    ED_OUTCOMES as ED_OUTCOMES,
    EDDispositionMIMIC4 as EDDispositionMIMIC4,
    EDOutcomesMIMIC4 as EDOutcomesMIMIC4,
)
from .ehr_cxr_mimic4 import (
    EHRCXRMortalityMIMIC4 as EHRCXRMortalityMIMIC4,
    EHRCXRPhenotypingMIMIC4 as EHRCXRPhenotypingMIMIC4,
)
from .maestro_mortality_mimic4 import (
    MAESTROMortalityMIMIC4 as MAESTROMortalityMIMIC4,
)
from .ehr_foundation_model import (
    EHRFoundationModelTask as EHRFoundationModelTask,
    EventSource as EventSource,
)
from .ehr_text_serialization import (
    CodeDescriber as CodeDescriber,
    EHRTextSerializationTask as EHRTextSerializationTask,
    serialize_ehr as serialize_ehr,
)
from .in_hospital_mortality_meds import (
    InHospitalMortalityMEDS as InHospitalMortalityMEDS,
)
from .in_hospital_mortality_mimic4 import InHospitalMortalityMIMIC4
from .kits23_classification import (
    KiTS23MalignancyClassification as KiTS23MalignancyClassification,
    KiTS23MalignancySliceClassification as KiTS23MalignancySliceClassification,
)
from .kits23_segmentation import (
    KiTS23Segmentation as KiTS23Segmentation,
)
from .lab_nowcasting import (
    LabNowcasting as LabNowcasting,
    LabNowcastingMIMIC3 as LabNowcastingMIMIC3,
    LabNowcastingMIMIC4 as LabNowcastingMIMIC4,
    lab_flag as lab_flag,
)
from .lndb_classification import (
    LNDbFindingClassification as LNDbFindingClassification,
)
from .lndb_texture import (
    LNDbTextureClassification as LNDbTextureClassification,
)
from .length_of_stay_prediction import (
    LengthOfStayPredictioneICU,
    LengthOfStayPredictionMIMIC3,
    LengthOfStayPredictionMIMIC4,
    LengthOfStayPredictionOMOP,
)
from .length_of_stay_stagenet_mimic4 import LengthOfStayStageNetMIMIC4
from .medical_coding import MIMIC3ICD9Coding
from .medmod_decompensation_mimic4 import (
    MedModDecompensationMIMIC4 as MedModDecompensationMIMIC4,
)
from .meds_outcome_tasks import (
    HospitalReadmissionMEDS as HospitalReadmissionMEDS,
    ICUAdmissionMEDS as ICUAdmissionMEDS,
    ProlongedLengthOfStayMEDS as ProlongedLengthOfStayMEDS,
)
from .ethos_tasks import (
    EthosOutcomeMEDS as EthosOutcomeMEDS,
    EthosRolloutRule as EthosRolloutRule,
    EthosTimelineMEDS as EthosTimelineMEDS,
)
from .msd_segmentation import MSDSegmentationTask as MSDSegmentationTask
from .next_visit_diagnosis_prediction import (
    NextVisitDiagnosisPredictionMIMIC3 as NextVisitDiagnosisPredictionMIMIC3,
    NextVisitDiagnosisPredictionMIMIC4 as NextVisitDiagnosisPredictionMIMIC4,
)
from .medical_transcriptions_classification import MedicalTranscriptionsClassification
from .mortality_prediction import (
    MortalityPredictionEICU,
    MortalityPredictionEICU2,
    MortalityPredictionMIMIC3,
    MortalityPredictionMIMIC4,
    MortalityPredictionOMOP,
    MultimodalMortalityPredictionMIMIC3,
    MultimodalMortalityPredictionMIMIC4,
)
from .survival_metabric import (
    MetabricSurvivalPrediction as MetabricSurvivalPrediction,
)
from .survival_mimic4 import InHospitalSurvivalMIMIC4 as InHospitalSurvivalMIMIC4
from .acuity_mimic4 import ICUAcuityMIMIC4 as ICUAcuityMIMIC4
from .survival_preprocess_support2 import SurvivalPreprocessSupport2
from .survival_time_to_event import (
    SurvivalTimeToEventGBSG as SurvivalTimeToEventGBSG,
)
from .survival_time_to_event import (
    SurvivalTimeToEventSupport2 as SurvivalTimeToEventSupport2,
)
from .mortality_prediction_stagenet_mimic4 import (
    MortalityPredictionStageNetMIMIC4,
)
from .dearllm_mortality import (
    DearLLMMortalityPredictionMIMIC3 as DearLLMMortalityPredictionMIMIC3,
    DearLLMMortalityPredictionMIMIC4 as DearLLMMortalityPredictionMIMIC4,
)
from .generate_ehr import (
    EHRCodeSetGenerationMIMIC3 as EHRCodeSetGenerationMIMIC3,
    EHRCodeSetGenerationMIMIC4 as EHRCodeSetGenerationMIMIC4,
    EHRGenerationMIMIC3,
    EHRGenerationMIMIC4,
    EHRSequenceGenerationMIMIC3 as EHRSequenceGenerationMIMIC3,
    EHRSequenceGenerationMIMIC4 as EHRSequenceGenerationMIMIC4,
    decode_dataset,
    to_evaluation_dataframe,
)
from .pad_ufes_20_classification import (
    PADUFES20Classification as PADUFES20Classification,
)
from .patient_linkage import patient_linkage_mimic3_fn
from .jichi_davis_grading import JichiDavisGrading as JichiDavisGrading
from .rsna_pneumonia_classification import (
    RSNAPneumoniaClassification as RSNAPneumoniaClassification,
)
from .bcss_tile_classification import (
    BCSSTileClassification as BCSSTileClassification,
)
from .bcss_tissue_segmentation import (
    BCSSTissueSegmentation as BCSSTissueSegmentation,
)
from .isic2020_melanoma_classification import (
    ISIC2020MelanomaClassification as ISIC2020MelanomaClassification,
)
from .cbis_ddsm_abnormality_classification import (
    CBISDDSMAbnormalityClassification as CBISDDSMAbnormalityClassification,
)
from .cbis_ddsm_classification import (
    CBISDDSMBIRADSClassification as CBISDDSMBIRADSClassification,
    CBISDDSMPathologyClassification as CBISDDSMPathologyClassification,
)
from .cbis_ddsm_density_classification import (
    CBISDDSMDensityClassification as CBISDDSMDensityClassification,
)
from .readmission_prediction import (
    ReadmissionPredictionEICU,
    ReadmissionPredictionMIMIC3,
    ReadmissionPredictionMIMIC4,
    ReadmissionPredictionOMOP,
)
from .sleep_staging import (
    sleep_staging_isruc_fn,
    sleep_staging_shhs_fn,
    sleep_staging_sleepedf_fn,
)
from .abide_classification import (
    ABIDEAutismClassification as ABIDEAutismClassification,
)
from .sleep_staging_v2 import SleepStagingSequenceSleepEDF as SleepStagingSequenceSleepEDF
from .covid_blues_classification import (
    COVIDBLUESFrameClassification as COVIDBLUESFrameClassification,
)
from .covid_blues_severity import (
    COVIDBLUESSeverityClassification as COVIDBLUESSeverityClassification,
)
from .sleep_staging_v2 import SleepStagingSleepEDF
from .sleep_staging_nsrr import SleepStagingCardioRespiratory  # noqa: F401
from .sleep_staging_isruc import SleepStagingISRUC as SleepStagingISRUC
from .sleep_staging_isruc import (
    SleepStagingSequenceISRUC as SleepStagingSequenceISRUC,
)
from .sleep_staging_hmc import SleepStagingHMC as SleepStagingHMC
from .temple_university_EEG_tasks import (
    EEGEventsTUEV,
    EEGAbnormalTUAB
)
from .seizure_detection_helsinki import (
    SeizureDetectionHelsinki as SeizureDetectionHelsinki,
)
from .seizure_detection_chbmit import (
    SeizureDetectionCHBMIT as SeizureDetectionCHBMIT,
)
from .sleep_staging_ear_eeg import (
    SleepStagingEarEEG as SleepStagingEarEEG,
)
from .dementia_classification_ahepa import (
    DementiaClassificationAHEPA as DementiaClassificationAHEPA,
)
from .soz_localization_hup import (
    SOZLocalizationHUP as SOZLocalizationHUP,
)
from .eegbci import (
    EEGBCIPatternDiscovery as EEGBCIPatternDiscovery,
    EEGMotorImageryEEGBCI as EEGMotorImageryEEGBCI,
)
from .bciiv2a_left_right_imagery import (
    BCIIV2aLeftRightImagery as BCIIV2aLeftRightImagery,
)
from .bciiv2a_motor_imagery import (
    BCIIV2aMotorImageryClassification as BCIIV2aMotorImageryClassification,
)
from .iiic_seizure_classification import (
    EEGSeizureIIIC as EEGSeizureIIIC,
    IIICSeizureClassification as IIICSeizureClassification,
)
from .variant_classification import (
    MutationPathogenicityPrediction,
    VariantClassificationClinVar,
)
from .patient_linkage_mimic3 import PatientLinkageMIMIC3Task
from .knowrare_mimic4 import (
    KnowRareICUMortalityMIMIC4 as KnowRareICUMortalityMIMIC4,
)
from .value_set_completion import (
    ValueSetCompletion as ValueSetCompletion,
    build_retrieval_index as build_retrieval_index,
)
from .value_set_pair_classification import (
    ValueSetPairClassification as ValueSetPairClassification,
    build_hybrid_retrieval_index as build_hybrid_retrieval_index,
)
from .yaib_mimic3 import (
    YAIBAKIMIMIC3 as YAIBAKIMIMIC3,
    YAIBKidneyFunctionMIMIC3 as YAIBKidneyFunctionMIMIC3,
    YAIBLengthOfStayMIMIC3 as YAIBLengthOfStayMIMIC3,
    YAIBMortalityMIMIC3 as YAIBMortalityMIMIC3,
    YAIBSepsisMIMIC3 as YAIBSepsisMIMIC3,
)
from .yaib_mimic4 import (
    YAIBAKIMIMIC4 as YAIBAKIMIMIC4,
    YAIBKidneyFunctionMIMIC4 as YAIBKidneyFunctionMIMIC4,
    YAIBLengthOfStayMIMIC4 as YAIBLengthOfStayMIMIC4,
    YAIBMortalityMIMIC4 as YAIBMortalityMIMIC4,
    YAIBSepsisMIMIC4 as YAIBSepsisMIMIC4,
)
from .yaib_eicu import (
    YAIBAKIEICU as YAIBAKIEICU,
    YAIBKidneyFunctionEICU as YAIBKidneyFunctionEICU,
    YAIBLengthOfStayEICU as YAIBLengthOfStayEICU,
    YAIBMortalityEICU as YAIBMortalityEICU,
    YAIBSepsisEICU as YAIBSepsisEICU,
)
from .tcga_survival_prediction import TCGASurvivalPrediction as TCGASurvivalPrediction

def __getattr__(name: str):
    if name == "MPFClinicalPredictionTask":
        from .mpf_clinical_prediction import MPFClinicalPredictionTask

        return MPFClinicalPredictionTask
    if name == "PTBXLSuperclassClassification":
        from .ptbxl import PTBXLSuperclassClassification

        return PTBXLSuperclassClassification
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from .tpc_eicu import TPCLengthOfStayEICU as TPCLengthOfStayEICU

from .base_task import BaseTask
from .benchmark_ehrshot import BenchmarkEHRShot
from .cancer_survival import CancerMutationBurden, CancerSurvivalPrediction
from .bmd_hs_disease_classification import BMDHSDiseaseClassification
from .cardiology_detect import (
    cardiology_isAD_fn,
    cardiology_isAR_fn,
    cardiology_isBBBFB_fn,
    cardiology_isCD_fn,
    cardiology_isWA_fn,
)
from .chestxray14_binary_classification import ChestXray14BinaryClassification
from .chestxray14_multilabel_classification import ChestXray14MultilabelClassification
from .covid19_cxr_classification import COVID19CXRClassification
from .drug_recommendation import (
    DrugRecommendationMIMIC3,
    DrugRecommendationMIMIC4,
    drug_recommendation_eicu_fn,
    drug_recommendation_mimic3_fn,
    drug_recommendation_mimic4_fn,
    drug_recommendation_omop_fn,
)
from .EEG_abnormal import EEG_isAbnormal_fn
from .EEG_events import EEG_events_fn
from .in_hospital_mortality_mimic4 import InHospitalMortalityMIMIC4
from .length_of_stay_prediction import (
    LengthOfStayPredictioneICU,
    LengthOfStayPredictionMIMIC3,
    LengthOfStayPredictionMIMIC4,
    LengthOfStayPredictionOMOP,
    length_of_stay_prediction_eicu_fn,
    length_of_stay_prediction_mimic3_fn,
    length_of_stay_prediction_mimic4_fn,
    length_of_stay_prediction_omop_fn,
)
from .medical_coding import MIMIC3ICD9Coding
from .medical_transcriptions_classification import MedicalTranscriptionsClassification
from .mortality_prediction import (
    MortalityPredictionEICU,
    MortalityPredictionEICU2,
    MortalityPredictionMIMIC3,
    MortalityPredictionMIMIC4,
    MortalityPredictionOMOP,
)
from .survival_preprocess_support2 import SurvivalPreprocessSupport2
from .mortality_prediction_stagenet_mimic4 import (
    MortalityPredictionStageNetMIMIC4,
)
from .patient_linkage import patient_linkage_mimic3_fn
from .readmission_30days_mimic4 import Readmission30DaysMIMIC4
from .readmission_prediction import (
    readmission_prediction_eicu_fn,
    readmission_prediction_eicu_fn2,
    readmission_prediction_mimic3_fn,
    readmission_prediction_mimic4_fn,
    readmission_prediction_omop_fn,
)
from .sleep_staging import (
    sleep_staging_isruc_fn,
    sleep_staging_shhs_fn,
    sleep_staging_sleepedf_fn,
)
from .sleep_staging_v2 import SleepStagingSleepEDF
from .temple_university_EEG_tasks import EEG_events_fn, EEG_isAbnormal_fn
from .variant_classification import (
    MutationPathogenicityPrediction,
    VariantClassificationClinVar,
)

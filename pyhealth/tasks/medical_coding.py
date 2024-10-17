import logging
from tqdm import tqdm
from pyhealth.data.data_v2 import Patient, Event
from pyhealth.tasks.task_template import TaskTemplate
from pyhealth.datasets.sample_dataset_v2 import SampleDataset
from pyhealth.datasets.base_dataset_v2 import BaseDataset

logger = logging.getLogger(__name__)
class MIMIC3ICD9Coding(TaskTemplate):
    def __init__(self, dataset: BaseDataset = None, cache_dir: str = "./cache", refresh_cache: bool = False):
        super().__init__(
            task_name="mimic3_icd9_coding",
            input_schema={"text": "str"},
            output_schema={"icd_codes": "List[str]"},
            dataset=dataset,
            cache_dir=cache_dir,
            refresh_cache=refresh_cache
        )

    def process(self):
        logger.debug(f"Setting task for {self.dataset.dataset_name} base dataset...")
        samples = []
        for patient_id, patient in tqdm(
            self.dataset.patients.items(), desc=f"Generating samples for {self.task_name}"
        ):
            samples.extend(self.sample(patient))
        return samples

    def sample(self, patient: Patient):
        text = ""
        icd_codes = set()
        for event in patient.events:
            if event.table == "NOTEEVENTS":
                text += event.code
            if event.table == "DIAGNOSES_ICD":
                icd_codes.add(event.code)
            if event.table == "PROCEDURES_ICD":
                icd_codes.add(event.code)
        if text == "" or len(icd_codes) < 1:
            return []
        return [{"text": text, "icd_codes": list(icd_codes)}]


class MIMIC4ICD9Coding(TaskTemplate):
    def __init__(self, dataset: BaseDataset = None, cache_dir: str = "./cache", refresh_cache: bool = False):
        super().__init__(
            task_name="mimic4_icd9_coding",
            input_schema={"text": "str"},
            output_schema={"icd_codes": "List[str]"},
            dataset=dataset,
            cache_dir=cache_dir,
            refresh_cache=refresh_cache
        )
    
    def sample(self, patient: Patient):
        text = ""
        icd_codes = set()
        for event in patient.events:
            if event.table == "discharge":
                text += event.code
            if event.vocabulary == "ICD9CM":
                if event.table == "diagnoses_icd":
                    icd_codes.add(event.code)
                if event.table == "procedures_icd":
                    icd_codes.add(event.code)
        if text == "" or len(icd_codes) < 1:
            return []
        return [{"text": text, "icd_codes": list(icd_codes)}]   
    
    def process(self):
        # load from raw data
        logger.debug(f"Setting task for {self.dataset.dataset_name} base dataset...")

        samples = []
        for patient_id, patient in tqdm(
            self.dataset.patients.items(), desc=f"Generating samples for {self.task_name}"
        ):
            samples.extend(self.sample(patient))
        return samples

class MIMIC4ICD10Coding(TaskTemplate):
    def __init__(self, dataset: BaseDataset = None, cache_dir: str = "./cache", refresh_cache: bool = False):
        super().__init__(
            task_name="mimic4_icd10_coding",
            input_schema={"text": "str"},
            output_schema={"icd_codes": "List[str]"},
            dataset=dataset,
            cache_dir=cache_dir,
            refresh_cache=refresh_cache
        )

    
    def sample(self, patient: Patient):
        text = ""
        icd_codes = set()
        for event in patient.events:
            if event.table == "discharge":
                text += event.code
            if event.vocabulary == "ICD10CM":
                if event.table == "diagnoses_icd":
                    icd_codes.add(event.code)
                if event.table == "procedures_icd":
                    icd_codes.add(event.code)
        if text == "" or len(icd_codes) < 1:
            return []
        return [{"text": text, "icd_codes": list(icd_codes)}]
    
    def process(self):
         # load from raw data
        logger.debug(f"Setting task for {self.dataset.dataset_name} base dataset...")

        samples = []
        for patient_id, patient in tqdm(
            self.dataset.patients.items(), desc=f"Generating samples for {self.task_name}"
        ):
            samples.extend(self.sample(patient))
        return samples

    # def to_torch_dataset(self):
       
       
        # sample_dataset = SampleDataset(
        #     samples,
        #     input_schema=self.input_schema,
        #     output_schema=self.output_schema,
        #     dataset_name=self.dataset.dataset_name,
        #     task_name=self.task_name,
        # )
        # return sample_dataset

# def icd9_coding_mimic3_fn(patient : Patient):
#     text = ""
#     icd_codes = []
#     for event in patient.events:
#         if event.table == "NOTEEVENTS":
#             text += event.code
#         if event.table == "DIAGNOSES_ICD":
#             icd_codes.append(event.code)
#         if event.table == "PROCEDURES_ICD":
#             icd_codes.append(event.code)
#     if text == "" or len(icd_codes) < 1:
#         return []
#     return [{"text": text, "icd_codes": icd_codes}]


# def icd9_coding_mimic4_fn(patient : Patient):
#     text = ""
#     icd_codes = [] # all notes are essentially concatenated into 1 with every code in the patient's events
#     for event in patient.events:
#         if event.table == "discharge":
#             text += event.code
#         if event.vocabulary == "ICD9CM":
#             if event.table == "diagnoses_icd":
#                 icd_codes.append(event.code)
#             if event.table == "procedures_icd":
#                 icd_codes.append(event.code)
#     # we need to probably add some check that the text is not empty, otherwise return with []
#     if text == "" or len(icd_codes) < 1:
#         return []
#     return [{"text": text, "icd_codes": icd_codes}]

# def icd10_coding_fn(patient : Patient): 
#     text = ""
#     icd_codes = []
#     for event in patient.events:
#         if event.table == "discharge":
#             text += event.code
#         if event.vocabulary == "ICD10CM":
#             if event.table == "diagnoses_icd":
#                 icd_codes.append(event.code)
#             if event.table == "procedures_icd":
#                 icd_codes.append(event.code)
#     if text == "" or len(icd_codes) < 1:
#         return []
#     return [{"text": text, "icd_codes": icd_codes}]   


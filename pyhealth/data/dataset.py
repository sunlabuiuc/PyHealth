import numpy as np
from torch.utils.data import Dataset

from pyhealth.data import Visit, Patient


class BaseDataset:
    def __init__(self, patients, dataset_info=None):
        self.patients = patients
        self.dataset_info = dataset_info

    def __len__(self):
        return len(self.patients)

    def __repr__(self):
        return f"{self.dataset_info['dataset']} dataset"


class TaskDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.preprocess()
        self.all_vocabs = self.get_all_vocabs()

    def preprocess(self):
        raise NotImplementedError

    def get_all_vocabs(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class DrugRecommendationDataset(TaskDataset):

    def __init__(self, base_dataset):
        super(DrugRecommendationDataset, self).__init__(base_dataset)

    def preprocess(self):
        from MedCode import CodeMapping
        tool = CodeMapping('RxNorm', 'ATC4')
        tool.load()

        def get_atc3(x):
            # one rxnorm maps to one or more ATC3
            result = []
            for rxnorm in x:
                if rxnorm in tool.RxNorm_to_ATC4:
                    result += tool.RxNorm_to_ATC4[rxnorm]
            result = np.unique([item[:-1] for item in result]).tolist()
            return result

        def remove_nan_from_list(lst):
            if type(lst) is not list:
                return []
            return [i for i in lst if not i != i]

        processed_patients = []
        for patient in self.base_dataset.patients:
            processed_visits = []
            for visit in patient.visits:
                conditions = list(set(remove_nan_from_list(visit.conditions)))
                procedures = list(set(remove_nan_from_list(visit.procedures)))
                drugs = get_atc3(["{:011}".format(int(med)) for med in set(remove_nan_from_list(visit.drugs))])
                # exclude: visits without condition, procedure, or drug code
                if len(conditions) * len(procedures) * len(drugs) == 0:
                    continue
                processed_visit = Visit(visit_id=visit.visit_id,
                                        patient_id=visit.patient_id,
                                        conditions=conditions,
                                        procedures=procedures,
                                        drugs=drugs)
                processed_visits.append(processed_visit)
            # exclude: patients with less than 2 visit
            if len(processed_visits) < 2:
                continue
            processed_patient = Patient(patient_id=patient.patient_id,
                                        visits=processed_visits,
                                        patient_info=patient.patient_info)
            processed_patients.append(processed_patient)

        self.processed_patients = processed_patients

    def get_all_vocabs(self):
        conditions = []
        procedures = []
        drugs = []
        for patient in self.processed_patients:
            for visit in patient.visits:
                conditions.extend(visit.conditions)
                procedures.extend(visit.procedures)
                drugs.extend(visit.drugs)
        conditions = list(set(conditions))
        procedures = list(set(procedures))
        drugs = list(set(drugs))
        return {'conditions': conditions, 'procedures': procedures, 'drugs': drugs}

    def __len__(self):
        return len(self.processed_patients)

    def __getitem__(self, index):
        data = []
        patient = self.processed_patients[index]
        for visit in patient.visits:
            data.append([visit.conditions, visit.procedures, visit.drugs])
        return data


class MortalityPredictionDataset(TaskDataset):
    pass


class LengthOfStayEstimationDataset(TaskDataset):
    pass


if __name__ == "__main__":
    from pyhealth.datasets.mimic3 import MIMIC3BaseDataset

    base_dataset = MIMIC3BaseDataset(root="/srv/local/data/physionet.org/files/mimiciii/1.4")
    task_taskset = DrugRecommendationDataset(base_dataset)
    print(task_taskset)
    print(len(task_taskset))
    print(task_taskset[0])
    print(task_taskset.all_vocabs.keys())
    pass

import os
from collections import defaultdict
from pathlib import Path
from urllib import request

import numpy as np
import pandas as pd

from pyhealth.data import TaskDataset
from pyhealth.tasks.utils import get_code_from_list_of_event


class DrugRecommendationDataset(TaskDataset):
    """Task specific dataset for drug recommendation.

    Drug recommendation aims at recommending a set of drugs given the patient health history (e.g., conditions
    and procedures).

    =====================================================================================
    Data structure:

        TaskDataset.samples: List[dict[str, Any]]
            - a list of samples, each sample is a dict with patient_id, visit_id, and
             other task-specific attributes as key
    =====================================================================================
    """

    def __init__(self, base_dataset):
        """Initialize the task dataset.

        This function should initialize any necessary tools for the task (e.g., code mapping tools),
        and then call the parent class's __init__ function to process the dataset.

        Args:
            base_dataset: a BaseDataset object
        """

        # TODO: might want to remove dependency on MedCode later
        from MedCode import CodeMapping
        self.tool = CodeMapping("NDC11", "ATC4")
        self.tool.load()

        super(DrugRecommendationDataset, self).__init__(task_name="drug_recommendation", base_dataset=base_dataset)

    def process_single_patient(self, patient, visits):
        """Process a single patient.

        It takes a patient and a dict of visits as input,
        and returns a list of samples, which are dicts with patient_id, visit_id, and other task-specific attributes
        as key. The samples will be concatenated to form the final samples of the task dataset.

        Note that a patient may be converted to multiple samples, e.g., a patient with three visits may be converted
        to three samples ([visit 1], [visit 1, visit 2], [visit 1, visit 2, visit 3]). Patients can also be excluded
        from the task dataset by returning an empty list.

        Args:
            patient: a Patient object
            visits: a dict of visits with visit_id as key

        Returns:
            samples: a list of samples, each sample is a dict with patient_id, visit_id, and other task-specific
             attributes as key
        """

        # TODO: should be more flexible with what inputs to use (not only conditions and procedures)
        # TODO: should be more flexible with what coding system to use (not only ATC3)

        def get_atc3(x):
            # one rxnorm maps to one or more ATC3
            result = []
            for rxnorm in x:
                if rxnorm in self.tool.NDC11_to_ATC4:
                    result += self.tool.NDC11_to_ATC4[rxnorm]
            result = list(dict.fromkeys([item[:-1] for item in result]))
            return result

        samples = []
        for visit_id in patient.visit_ids:
            visit = visits[visit_id]
            conditions = get_code_from_list_of_event(visit.conditions)
            procedures = get_code_from_list_of_event(visit.procedures)
            drugs = get_code_from_list_of_event(visit.drugs)
            drugs = get_atc3(drugs)
            # exclude: visits without condition, procedure, or drug code
            if (len(conditions) + len(procedures)) * len(drugs) == 0:
                continue
            # TODO: should also exclude visit with age < 18
            samples.append({"visit_id": visit_id,
                            "patient_id": patient.patient_id,
                            "conditions": conditions,
                            "procedures": procedures,
                            "drugs": drugs})
        # exclude: patients with less than 2 visit
        if len(samples) < 2:
            return []

        # add history
        samples[0]["conditions"] = [samples[0]["conditions"]]
        samples[0]["procedures"] = [samples[0]["procedures"]]
        samples[0]["drugs"] = [samples[0]["drugs"]]
        for i in range(1, len(samples)):
            samples[i]["conditions"] = samples[i - 1]["conditions"] + [samples[i]["conditions"]]
            samples[i]["procedures"] = samples[i - 1]["procedures"] + [samples[i]["procedures"]]
            samples[i]["drugs"] = samples[i - 1]["drugs"] + [samples[i]["drugs"]]
        return samples

    # TODO: examine this function
    def get_ddi_matrix(self):
        """get drug-drug interaction (DDI)"""
        cid2atc_dic = defaultdict(set)
        med_voc_size = self.voc_size[2]

        vocab_to_index = self.tokenizers[2].vocabulary.word2idx

        # load cid2atc
        if not os.path.exists(
                os.path.join(str(Path.home()), ".cache/pyhealth/cid_to_ATC6.csv")
        ):
            cid_to_ATC6 = request.urlopen(
                "https://drive.google.com/uc?id=1CVfa91nDu3S_NTxnn5GT93o-UfZGyewI"
            ).readlines()
            with open(
                    os.path.join(str(Path.home()), ".cache/pyhealth/cid_to_ATC6.csv"), "w"
            ) as outfile:
                for line in cid_to_ATC6:
                    print(str(line[:-1]), file=outfile)
        else:
            cid_to_ATC6 = open(
                os.path.join(str(Path.home()), ".cache/pyhealth/cid_to_ATC6.csv"), "r"
            ).readlines()

        # map cid to atc
        for line in cid_to_ATC6:
            line_ls = str(line[:-1]).split(",")
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if atc[:4] in vocab_to_index:
                    cid2atc_dic[cid[2:]].add(atc[:4])

        # ddi on (cid, cid)
        if not os.path.exists(
                os.path.join(str(Path.home()), ".cache/pyhealth/drug-DDI-TOP40.csv")
        ):
            ddi_df = pd.read_csv(
                request.urlopen(
                    "https://drive.google.com/uc?id=1R88OIhn-DbOYmtmVYICmjBSOIsEljJMh"
                )
            )
            ddi_df.to_csv(
                os.path.join(str(Path.home()), ".cache/pyhealth/drug-DDI-TOP40.csv"),
                index=False,
            )
        else:
            ddi_df = pd.read_csv(
                os.path.join(str(Path.home()), ".cache/pyhealth/drug-DDI-TOP40.csv")
            )

        # map to ddi on (atc, atc)
        ddi_adj = np.zeros((med_voc_size, med_voc_size))
        for index, row in ddi_df.iterrows():
            # ddi
            cid1 = row["STITCH 1"]
            cid2 = row["STITCH 2"]

            # cid -> atc_level3
            for atc_i in cid2atc_dic[cid1]:
                for atc_j in cid2atc_dic[cid2]:
                    ddi_adj[
                        vocab_to_index.get(atc_i, 0), vocab_to_index.get(atc_j, 0)
                    ] = 1
                    ddi_adj[
                        vocab_to_index.get(atc_j, 0), vocab_to_index.get(atc_i, 0)
                    ] = 1

        self.ddi_adj = ddi_adj
        return ddi_adj


if __name__ == "__main__":
    from pyhealth.datasets import MIMIC3BaseDataset

    base_dataset = MIMIC3BaseDataset(root="/srv/local/data/physionet.org/files/mimiciii/1.4", dev=True)
    task_dataset = DrugRecommendationDataset(base_dataset)
    print(task_dataset[1])
    task_dataset.info()
    task_dataset.stat()

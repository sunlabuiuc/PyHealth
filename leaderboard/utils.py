import datetime
import logging

import gspread
from oauth2client.service_account import ServiceAccountCredentials

from pyhealth.datasets import MIMIC3Dataset, eICUDataset, MIMIC4Dataset, OMOPDataset
from pyhealth.tasks import *
from pyhealth.metrics import *

from torch.utils.data import DataLoader


# connect to the Google Spreadsheet
def get_leaderboard_sheet(credential_file, doc_name, worksheet_id):
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    credentials = ServiceAccountCredentials.from_json_keyfile_name(credential_file,
                                                                   scopes)  # access the json key you downloaded earlier
    file = gspread.authorize(credentials)  # authenticate the JSON key with gspread
    sheet = file.open(doc_name)  # open sheet
    sheet = sheet.get_worksheet_by_id(worksheet_id)  # replace sheet_name with the name that corresponds to yours
    return sheet


# function to save the leaderboard data locally
def save_leaderboard_log(out_path, dataset_task_name, data, models):
    filename = out_path + '/' + str(datetime.date.today()) + '-' + dataset_task_name + '.log'
    logging.basicConfig(filename=filename, level=logging.INFO)
    logging.info(models)
    logging.info(data)


def get_dataset(dataset_name):
    dataset = None

    if dataset_name == "mimic3":
        mimic3dataset = MIMIC3Dataset(
            root="/srv/local/data/physionet.org/files/mimiciii/1.4",
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            dev=False,
            code_mapping={"NDC": "ATC"},
            refresh_cache=False,
        )
        dataset = mimic3dataset

    elif dataset_name == "eicu":
        eicudataset = eICUDataset(
            root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
            tables=["diagnosis", "medication", "physicalExam"],
            dev=False,
            refresh_cache=False,
        )
        dataset = eicudataset

    elif dataset_name == "mimic4":
        mimic4dataset = MIMIC4Dataset(
            root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            dev=False,
            code_mapping={"NDC": "ATC"},
            refresh_cache=False,
        )
        dataset = mimic4dataset

    elif dataset_name == "omop":
        omopdataset = OMOPDataset(
            root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2",
            tables=[
                "condition_occurrence",
                "procedure_occurrence",
                "drug_exposure",
                "measurement",
            ],
            dev=False,
            refresh_cache=False,
        )
        dataset = omopdataset

    return dataset


def get_data_location_on_sheet(eval_data_task):
    end_line = 2 + len(eval_data_task) - 1
    location = f'A2:E{end_line}'

    return location


def get_filtered_models(all_models, remove_models):
    for rm_model in remove_models:
        if rm_model in all_models:
            all_models.remove(rm_model)

    return all_models


def get_tasks_fn_for_datasets():
    tasks_mimic3 = [
        drug_recommendation_mimic3_fn,
        length_of_stay_prediction_mimic3_fn,
        mortality_prediction_mimic3_fn,
        readmission_prediction_mimic3_fn
    ]

    tasks_mimic4 = [
        drug_recommendation_mimic4_fn,
        length_of_stay_prediction_mimic4_fn,
        mortality_prediction_mimic4_fn,
        readmission_prediction_mimic4_fn
    ]

    tasks_eicu = [
        drug_recommendation_eicu_fn,
        length_of_stay_prediction_eicu_fn,
        mortality_prediction_eicu_fn,
        readmission_prediction_eicu_fn
    ]

    tasks_omop = [
        drug_recommendation_omop_fn,
        length_of_stay_prediction_omop_fn,
        mortality_prediction_omop_fn,
        readmission_prediction_omop_fn
    ]
    return tasks_mimic3, tasks_mimic4, tasks_eicu, tasks_omop


def get_metrics_result(mode, y_gt, y_pred, y_prob):
    jaccard, accuracy, f1, prauc = 0, 0, 0, 0

    if mode == "multilabel":
        jaccard = jaccard_multilabel(y_gt, y_pred)
        accuracy = accuracy_multilabel(y_gt, y_pred)
        f1 = f1_multilabel(y_gt, y_pred, average='macro')
        prauc = pr_auc_multilabel(y_gt, y_prob)

    elif mode == "binary":
        jaccard = jaccard_score(y_gt, y_pred, average='macro')
        accuracy = accuracy_score(y_gt, y_pred)
        f1 = f1_score(y_gt, y_pred, average='macro')
        prauc = average_precision_score(y_gt, y_prob)

    elif mode == "multiclass":
        jaccard = jaccard_score(y_gt, y_pred, average='macro')
        accuracy = accuracy_score(y_gt, y_pred)
        f1 = f1_score(y_gt, y_pred, average='macro')
        prauc = '-'

    # print metric name and score
    print("jaccard: ", jaccard)
    print("accuracy: ", accuracy)
    print("f1: ", f1)
    print("prauc: ", prauc)
    print('\n\n')

    return jaccard, accuracy, f1, prauc


def only_upper(s):
    return "".join(c for c in s if c.isupper())


def split_dataset_and_get_dataloaders(dataset, split_fn, ratio, collate_fn_dict):
    train_dataset, val_dataset, test_dataset = split_fn(dataset, ratio)

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_dict
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_dict
    )

    return train_loader, val_loader, test_loader


def train_process(trainer, model, train_loader, val_loader, val_metric):
    try:

        trainer.fit(model,
                    train_loader=train_loader,
                    epochs=50,
                    val_loader=val_loader,
                    val_metric=val_metric,
                    show_progress_bar=False)

        return True

    except:
        return False

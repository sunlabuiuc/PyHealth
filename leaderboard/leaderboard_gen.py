import argparse
import sys

from pyhealth.datasets import MIMIC3Dataset, eICUDataset, MIMIC4Dataset, OMOPDataset
from pyhealth.models import *
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.tasks import *
from pyhealth.datasets.utils import collate_fn_dict
from pyhealth.trainer import Trainer
from pyhealth.metrics import *
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neural_network import MLPClassifier as NN
from torch.utils.data import DataLoader

import gspread
from oauth2client.service_account import ServiceAccountCredentials

import time
import datetime
import logging
import warnings

sys.path.append('..')
warnings.filterwarnings('ignore')

RF = RF(max_depth=6, max_features="sqrt", n_jobs=-1, n_estimators=20)
NN = NN(alpha=1e-04, hidden_layer_sizes=(10, 1), early_stopping=True, max_iter=50, solver='lbfgs', max_fun=1500)


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
    return


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


def leaderboard_generation(args):

    if args.remote:
        # get our leaderboard sheet on:
        # https://docs.google.com/spreadsheets/d/1c4OwCSDaEt7vGmocidq1hK2HCTeB6ZHDzAZvlubpi08/edit#gid=1602645797
        leaderboard_sheet = get_leaderboard_sheet(credential_file=args.credentials,
                                                  doc_name=args.doc_name,
                                                  worksheet_id=args.sheet_id)

        # specify the areas to input result data
        leaderboard_location = {
            'mimic3-drugrec': 'C2:F11',
            'mimic4-drugrec': 'C16:F25',
            'eicu-drugrec': 'I2:L11',
            'omop-drugrec': 'I16:L25',

            'mimic3-mortality': 'C32:F38',
            'mimic4-mortality': 'C46:F52',
            'eicu-mortality': 'I32:L38',
            'omop-mortality': 'I46:L52',

            'mimic3-readmission': 'C61:F67',
            'mimic4-readmission': 'C75:F81',
            'eicu-readmission': 'I61:L67',
            'omop-readmission': 'I75:L81',

        }

    datasets = [
        "mimic3",
        "eicu",
        "omop",
        "mimic4"
    ]

    classic_ml_models = [LR(), RF, NN]
    tasks_mimic3 = [
        drug_recommendation_mimic3_fn,
        # length_of_stay_prediction_mimic3_fn,
        mortality_prediction_mimic3_fn,
        readmission_prediction_mimic3_fn]

    tasks_mimic4 = [
        drug_recommendation_mimic4_fn,
        # length_of_stay_prediction_mimic4_fn,
        mortality_prediction_mimic4_fn,
        readmission_prediction_mimic4_fn
    ]

    tasks_eicu = [
        drug_recommendation_eicu_fn,
        # length_of_stay_prediction_eicu_fn,
        mortality_prediction_eicu_fn,
        readmission_prediction_eicu_fn
    ]

    tasks_omop = [
        drug_recommendation_omop_fn,
        # length_of_stay_prediction_omop_fn,
        mortality_prediction_omop_fn,
        readmission_prediction_omop_fn
    ]

    # ==============================
    # traverse through all datasets
    for dataset_name in datasets:
        if dataset_name == "mimic3":
            task_list = tasks_mimic3
        elif dataset_name == "mimic4":
            task_list = tasks_mimic4
        elif dataset_name == "eicu":
            task_list = tasks_eicu
        elif dataset_name == "omop":
            task_list = tasks_omop
        else:
            print("Current leaderboard generation only supports mimic3, mimic4, eicu, omop datasets")
            raise ValueError

        dataset = get_dataset(dataset_name)

        for task in task_list:
            # set task to the dataset
            dataset.set_task(task)

            # split the dataset and create dataloaders
            train_dataset, val_dataset, test_dataset = split_by_patient(dataset, [0.8, 0.1, 0.1])
            train_loader = DataLoader(
                train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict
            )
            val_loader = DataLoader(
                val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_dict
            )
            test_loader = DataLoader(
                test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_dict
            )

            # specify tables and modes to use for different tasks
            task_name = task.__name__
            if "drug_recommendation" in task_name:
                models = [
                    ClassicML,
                    RNN,
                    CNN,
                    Transformer,
                    RETAIN,
                    GAMENet,
                    MICRON,
                    SafeDrug
                ]
                # Safedrug can only be used in mimic3 and mimic4
                if (dataset_name != "mimic3") and (dataset_name != "mimic4") and (SafeDrug in models):
                    models.remove(SafeDrug)

                tables_ = ["conditions", "procedures"]
                mode_ = "multilabel"
                val_metric = 'pr_auc_multilabel'
                dataset_task = dataset_name + "-drugrec"

            elif "mortality_prediction" in task_name:
                models = [
                    ClassicML,
                    RNN,
                    CNN,
                    Transformer,
                    RETAIN
                ]
                tables_ = ["conditions", "procedures", "drugs"]
                mode_ = "binary"
                val_metric = 'average_precision_score'
                dataset_task = dataset_name + "-mortality"

            elif "readmission_prediction" in task_name:
                models = [
                    ClassicML,
                    RNN,
                    CNN,
                    Transformer,
                    RETAIN
                ]
                tables_ = ["conditions", "procedures", "drugs"]
                mode_ = "binary"
                val_metric = 'average_precision_score'
                dataset_task = dataset_name + "-readmission"

            print("current task: " + task_name)

            # input leaderboard for each dataset-task
            eval_data_task = []

            # traverse all the models
            for current_model in models:
                if current_model.__name__ == "ClassicML":
                    for mlmodel in classic_ml_models:
                        print("current model: " + str(mlmodel))
                        model = current_model(
                            dataset=dataset,
                            tables=tables_,
                            target="label",
                            classifier=mlmodel,
                            mode=mode_,
                            output_path="./ckpt/" + str(mlmodel)[:-2]
                        )

                        trainer = Trainer(enable_logging=True, output_path="./output")
                        start = time.time()
                        trainer.fit(model,
                                    train_loader=train_loader,
                                    epochs=50,
                                    val_loader=val_loader,
                                    val_metric=val_metric,
                                    show_progress_bar=False)
                        end = time.time()
                        print('training time: ', end - start)

                        y_gt, y_prob, avg_loss = trainer.inference(model, test_loader)
                        y_pred = (y_prob > 0.5).astype(int)
                        
                        jaccard, accuracy, f1, prauc = get_metrics_result(mode_, y_gt, y_pred, y_prob)

                        # input leaderboard for each dataset-task-model
                        eval_data_model = [jaccard, accuracy, f1, prauc]
                        eval_data_task.append(eval_data_model)

                        # print metric name and score
                        print("jaccard: ", jaccard)
                        print("accuracy: ", accuracy)
                        print("f1: ", f1)
                        print("prauc: ", prauc)
                        print('\n\n')

                else:
                    device = "cuda:0"
                    print("current model: " + str(current_model))

                    model = current_model(
                        dataset=dataset,
                        tables=tables_,
                        target="label",
                        mode=mode_,
                    )

                    model.to(device)

                    trainer = Trainer(enable_logging=True, output_path="./output", device=device)

                    start = time.time()
                    trainer.fit(model,
                                train_loader=train_loader,
                                epochs=50,
                                val_loader=val_loader,
                                val_metric=val_metric,
                                show_progress_bar=False)
                    end = time.time()
                    print('training time: ', end - start)

                    y_gt, y_prob, avg_loss = trainer.inference(model, test_loader)
                    y_pred = (y_prob > 0.5).astype(int)

                    if mode_ == "multilabel":

                        jaccard = jaccard_multilabel(y_gt, y_pred)
                        accuracy = accuracy_multilabel(y_gt, y_pred)
                        f1 = f1_multilabel(y_gt, y_pred, average='macro')
                        prauc = pr_auc_multilabel(y_gt, y_prob)

                    elif mode_ == "binary":
                        jaccard = jaccard_score(y_gt, y_pred, average='macro')
                        accuracy = accuracy_score(y_gt, y_pred)
                        f1 = f1_score(y_gt, y_pred, average='macro')
                        prauc = average_precision_score(y_gt, y_prob)

                    # input leaderboard for each dataset-task-model
                    eval_data_model = [jaccard, accuracy, f1, prauc]
                    eval_data_task.append(eval_data_model)

                    # print metric name and score
                    print("jaccard: ", jaccard)
                    print("accuracy: ", accuracy)
                    print("f1: ", f1)
                    print("prauc: ", prauc)
                    print('\n\n')

            if args.remote:
                location = leaderboard_location[dataset_task]
                leaderboard_sheet.update(location, eval_data_task)

            print(eval_data_task)
            save_leaderboard_log(out_path=args.log_path, dataset_task_name=dataset_task, data=eval_data_task, models=models)
            print('Leaderboard updated for ' + dataset_task + '!')


def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", type=str, default='credentials.json')
    parser.add_argument("--doc_name", type=str, default='Pyhealth tracker')
    parser.add_argument("--sheet_id", type=int, default=1602645797)
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--remote", type=bool, default=True)

    args = parser.parse_args()

    return args


def main():
    args = construct_args()
    leaderboard_generation(args)


if __name__ == '__main__':
    main()

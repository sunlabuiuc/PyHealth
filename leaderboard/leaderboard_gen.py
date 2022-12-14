import sys

sys.path.append("..")

from pyhealth.models import *
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.datasets.utils import collate_fn_dict
from pyhealth.trainer import Trainer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neural_network import MLPClassifier as NN

from leaderboard.utils import *

import time
import copy
import argparse
import warnings

warnings.filterwarnings("ignore")

RF = RF(max_depth=6, max_features="sqrt", n_jobs=-1, n_estimators=20)
NN = NN(
    alpha=1e-04,
    hidden_layer_sizes=(10, 1),
    early_stopping=True,
    max_iter=50,
    solver="lbfgs",
    max_fun=1500,
)

leaderboard_sheet = None


def leaderboard_generation(args):
    global leaderboard_sheet

    if args.remote:
        # get our leaderboard sheet on:
        # https://docs.google.com/spreadsheets/d/1c4OwCSDaEt7vGmocidq1hK2HCTeB6ZHDzAZvlubpi08/edit#gid=1602645797
        leaderboard_sheet = get_leaderboard_sheet(
            credential_file=args.credentials,
            doc_name=args.doc_name,
            worksheet_id=args.sheet_id,
        )

    datasets = args.datasets

    tasks_mimic3, tasks_mimic4, tasks_eicu, tasks_omop = get_tasks_fn_for_datasets()

    eval_data_task = []

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
            print(
                "Current leaderboard generation only supports mimic3, mimic4, eicu, omop-format datasets"
            )
            raise ValueError

        base_dataset = get_dataset(dataset_name)

        for task in task_list:
            # set task to the dataset
            dataset = base_dataset.set_task(task)

            # split the dataset and create dataloaders
            train_loader, val_loader, test_loader = split_dataset_and_get_dataloaders(
                dataset,
                split_fn=split_by_patient,
                ratio=[0.8, 0.1, 0.1],
                collate_fn_dict=collate_fn_dict,
            )

            # specify tables and modes to use for different tasks
            task_name = task.__name__
            models = copy.deepcopy(args.model_list)
            tables_ = []
            mode_ = ""
            dataset_task = ""
            val_metric = None

            if "drug_recommendation" in task_name:
                # Safedrug can only be used in mimic3 and mimic4
                if (
                    (dataset_name != "mimic3")
                    and (dataset_name != "mimic4")
                    and (SafeDrug in models)
                ):
                    models.remove(SafeDrug)

                tables_ = ["conditions", "procedures"]
                mode_ = "multilabel"
                val_metric = "pr_auc"
                dataset_task = dataset_name + "-drugrec"

            elif "mortality_prediction" in task_name:

                models = get_filtered_models(models, [GAMENet, MICRON, SafeDrug])

                tables_ = ["conditions", "procedures", "drugs"]
                mode_ = "binary"
                val_metric = "pr_auc"
                dataset_task = dataset_name + "-mortality"

            elif "readmission_prediction" in task_name:

                models = get_filtered_models(models, [GAMENet, MICRON, SafeDrug])

                tables_ = ["conditions", "procedures", "drugs"]
                mode_ = "binary"
                val_metric = "pr_auc"
                dataset_task = dataset_name + "-readmission"

            elif "length_of_stay_prediction" in task_name:

                models = get_filtered_models(models, [GAMENet, MICRON, SafeDrug])
                tables_ = ["conditions", "procedures"]
                mode_ = "multiclass"
                val_metric = "accuracy"
                dataset_task = dataset_name + "-lenOfStay"

            print("current task: " + task_name)

            # input leaderboard for each dataset-task
            classic_ml_models = [LR(), RF, NN]

            # traverse all the models
            for current_model in models:
                if current_model.__name__ == "ClassicML":
                    for ml_model in classic_ml_models:
                        print("current model: " + str(ml_model))
                        model_name = only_upper(str(ml_model))
                        model = current_model(
                            dataset=dataset,
                            tables=tables_,
                            target="label",
                            classifier=ml_model,
                            mode=mode_,
                            output_path="./ckpt/" + str(ml_model)[:-2],
                        )

                        trainer = Trainer(
                            model=model, enable_logging=True, output_path="./output"
                        )

                        start = time.time()

                        # in case there is only one class in the samples, the dataloader should be re-created
                        while True:
                            if train_process(
                                trainer, model, train_loader, val_loader, val_metric
                            ):
                                break
                            else:
                                # split the dataset and create dataloaders
                                (
                                    train_loader,
                                    val_loader,
                                    test_loader,
                                ) = split_dataset_and_get_dataloaders(
                                    dataset,
                                    split_fn=split_by_patient,
                                    ratio=[0.8, 0.1, 0.1],
                                    collate_fn_dict=collate_fn_dict,
                                )
                                continue

                        end = time.time()

                        print("training time: ", end - start)

                        y_gt, y_prob, avg_loss = trainer.inference(test_loader)

                        jaccard, accuracy, f1, prauc = get_metrics_result(
                            mode_, y_gt, y_prob
                        )

                        # input leaderboard for each dataset-task-model
                        dataset_task_model = dataset_task + "-" + model_name
                        eval_data_model = [
                            dataset_task_model,
                            jaccard,
                            accuracy,
                            f1,
                            prauc,
                        ]
                        eval_data_task.append(eval_data_model)

                else:
                    device = "cuda:0"
                    print("current model: " + str(current_model))
                    model_name = current_model.__name__
                    model = current_model(
                        dataset=dataset,
                        tables=tables_,
                        target="label",
                        mode=mode_,
                    )

                    model.to(device)

                    trainer = Trainer(
                        model=model,
                        enable_logging=True,
                        output_path="./output",
                        device=device,
                    )

                    start = time.time()

                    # in case there is only one class in the samples, the dataloader should be re-created
                    while True:
                        if train_process(
                            trainer, model, train_loader, val_loader, val_metric
                        ):
                            break
                        else:
                            # split the dataset and create dataloaders
                            (
                                train_loader,
                                val_loader,
                                test_loader,
                            ) = split_dataset_and_get_dataloaders(
                                dataset,
                                split_fn=split_by_patient,
                                ratio=[0.8, 0.1, 0.1],
                                collate_fn_dict=collate_fn_dict,
                            )
                            continue

                    end = time.time()
                    print("training time: ", end - start)

                    y_gt, y_prob, avg_loss = trainer.inference(test_loader)
                    y_pred = (y_prob > 0.5).astype(int)

                    jaccard, accuracy, f1, prauc = get_metrics_result(
                        mode_, y_gt, y_prob
                    )

                    # input leaderboard for each dataset-task-model
                    dataset_task_model = dataset_task + "-" + model_name
                    eval_data_model = [dataset_task_model, jaccard, accuracy, f1, prauc]
                    eval_data_task.append(eval_data_model)

            if args.remote:
                location = get_data_location_on_sheet(eval_data_task)
                leaderboard_sheet.update(location, eval_data_task)

            print(eval_data_task)
            save_leaderboard_log(
                out_path=args.log_path,
                dataset_task_name=dataset_task,
                data=eval_data_task,
                models=models,
            )
            print("Leaderboard updated for " + dataset_task + "!")


def plots_generation(args):
    if args.plot is False:
        return

    # dfs = read_dataframes_by_time_from_gcp(args.credentials)
    dfs = read_dataframes_by_time_from_gcp_with_no_credentials()

    bokeh_figures = []
    # for task in args.tasks:
    #     for dataset in args.datasets:
    df = get_spec_df_with_time(dfs, "", "")
    bokeh_figure = generate_bokeh_figure(df)
    bokeh_figures.append(bokeh_figure)

    show(column(bokeh_figures))


def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", type=str, default="./credentials.json")
    parser.add_argument("--doc_name", type=str, default="Pyhealth tracker")
    parser.add_argument("--sheet_id", type=int, default=2062485923)
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument(
        "--datasets", type=list, default=["mimic3", "eicu", "omop", "mimic4"]
    )
    parser.add_argument(
        "--tasks",
        type=list,
        default=["drugrec", "lenOfStay", "mortality", "readmission"],
    )
    parser.add_argument(
        "--model_list",
        type=list,
        default=[RNN, CNN, Transformer, RETAIN, GAMENet, MICRON, SafeDrug],
    )
    parser.add_argument("--remote", type=bool, default=True)
    parser.add_argument("--plot", type=bool, default=True)

    args = parser.parse_args()

    return args


def main():
    args = construct_args()
    leaderboard_generation(args)
    plots_generation(args)


if __name__ == "__main__":
    main()

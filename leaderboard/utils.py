import datetime
import logging

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google.cloud import storage
import io
import pandas as pd

from bokeh.models import HoverTool, CDSView
from bokeh.layouts import row, column, gridplot
from bokeh.models import (
    ColumnDataSource,
    CDSView,
    CheckboxGroup,
    CustomJS,
    BooleanFilter,
    Select,
    Div
)
from bokeh.plotting import figure, show, curdoc
import numpy as np
import random

from pyhealth.datasets import MIMIC3Dataset, eICUDataset, MIMIC4Dataset, OMOPDataset
from pyhealth.tasks import *
from pyhealth.metrics import *

from torch.utils.data import DataLoader


# connect to the Google Spreadsheet
def get_leaderboard_sheet(credential_file, doc_name, worksheet_id):
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        credential_file, scopes
    )  # access the json key you downloaded earlier
    file = gspread.authorize(credentials)  # authenticate the JSON key with gspread
    sheet = file.open(doc_name)  # open sheet
    sheet = sheet.get_worksheet_by_id(
        worksheet_id
    )  # replace sheet_name with the name that corresponds to yours
    return sheet


# function to save the leaderboard data locally
def save_leaderboard_log(out_path, dataset_task_name, data, models):
    filename = (
        out_path + "/" + str(datetime.date.today()) + "-" + dataset_task_name + ".log"
    )
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
    location = f"A2:E{end_line}"

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
        readmission_prediction_mimic3_fn,
    ]

    tasks_mimic4 = [
        drug_recommendation_mimic4_fn,
        length_of_stay_prediction_mimic4_fn,
        mortality_prediction_mimic4_fn,
        readmission_prediction_mimic4_fn,
    ]

    tasks_eicu = [
        drug_recommendation_eicu_fn,
        length_of_stay_prediction_eicu_fn,
        mortality_prediction_eicu_fn,
        readmission_prediction_eicu_fn,
    ]

    tasks_omop = [
        drug_recommendation_omop_fn,
        length_of_stay_prediction_omop_fn,
        mortality_prediction_omop_fn,
        readmission_prediction_omop_fn,
    ]
    return tasks_mimic3, tasks_mimic4, tasks_eicu, tasks_omop


def get_metrics_result(mode, y_gt, y_prob):
    metrics = []
    metrics_fn = None

    if mode == "multilabel":
        metrics_fn = multilabel_metrics_fn
        metrics = ["jaccard_macro", "accuracy", "f1_macro", "pr_auc_macro"]

    elif mode == "binary":
        metrics_fn = binary_metrics_fn
        metrics = ["jaccard", "accuracy", "f1", "pr_auc"]

    elif mode == "multiclass":
        metrics_fn = multiclass_metrics_fn
        metrics = ["jaccard_macro", "accuracy", "f1_macro"]

    results = metrics_fn(y_gt, y_prob, metrics=metrics, threshold=0.5)

    jaccard = results["jaccard"] if ("jaccard" in metrics) else results["jaccard_macro"]
    accuracy = results["accuracy"]
    f1 = results["f1"] if ("f1" in metrics) else results["f1_macro"]
    prauc = (
        results["pr_auc"]
        if ("pr_auc" in metrics)
        else results["pr_auc_macro"]
        if ("pr_auc_macro" in metrics)
        else "-"
    )

    # print metric name and score
    print("jaccard: ", jaccard)
    print("accuracy: ", accuracy)
    print("f1: ", f1)
    print("prauc: ", prauc)
    print("\n\n")

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

        trainer.train(
            train_dataloader=train_loader,
            epochs=50,
            val_dataloader=val_loader,
            monitor=val_metric,
        )

        return True

    except:
        return False


def read_dataframes_by_time_from_gcp(credentials):
    import os

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials

    dfs = {}

    storage_client = storage.Client()

    bucket_name = "pyhealth"
    prefix = "leaderboard_data/data/"
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    cnt = 0
    for blob in blobs:
        if cnt > 0:
            data = blob.download_as_bytes()
            df = pd.read_csv(io.BytesIO(data))
            name_ = blob.name.split("-")
            time = name_[1] + name_[2] + name_[3][:-4]
            dfs[time] = df
        cnt += 1

    return dfs


def read_dataframes_by_time_from_gcp_with_no_credentials():
    from datetime import datetime, date, timedelta

    first_data_time = datetime.strptime("2022-10-28", "%Y-%m-%d")
    now = datetime.now()
    weeks = int(abs(now - first_data_time).days / 7) + 1
    data = []
    for week in range(weeks):
        data_time = (first_data_time + timedelta(weeks=week)).strftime("%Y-%m-%d")
        file_name = f"https://storage.googleapis.com/pyhealth/leaderboard_data/data/DATA-{data_time}.csv"
        data.append(file_name)

    dfs = {}

    for d in data:
        try:
            df = pd.read_csv(d)
            name_ = d.split("-")
            time = name_[1] + name_[2] + name_[3][:-4]
            dfs[time] = df
        except:
            break

    return dfs


def get_spec_df_with_time(dfs, dataset, task):
    from datetime import datetime

    for key in dfs.keys():
        dfs[key]["date"] = key

    df = pd.concat(dfs.values())
    df["date"] = [datetime.strptime(date, "%Y%m%d") for date in df["date"].iloc()]

    if dataset == "" and task == "":
        return df

    df = df[df["Dataset-Task-Model"].str.contains(dataset)]
    df = df[df["Dataset-Task-Model"].str.contains(task)]

    return df


def make_bokeh_plot(source, metric, f):
    p = figure(
        height=190,
        width=500,
        tools=["pan, box_zoom, reset, save, crosshair"],
        toolbar_location="above",
        y_range=[-0.19, 1.2],
        # x_axis_label="date",
        y_axis_label=metric,
        x_axis_type="datetime",
    )

    if metric == "Jaccard":
        p.triangle(
            source=source,
            x="date",
            y="Jaccard",
            color="color",
            size=8,
            alpha=0.5,
            view=CDSView(source=source, filters=[f]),
        )
    if metric == "Accuracy":
        p.circle(
            source=source,
            x="date",
            y="Accuracy",
            color="color",
            size=8,
            alpha=0.5,
            view=CDSView(source=source, filters=[f]),
        )
    if metric == "F1":
        p.square(
            source=source,
            x="date",
            y="F1",
            color="color",
            size=8,
            alpha=0.5,
            view=CDSView(source=source, filters=[f]),
        )
    if metric == "PRAUC":
        p.circle(
            source=source,
            x="date",
            y="PRAUC",
            color="color",
            size=8,
            alpha=0.5,
            view=CDSView(source=source, filters=[f]),
        )
        p.xaxis.axis_label = "date"
        p.height = 220

    hover = HoverTool(
        tooltips=[
            ("model", "@{Dataset-Task-Model}"),
            ("date", "$x{%F}"),
            ("value", f"@{metric}"),
        ],
        formatters={"$x": "datetime", "Dataset-Task-Model": "printf"},
    )

    p.add_tools(hover)

    p.legend.location = "bottom_right"
    # p.xaxis.major_label_orientation = 3.14 / 2
    p.yaxis.axis_label_text_font_size = "11pt"
    p.xaxis.axis_label_text_font_size = "11pt"
    p.axis.axis_label_text_font_style = "bold"
    p.margin = (5, 5, 0, 10)

    return p


def dropdown(items, init, label):
    dp = Select(options=items, value=init, title=label)
    return dp


def generate_bokeh_figure(df):
    curdoc().theme = "caliber"

    df["letter"] = df["Dataset-Task-Model"]

    number_of_colors = 256
    colors = np.array(
        [
            "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
            for i in range(number_of_colors)
        ]
    )
    df["color"] = np.zeros(len(df))

    df = df.rename(columns={"Macro-F1": "F1"})

    models = sorted(set(df["Dataset-Task-Model"]))

    for model in models:
        color = random.choice(colors)
        df.loc[df["Dataset-Task-Model"].str.contains(model), "color"] = color

    dt = pd.DataFrame()
    dt["model"] = models
    for model in models:
        dataset, task = model.split("-")[0], model.split("-")[1]
        dt.loc[dt["model"].str.contains(model), "dataset"] = dataset
        dt.loc[dt["model"].str.contains(model), "task"] = task

    sc = ColumnDataSource(df)
    sc_o = ColumnDataSource(dt)
    dt_d = dt[dt["dataset"] == "mimic3"]
    sc_d = ColumnDataSource(dt_d)
    dt_t = dt[dt["task"] == "DrugRec"]
    sc_t = ColumnDataSource(dt_t)

    dp_dataset = dropdown(
        items=list(dt["dataset"].unique()), init="mimic3", label="Dataset"
    )
    dp_task = dropdown(items=list(dt["task"].unique()), init="DrugRec", label="Task")
    # dp_models = dropdown(models, "Models")

    # prefix = dp_dataset.value + '-' + dp_task.value
    # dp_models.options = [model for model in models if model.startswith(prefix)]

    init_labels = [
        *set(list(sc_t.data["model"])).intersection(list(sc_d.data["model"]))
    ]
    # active_letter = df['Dataset-Task-Model'].iloc()[0]
    f = BooleanFilter(booleans=[l == init_labels[0] for l in df["Dataset-Task-Model"]])
    cg = CheckboxGroup(labels=init_labels, active=[models.index(init_labels[0])])
    cg.js_on_change(
        "active",
        CustomJS(
            args=dict(source=sc, f=f, sc_d=sc_d, sc_t=sc_t),
            code="""\
                                const letters = cb_obj.active.map(idx => cb_obj.labels[idx]);
                                f.booleans = source.data.letter.map(l => letters.includes(l));
                                source.change.emit();
                             """,
        ),
    )

    callback_ds = CustomJS(
        args=dict(source=sc_o, sc_d=sc_d, sc_t=sc_t, other=cg),
        code="""\
                            sc_d.data['task'] = [];
                            sc_d.data['model'] = [];
                            sc_d.data['dataset'] = [];
                            for(var i = 0; i <= source.get_length(); i++){
                                if(source.data['dataset'][i] == cb_obj.value){
                                    sc_d.data['task'].push(source.data['task'][i]);
                                    sc_d.data['model'].push(source.data['model'][i]);
                                    sc_d.data['dataset'].push(source.data['dataset'][i]);
                                }
                            }
                            const intersect = sc_d.data['model'].filter(x => sc_t.data['model'].indexOf(x) !== -1);
                            other.labels = intersect;
                            other.active = new Array()
                            other.change.emit();
                            sc_d.change.emit();
                        """,
    )
    callback_tk = CustomJS(
        args=dict(source=sc_o, sc_t=sc_t, sc_d=sc_d, other=cg),
        code="""\
                            sc_t.data['model'] = [];
                            sc_t.data['task'] = [];
                            sc_t.data['dataset'] = [];
                            for(var i = 0; i <= source.get_length(); i++){
                                if(source.data['task'][i] == cb_obj.value){
                                    sc_t.data['task'].push(source.data['task'][i]);
                                    sc_t.data['model'].push(source.data['model'][i]);
                                    sc_t.data['dataset'].push(source.data['dataset'][i]);
                                }
                            }
                            const intersect = sc_t.data['model'].filter(x => sc_d.data['model'].indexOf(x) !== -1);
                            other.labels = intersect;
                            other.active = new Array()
                            other.change.emit();
                            sc_t.change.emit();
                        """,
    )
    dp_dataset.js_on_change("value", callback_ds)
    dp_task.js_on_change("value", callback_tk)
    dp_dataset.js_link("value", cg, "labels")
    dp_dataset.js_link("value", cg, "active")
    dp_task.js_link("value", cg, "labels")
    dp_task.js_link("value", cg, "active")

    div = Div(
        text="""Models (run by <a href="https://pyhealth.readthedocs.io/en/latest/api/models.html">pyhealth.models</a>)""",
        width=200,
        height=20,
    )

    cg.width = 220
    cg.height = 700
    div.margin = (20, 0, 0, 5)
    cg.margin = (5, 0, 0, 5)
    dp_task.width = 160
    dp_dataset.width = 160

    p_jac = make_bokeh_plot(sc, "Jaccard", f)
    p_acc = make_bokeh_plot(sc, "Accuracy", f)
    p_f1 = make_bokeh_plot(sc, "F1", f)
    p_prauc = make_bokeh_plot(sc, "PRAUC", f)

    # return row(column(dp_dataset, dp_task, cg), gridplot([[p_jac, p_acc], [p_f1, p_prauc]]))
    return row(
        column(dp_dataset, dp_task, div, cg), column(p_jac, p_acc, p_f1, p_prauc)
    )

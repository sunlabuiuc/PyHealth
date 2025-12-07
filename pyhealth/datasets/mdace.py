"""
MDACE is the first publicly available code evidence dataset, which is built on a subset of the MIMIC-III clinical records.
The dataset – annotated by professional medical coders – consists of 302 Inpatient charts with 3,934 evidence spans and 52 Profee charts with 5,563 evidence spans.

The dataset can be found at
https://github.com/3mcloud/MDACE.git


@inproceedings{cheng-etal-2023-mdace,
    title = "{MDACE}: {MIMIC} Documents Annotated with Code Evidence",
    author = "Cheng, Hua  and
      Jafari, Rana  and
      Russell, April  and
      Klopfer, Russell  and
      Lu, Edmond  and
      Striner, Benjamin  and
      Gormley, Matthew",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.416/",
    doi = "10.18653/v1/2023.acl-long.416",
    pages = "7534--7550",
    abstract = "We introduce a dataset for evidence/rationale extraction on an extreme multi-label classification task over long medical documents. One such task is Computer-Assisted Coding (CAC) which has improved significantly in recent years, thanks to advances in machine learning technologies. Yet simply predicting a set of final codes for a patient encounter is insufficient as CAC systems are required to provide supporting textual evidence to justify the billing codes. A model able to produce accurate and reliable supporting evidence for each code would be a tremendous benefit. However, a human annotated code evidence corpus is extremely difficult to create because it requires specialized knowledge. In this paper, we introduce MDACE, the first publicly available code evidence dataset, which is built on a subset of the MIMIC-III clinical records. The dataset {--} annotated by professional medical coders {--} consists of 302 Inpatient charts with 3,934 evidence spans and 52 Profee charts with 5,563 evidence spans. We implemented several evidence extraction methods based on the EffectiveCAN model (Liu et al., 2021) to establish baseline performance on this dataset. MDACE can be used to evaluate code evidence extraction methods for CAC systems, as well as the accuracy and interpretability of deep learning models for multi-label classification. We believe that the release of MDACE will greatly improve the understanding and application of deep learning technologies for medical coding and document classification."
}


The loading code is from the GitHub repository of following paper
https://github.com/JoakimEdin/explainable-medical-coding.git

@inproceedings{edin-etal-2024-unsupervised,
    title = "An Unsupervised Approach to Achieve Supervised-Level Explainability in Healthcare Records",
    author = "Edin, Joakim  and
      Maistro, Maria  and
      Maal{\o}e, Lars  and
      Borgholt, Lasse  and
      Havtorn, Jakob Drachmann  and
      Ruotsalo, Tuukka",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.280/",
    doi = "10.18653/v1/2024.emnlp-main.280",
    pages = "4869--4890",
    abstract = "Electronic healthcare records are vital for patient safety as they document conditions, plans, and procedures in both free text and medical codes. Language models have significantly enhanced the processing of such records, streamlining workflows and reducing manual data entry, thereby saving healthcare providers significant resources. However, the black-box nature of these models often leaves healthcare professionals hesitant to trust them. State-of-the-art explainability methods increase model transparency but rely on human-annotated evidence spans, which are costly. In this study, we propose an approach to produce plausible and faithful explanations without needing such annotations. We demonstrate on the automated medical coding task that adversarial robustness training improves explanation plausibility and introduce AttInGrad, a new explanation method superior to previous ones. By combining both contributions in a fully unsupervised setup, we produce explanations of comparable quality, or better, to that of a supervised approach. We release our code and model weights."
}

Script usage
$(PYTHON_INTERPRETER) explainable_medical_coding/data/prepare_mdace.py data/raw data/processed

"""
import json
import logging
import random
import string
from collections import defaultdict
from pathlib import Path

import click
import polars as pl
from dotenv import find_dotenv, load_dotenv

# Column names
ID_COLUMN = "_id"
SUBJECT_ID_COLUMN = "subject_id"
TEXT_COLUMN = "text"

random.seed(10)


def parse_code_dataframe(
    df: pl.DataFrame,
    code_column: str = "diagnosis_codes",
    code_type_column: str = "diagnosis_code_type",
) -> pl.DataFrame:
    """Change names of colums, remove duplicates and Nans, and takes a dataframe and a column name
    and returns a series with the column name and a list of codes.

    Example:
        Input:
                subject_id  _id     target
                       2   163353     V3001
                       2   163353      V053
                       2   163353      V290

        Output:
            target    [V053, V290, V3001]

    Args:
        row (pd.DataFrame): Dataframe with a column of codes.
        col (str): column name of the codes.

    Returns:
        pd.Series: Series with the column name and a list of codes.
    """

    df = df.filter(df[code_column].is_not_null())
    df = df.unique(subset=[ID_COLUMN, code_column])
    df = df.group_by([ID_COLUMN, code_type_column]).agg(
        pl.col(code_column).map_elements(list).alias(code_column)
    )
    return df


def get_mdace_annotations(path: Path) -> pl.DataFrame:
    rows = []
    for json_path in path.glob("**/*.json"):
        with open(json_path, "r", encoding="utf8") as json_file:
            case_annotations = json.load(json_file)
            hadm_id = case_annotations["hadm_id"]

            for note in case_annotations["notes"]:
                note_id = note["note_id"]
                note_category = note["category"]
                note_description = note["description"]

                code2spans = defaultdict(list)  # code -> list of spans
                code2system = {}  # code -> code system (e.g. ICD-9, ICD-10, etc.)

                for annotation in note["annotations"]:
                    code = annotation["code"]
                    code2system[code] = annotation["code_system"]
                    code2spans[code].append((annotation["begin"], annotation["end"]))

                for code, spans in code2spans.items():
                    rows.append(
                        (
                            hadm_id,
                            note_id,
                            note_category,
                            note_description,
                            code2system[code],
                            code,
                            spans,
                        )
                    )
                # print(code_dict)

    schema = {
        ID_COLUMN: pl.Int64,
        "note_id": pl.Int64,
        "note_type": pl.Utf8,
        "note_subtype": pl.Utf8,
        "code_type": pl.Utf8,
        "code": pl.Utf8,
        "spans": pl.List,
    }
    return pl.DataFrame(schema=schema, data=rows)


def trim_annotations(
    span: tuple[int, int],
    text: str,
    punctuations: set[str] = set(string.punctuation + "\n\t "),
) -> tuple[int, int]:
    start = span[0]
    end = span[1]

    if text[end] in punctuations:
        end -= 1

    if text[start] in punctuations:
        start += 1

    return start, end


def clean_mdace_annotations(
    mdace_annotations: pl.DataFrame, mdace_notes: pl.DataFrame
) -> pl.DataFrame:
    mdace_annotations = mdace_annotations.join(
        mdace_notes[["note_id", "text"]], on="note_id", how="inner"
    )
    mdace_annotations = mdace_annotations.with_columns(
        spans=pl.struct("text", "spans").map_elements(
            lambda row: [trim_annotations(span, row["text"]) for span in row["spans"]]
        )
    )

    return mdace_annotations


@click.command()
@click.argument("input_filepath_str", type=click.Path(exists=True))
@click.argument("output_filepath_str", type=click.Path())
def main(input_filepath_str: str, output_filepath_str: str):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    input_filepath = Path(input_filepath_str)
    output_filepath = Path(output_filepath_str)
    output_filepath.mkdir(parents=True, exist_ok=True)

    # Load the dataframes
    mimic_notes = pl.read_csv(
        input_filepath / "physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz"
    )
    mimic_notes = mimic_notes.rename(
        {
            "HADM_ID": ID_COLUMN,
            "SUBJECT_ID": SUBJECT_ID_COLUMN,
            "ROW_ID": "note_id",
            "TEXT": TEXT_COLUMN,
            "CATEGORY": "note_type",
            "DESCRIPTION": "note_subtype",
        }
    )

    mdace_inpatient_annotations = get_mdace_annotations(
        Path("data/raw/MDace/Inpatient")
    )
    mdace_profee_annotations = get_mdace_annotations(Path("data/raw/MDace/Profee"))
    mdace_notes = mimic_notes.filter(
        pl.col("note_id").is_in(mdace_inpatient_annotations["note_id"])
    )
    mdace_inpatient_annotations = clean_mdace_annotations(
        mdace_inpatient_annotations, mdace_notes
    )
    mdace_profee_annotations = clean_mdace_annotations(
        mdace_profee_annotations, mdace_notes
    )

    mdace_inpatient_annotations = mdace_inpatient_annotations.with_columns(
        pl.col("code_type")
        .str.replace("ICD-9-CM", "icd9cm")
        .str.replace("ICD-10-CM", "icd10cm")
        .str.replace("ICD-10-PCS", "icd10pcs")
        .str.replace("CPT", "cpt")
        .str.replace("ICD-9-PCS", "icd9pcs")
    )

    mdace_profee_annotations = mdace_profee_annotations.with_columns(
        pl.col("code_type")
        .str.replace("ICD-9-CM", "icd9cm")
        .str.replace("ICD-10-CM", "icd10cm")
        .str.replace("ICD-10-PCS", "icd10pcs")
        .str.replace("CPT", "cpt")
        .str.replace("ICD-9-PCS", "icd9pcs")
    )

    # convert note_id to string
    mdace_notes = mdace_notes.with_columns(
        note_id=pl.col("note_id").cast(pl.Utf8),
    )
    mdace_inpatient_annotations = mdace_inpatient_annotations.with_columns(
        note_id=pl.col("note_id").cast(pl.Utf8),
    )
    mdace_profee_annotations = mdace_profee_annotations.with_columns(
        note_id=pl.col("note_id").cast(pl.Utf8),
    )

    # save files to disk
    mdace_notes.write_parquet(output_filepath / "mdace_notes.parquet")
    mdace_inpatient_annotations.write_parquet(
        output_filepath / "mdace_inpatient_annotations.parquet"
    )
    mdace_profee_annotations.write_parquet(
        output_filepath / "mdace_profee_annotations.parquet"
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

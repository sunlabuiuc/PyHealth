from pathlib import Path
import pandas as pd


DATA_PATH = Path.home() / "Desktop" / "mimiciv-3.1"
OUTPUT_PATH = DATA_PATH / "derived"

# Common ICD codes for sepsis and severe sepsis
ICD9_PREFIXES = ("038",)
ICD9_CODES = {"99591", "99592", "78552"}

ICD10_PREFIXES = ("A40", "A41")
ICD10_CODES = {"R6520", "R6521"}


def load_data():
    patients = pd.read_csv(
        DATA_PATH / "hosp" / "patients.csv.gz",
        usecols=[
            "subject_id", "gender", "anchor_age",
            "anchor_year", "anchor_year_group"
        ],
    )

    admissions = pd.read_csv(
        DATA_PATH / "hosp" / "admissions.csv.gz",
        usecols=[
            "subject_id", "hadm_id", "admittime",
            "dischtime", "hospital_expire_flag"
        ],
        parse_dates=["admittime", "dischtime"],
    )

    diagnoses = pd.read_csv(
        DATA_PATH / "hosp" / "diagnoses_icd.csv.gz",
        usecols=["hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": str},
    )

    icustays = pd.read_csv(
        DATA_PATH / "icu" / "icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "intime"],
        parse_dates=["intime"],
    )

    return patients, admissions, diagnoses, icustays


def select_sepsis(diagnoses):
    codes = (
        diagnoses["icd_code"]
        .str.upper()
        .str.replace(".", "", regex=False)
        .str.strip()
    )

    icd9 = (
        (diagnoses["icd_version"] == 9)
        & (
            codes.str.startswith(ICD9_PREFIXES, na=False)
            | codes.isin(ICD9_CODES)
        )
    )

    icd10 = (
        (diagnoses["icd_version"] == 10)
        & (
            codes.str.startswith(ICD10_PREFIXES, na=False)
            | codes.isin(ICD10_CODES)
        )
    )

    return diagnoses.loc[icd9 | icd10, "hadm_id"].drop_duplicates()


def build_cohort(patients, admissions, diagnoses, icustays):
    sepsis_ids = select_sepsis(diagnoses)

    cohort = admissions[
        admissions["hadm_id"].isin(sepsis_ids)
    ].copy()

    print("ICD-coded sepsis admissions:", len(cohort))

    cohort = cohort.merge(
        patients,
        on="subject_id",
        how="inner",
        validate="many_to_one",
    )

    # Age at admission based on the shifted MIMIC dates
    cohort["age"] = (
        cohort["anchor_age"]
        + cohort["admittime"].dt.year
        - cohort["anchor_year"]
    )

    cohort = cohort[cohort["age"] >= 18].copy()

    print("Adult sepsis admissions:", len(cohort))

    # Keep the first eligible admission for each patient
    cohort = cohort.sort_values(["subject_id", "admittime", "hadm_id"])
    cohort = cohort.drop_duplicates("subject_id", keep="first")

    print("First admission per patient:", len(cohort))

    # First ICU stay within each hospital admission, if present
    first_icu = (
        icustays.sort_values(["hadm_id", "intime", "stay_id"])
        .drop_duplicates("hadm_id", keep="first")
        [["hadm_id", "stay_id", "intime"]]
    )

    cohort = cohort.merge(
        first_icu,
        on="hadm_id",
        how="left",
        validate="one_to_one",
    )

    cohort["icu_flag"] = cohort["stay_id"].notna().astype(int)
    cohort["hospital_mortality"] = cohort["hospital_expire_flag"]

    return cohort


def main():
    print("MIMIC-IV 3.1: ICD-defined sepsis cohort extraction")

    patients, admissions, diagnoses, icustays = load_data()

    print("\nOriginal tables:")
    print("Patients:", len(patients))
    print("Admissions:", len(admissions))
    print("Diagnoses:", len(diagnoses))
    print("ICU stays:", len(icustays))

    cohort = build_cohort(
        patients, admissions, diagnoses, icustays
    )

    print("\nFinal cohort:")
    print("Patients:", cohort["subject_id"].nunique())
    print("Admissions:", cohort["hadm_id"].nunique())
    print("ICU admissions:", cohort["icu_flag"].sum())
    print("Hospital mortality:",
          round(cohort["hospital_mortality"].mean(), 4))

    print("\nMortality distribution:")
    print(cohort["hospital_mortality"].value_counts())

    print("\nAnchor year groups:")
    print(cohort["anchor_year_group"].value_counts().sort_index())

    print("\nAge summary:")
    print(cohort["age"].describe().round(1))

    print("\nHospital mortality by anchor year group:")
    print(
        cohort.groupby("anchor_year_group")["hospital_mortality"]
        .agg(["count", "mean"])
        .round(4)
    )

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_PATH / "sepsis_icd_cohort.csv"
    cohort.to_csv(output_file, index=False)

    print("\nSaved cohort to:", output_file)


if __name__ == "__main__":
    main()

"""MedMod ConVIRT pretraining set, on a tiny fake MedMod extraction (no MIMIC)."""

import numpy as np
import pandas as pd
import pytest

from pyhealth.datasets import get_dataloader
from pyhealth.datasets.medmod.discretizer import Discretizer
from pyhealth.datasets.medmod.pretraining import (
    create_pretraining_dataset,
    load_pretraining_samples,
    pretraining_cohort,
)
from pyhealth.datasets.medmod.synthetic import write_dummy_jpg

INTIME = pd.Timestamp("2150-01-01 00:00")


def _episode(path, hours_and_hr):
    disc = Discretizer()
    hr = 1 + disc.channels.index("Heart Rate")
    lines = [",".join(disc.header)]
    for t, v in hours_and_hr:
        row = [repr(float(t))] + [""] * len(disc.channels)
        row[hr] = v
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture()
def fake(tmp_path):
    ext, cxr, img = tmp_path / "ext", tmp_path / "cxr", tmp_path / "resized"
    (ext / "root").mkdir(parents=True)
    (ext / "phenotyping" / "train").mkdir(parents=True)
    cxr.mkdir()
    # stays 1-5 (subjects 11-15), 30 h each except stay 4 (10 h)
    stays = [(11, 1, 30), (12, 2, 30), (13, 3, 30), (14, 4, 10), (15, 5, 30)]
    pd.DataFrame(
        [{"subject_id": s, "stay_id": st, "intime": str(INTIME),
          "outtime": str(INTIME + pd.Timedelta(hours=float(h)))} for s, st, h in stays]
    ).to_csv(ext / "root" / "all_stays.csv", index=False)

    def img_row(dicom, subject, study, hours, view="AP"):
        t = INTIME + pd.Timedelta(hours=float(hours))
        return {"dicom_id": dicom, "subject_id": subject, "study_id": study,
                "StudyDate": int(t.strftime("%Y%m%d")),
                "StudyTime": float(t.strftime("%H%M%S")), "ViewPosition": view}

    pd.DataFrame([
        img_row("a_early", 11, 100, 2), img_row("a_late", 11, 101, 20),
        img_row("a_pa", 11, 102, 25, view="PA"), img_row("a_after", 11, 103, 40),
        img_row("b_nolabel", 12, 200, 5),   # study without a CheXpert row
        img_row("c", 13, 300, 5),           # stay not on a listfile
        img_row("d", 14, 400, 5),           # val stay
        img_row("e_missing", 15, 500, 5),   # resized file missing
    ]).to_csv(cxr / "mimic-cxr-2.0.0-metadata.csv", index=False)
    pd.DataFrame({"study_id": [100, 101, 102, 103, 300, 400, 500]}).to_csv(
        cxr / "mimic-cxr-2.0.0-chexpert.csv", index=False)
    header = "stay,period_length,stay_id,label\n"
    (ext / "phenotyping" / "train_listfile.csv").write_text(
        header + "e1.csv,30.0,1,0\ne2.csv,30.0,2,0\ne5.csv,30.0,5,0\n")
    (ext / "phenotyping" / "val_listfile.csv").write_text(header + "e4.csv,10.0,4,1\n")
    # stay 1: Hours 0 is dropped (0 < t), 31 is beyond the stay; stay 4: no rows
    _episode(ext / "phenotyping" / "train" / "e1.csv",
             [(0.0, "60"), (1.5, "70"), (29.0, "80"), (31.0, "90")])
    _episode(ext / "phenotyping" / "train" / "e2.csv", [(1.0, "70")])
    _episode(ext / "phenotyping" / "train" / "e4.csv", [(0.0, "70")])
    _episode(ext / "phenotyping" / "train" / "e5.csv", [(1.0, "70")])
    rng = np.random.default_rng(0)
    for dicom in ("a_early", "a_late", "a_pa", "a_after", "b_nolabel", "c", "d"):
        write_dummy_jpg(img / f"{dicom}.jpg", rng)
    return ext, cxr, img


def test_cohort_pairs_latest_ap_with_chexpert_on_listfiles(fake):
    ext, cxr, _ = fake
    cohort = pretraining_cohort(ext, cxr)
    got = sorted(cohort[["split", "stay_id", "dicom_id"]].values.tolist())
    assert got == [["train", 1, "a_late"], ["train", 5, "e_missing"], ["val", 4, "d"]]


def test_samples_follow_medmods_loader(fake):
    ext, cxr, img = fake
    samples, funnel = load_pretraining_samples(ext, cxr, img)
    assert funnel["train"]["image_missing"] == 1
    assert funnel["val"]["dummy_rows"] == 1  # stay 4 has only Hours == 0
    by_stay = {s["visit_id"]: s for s in samples}
    assert sorted(by_stay) == ["1", "4"]
    ehr = by_stay["1"]["ehr"]
    assert ehr.shape == (30, 76)  # int(30 + 1 - 1e-6) bins
    hr = Discretizer().channels.index("Heart Rate")
    col = Discretizer()._begin_pos[hr]
    assert ehr[1, col] == 70.0 and ehr[29, col] == 80.0  # 1.5 h, 29 h
    assert ehr[0, col] != 60.0  # the Hours == 0 row is dropped
    assert by_stay["4"]["ehr"].shape == (10, 76)
    assert by_stay["4"]["split"] == "val"


def test_dataset_batches_carry_lengths(fake):
    ext, cxr, img = fake
    samples, _ = load_pretraining_samples(ext, cxr, img)
    dataset = create_pretraining_dataset(samples)
    batch = next(iter(get_dataloader(dataset, batch_size=2, shuffle=False)))
    ehr, lengths = batch["ehr"]
    assert sorted(lengths.tolist()) == [10, 30]
    assert ehr.shape == (2, 30, 76)

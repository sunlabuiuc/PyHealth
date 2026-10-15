"""
tests/core/test_synthetic_ehr.py
================================

:class:`pyhealth.datasets.synthetic_ehr.SyntheticEHRDataset` 的单元测试。

被测结构回顾（见 ``pyhealth/datasets/synthetic_ehr.py``）：

    SyntheticEHRDataset(root)            # root 为目录或 csv 文件路径
        .patients                         # {patient_id: Patient}
        Patient.patient_id / Patient.metadata  # gender, age
        Patient.visits                    # [Visit, ...]（按 timestamp 升序）
        Visit.visit_id / Visit.timestamp / Visit.events / Visit.labs
        Event.event_type / Event.code（或 Event.lab_name + Event.value）

测试数据用标准库 ``csv`` 在 pytest 的 ``tmp_path`` 下现场生成，
不依赖任何仓库内的固定数据文件。
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pyhealth.datasets.synthetic_ehr import SyntheticEHRDataset


# --------------------------------------------------------------------------- #
# 公共工具
# --------------------------------------------------------------------------- #
COLUMNS = [
    "patient_id",
    "visit_id",
    "visit_date",
    "age",
    "gender",
    "diagnosis_code",
    "procedure_code",
    "drug_code",
    "lab_glucose",
    "lab_creatinine",
]


def _row(pid, vid, date, age=60, gender="M", diag="250.00",
         proc="89.52", drug="860975", glu=120.0, cre=1.0):
    """构造一条完整的就诊记录。"""
    return {
        "patient_id": pid,
        "visit_id": vid,
        "visit_date": date,
        "age": age,
        "gender": gender,
        "diagnosis_code": diag,
        "procedure_code": proc,
        "drug_code": drug,
        "lab_glucose": glu,
        "lab_creatinine": cre,
    }


def _write_csv(tmp_path, rows):
    """把若干行记录写入 ``tmp_path/synthetic_ehr.csv``。"""
    path = os.path.join(str(tmp_path), "synthetic_ehr.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _load(tmp_path):
    """按目录路径加载数据集（被测代码会自动拼接 synthetic_ehr.csv）。"""
    return SyntheticEHRDataset(root=str(tmp_path))


def _events_of_type(visit, event_type):
    """返回某个 visit 中指定 event_type 的事件列表。"""
    return [e for e in visit.events if e.event_type == event_type]


# --------------------------------------------------------------------------- #
# 测试用例
# --------------------------------------------------------------------------- #
def test_load_normal_csv(tmp_path):
    """构造 3 患者 × 各 2 次就诊，验证患者数=3、就诊总数=6。"""
    rows = []
    for i in range(1, 4):
        pid = f"P{i:03d}"
        for j in range(1, 3):
            rows.append(_row(pid, f"V{i}{j}", f"2022-0{i}-0{j}"))

    _write_csv(tmp_path, rows)
    ds = _load(tmp_path)

    assert len(ds.patients) == 3
    total_visits = sum(len(p.visits) for p in ds.patients.values())
    assert total_visits == 6


def test_patient_metadata(tmp_path):
    """验证 patient.metadata 正确存储 age 与 gender。"""
    rows = [
        _row("P001", "V001", "2022-01-01", age=72, gender="M"),
        _row("P002", "V001", "2022-01-02", age=58, gender="F"),
    ]
    _write_csv(tmp_path, rows)
    ds = _load(tmp_path)

    p1 = ds.get_patient("P001")
    p2 = ds.get_patient("P002")
    assert p1.metadata["age"] == 72
    assert p1.metadata["gender"] == "M"
    assert p2.metadata["age"] == 58
    assert p2.metadata["gender"] == "F"


def test_visit_timestamp_sorting(tmp_path):
    """同一患者 3 次就诊日期乱序输入，验证 visit 按日期升序排列。"""
    rows = [
        _row("P001", "V003", "2022-03-01"),
        _row("P001", "V001", "2022-01-01"),
        _row("P001", "V002", "2022-02-01"),
    ]
    _write_csv(tmp_path, rows)
    ds = _load(tmp_path)

    visits = ds.get_patient("P001").visits
    dates = [v.timestamp.strftime("%Y-%m-%d") for v in visits]
    assert dates == ["2022-01-01", "2022-02-01", "2022-03-01"]


def test_event_types_present(tmp_path):
    """验证一次 visit 中 diagnosis / procedure / prescription 三类事件均被生成。"""
    rows = [_row("P001", "V001", "2022-01-01")]
    _write_csv(tmp_path, rows)
    ds = _load(tmp_path)

    visit = ds.get_patient("P001").visits[0]
    types = {e.event_type for e in visit.events}
    assert {"diagnosis", "procedure", "prescription"} <= types


def test_lab_values(tmp_path):
    """验证 lab 事件携带正确的 lab_name 与数值 value，并写入 visit.labs。"""
    rows = [_row("P001", "V001", "2022-01-01", glu=185.5, cre=1.2)]
    _write_csv(tmp_path, rows)
    ds = _load(tmp_path)

    visit = ds.get_patient("P001").visits[0]
    lab_events = _events_of_type(visit, "lab")
    lab_map = {e.lab_name: e.value for e in lab_events}

    assert lab_map["glucose"] == 185.5
    assert lab_map["creatinine"] == 1.2
    # 同时确认 visit 级 labs 元数据
    assert visit.labs["glucose"] == 185.5
    assert visit.labs["creatinine"] == 1.2


def test_empty_csv(tmp_path):
    """空 CSV（仅表头）加载不崩溃，患者数与就诊数均为 0。"""
    _write_csv(tmp_path, [])  # 只有表头，无数据行
    ds = _load(tmp_path)

    assert len(ds.patients) == 0
    assert ds.unique_patient_ids == []


def test_multiple_visits_same_patient(tmp_path):
    """同一患者 3 次就诊，验证均归属同一 patient 且 visit_id 正确。"""
    rows = [
        _row("P001", "V001", "2022-01-01"),
        _row("P001", "V002", "2022-04-01"),
        _row("P001", "V003", "2022-07-01"),
    ]
    _write_csv(tmp_path, rows)
    ds = _load(tmp_path)

    assert len(ds.patients) == 1
    p = ds.get_patient("P001")
    assert len(p.visits) == 3
    assert [v.visit_id for v in p.visits] == ["V001", "V002", "V003"]

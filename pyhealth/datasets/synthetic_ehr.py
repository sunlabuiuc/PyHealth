"""
synthetic_ehr_dataset.py
=======================

PyHealth 自定义 CSV 数据集的 adapter（教学用）。

将 ``synthetic_ehr.csv`` 解析为 PyHealth 风格的
``Patient -> Visit -> Event`` 结构，事件类型包括：

    - diagnosis     (ICD-9,   ``|`` 分隔多码)
    - procedure     (ICD-9-CM,``|`` 分隔多码)
    - prescription  (NDC,     ``|`` 分隔多码)
    - lab           (数值型: glucose / creatinine, 存为 Event 的 value)

设计说明
--------
* PyHealth 2.x 中 ``Patient`` 改为 Polars DataFrame 驱动、``BaseDataset`` 改为
  YAML 配置驱动，已不存在 ``BaseEHRDataset`` 与 ``Visit`` 类。因此本 adapter
  在**尽量贴近 MIMIC3Dataset 的解析风格**（parse_basic_info + 按 patient 分组
  构造事件）的同时，自包含地实现了 ``Patient -> Visit -> Event`` 结构，并保证：

  1) 若安装了 PyHealth，``SyntheticEHRDataset`` 继承 ``pyhealth.datasets.BaseDataset``，
     同时保留 PyHealth 原生的 ``Patient``/``Event`` 作为底层事件对象；
  2) 若未安装 PyHealth，则回退到一个等价的轻量基类，代码仍可独立运行。

* lab 值既作为 ``Event``（event_type="lab"，属性含 ``lab_name`` 与 ``value``），
  也写入 visit 级 metadata（``visit.labs`` dict），便于下游按需取用。
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# 可选依赖：PyHealth 的 Patient / Event / BaseDataset
# 未安装时回退到本文件内定义的轻量实现，保证脚本可独立运行。
# --------------------------------------------------------------------------- #
try:
    from pyhealth.data import Event as PyHealthEvent
    from pyhealth.data import Patient as PyHealthPatient
    from pyhealth.datasets import BaseDataset as PyHealthBaseDataset

    _HAS_PYHEALTH = True
except Exception:  # pragma: no cover - 环境未装 PyHealth
    PyHealthEvent = None
    PyHealthPatient = None
    PyHealthBaseDataset = None
    _HAS_PYHEALTH = False


# --------------------------------------------------------------------------- #
# 轻量事件类（无 PyHealth 时使用）
# --------------------------------------------------------------------------- #
class Event:
    """单个临床事件：event_type + timestamp + 任意属性。"""

    __slots__ = ("event_type", "timestamp", "attr_dict")

    def __init__(self, event_type: str, timestamp: Optional[datetime] = None, **kwargs):
        self.event_type = event_type
        self.timestamp = timestamp
        self.attr_dict = dict(kwargs)

    def __getattr__(self, name):
        if name in self.attr_dict:
            return self.attr_dict[name]
        raise AttributeError(name)

    def __getitem__(self, key):
        return self.attr_dict[key]

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.attr_dict.items())
        return f"Event({self.event_type!r}, {self.timestamp}, {attrs})"


# 统一事件类：有 PyHealth 用其原生 Event，否则用上面的轻量实现。
_EventCls = PyHealthEvent if _HAS_PYHEALTH else Event


class Visit:
    """一次就诊，聚合该次就诊的所有 Event。"""

    def __init__(self, visit_id: str, timestamp: datetime):
        self.visit_id = visit_id
        self.timestamp = timestamp
        self.events: List = []
        self.metadata: Dict = {}  # age, gender 等 visit 级字段
        self.labs: Dict[str, float] = {}  # lab 值也可作为 visit 级 metadata

    def add_event(self, event) -> None:
        self.events.append(event)

    def __repr__(self):
        return f"Visit({self.visit_id!r}, {self.timestamp.date()}, {len(self.events)} events)"


class Patient:
    """一个患者，聚合其所有 Visit。"""

    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.visits: List[Visit] = []
        self.metadata: Dict = {}

    def add_visit(self, visit: Visit) -> None:
        self.visits.append(visit)

    def __repr__(self):
        return f"Patient({self.patient_id!r}, {len(self.visits)} visits)"


# --------------------------------------------------------------------------- #
# 基类（有 PyHealth 时继承其 BaseDataset，否则用轻量基类）
# --------------------------------------------------------------------------- #
if _HAS_PYHEALTH:

    class _BaseDataset(PyHealthBaseDataset):
        pass

else:

    class _BaseDataset:
        """轻量基类，仅提供与 BaseDataset 同名的最小组件。"""

        def __init__(self, *args, **kwargs):
            self.patients: Dict[str, Patient] = {}
            self._unique_patient_ids: List[str] = []

        @property
        def unique_patient_ids(self) -> List[str]:
            return list(self._unique_patient_ids)

        def get_patient(self, patient_id: str) -> Patient:
            if patient_id not in self.patients:
                raise KeyError(f"patient {patient_id} not found")
            return self.patients[patient_id]

        def iter_patients(self):
            for pid in self._unique_patient_ids:
                yield self.patients[pid]


# --------------------------------------------------------------------------- #
# 数据集 adapter
# --------------------------------------------------------------------------- #
class SyntheticEHRDataset(_BaseDataset):
    """读取 ``synthetic_ehr.csv`` 并构造 Patient -> Visit -> Event 结构。

    Parameters
    ----------
    root : str | Path
        数据集文件（CSV）所在目录；若传入的是文件路径则直接使用该文件。
    dataset_name : str, optional
        数据集名称，默认为 "synthetic_ehr"。
    """

    # 列名到事件类型的映射（lab 两列合并为同一 event_type="lab"）
    EVENT_TYPE_MAP = {
        "diagnosis_code": "diagnosis",
        "procedure_code": "procedure",
        "drug_code": "prescription",
        "lab_glucose": "lab",
        "lab_creatinine": "lab",
    }

    LAB_NAMES = {"lab_glucose": "glucose", "lab_creatinine": "creatinine"}

    def __init__(
        self,
        root: str | Path,
        dataset_name: str = "synthetic_ehr",
        **kwargs,
    ):
        self.root = Path(root)
        # 兼容：root 既可指向目录也可指向 csv 文件
        self.csv_path = self.root if self.root.is_file() else self.root / "synthetic_ehr.csv"
        if not self.csv_path.exists():
            raise FileNotFoundError(f"未找到数据文件: {self.csv_path}")

        self.dataset_name = dataset_name

        if _HAS_PYHEALTH:
            # 用 PyHealth BaseDataset 的最小初始化（不触发其 YAML 加载逻辑）。
            # 我们自行构造 patient 树，因此直接跳过 BaseDataset 的 load_data 机制。
            PyHealthBaseDataset.__init__(
                self,
                root=str(self.root),
                tables=[],  # 不使用原生 table 加载
                dataset_name=dataset_name,
                **kwargs,
            )
        else:
            super().__init__(root=str(self.root), dataset_name=dataset_name, **kwargs)

        # 核心：解析 CSV 并构造 Patient -> Visit -> Event
        self.patients: Dict[str, Patient] = {}
        self._unique_patient_ids: List[str] = []
        self.parse_basic_info()

        self._print_stats()

    # ------------------------------------------------------------------ #
    # 解析主流程
    # ------------------------------------------------------------------ #
    def parse_basic_info(self) -> None:
        """读取 CSV，按 patient_id 分组，构造 Patient -> Visit -> Event。"""
        rows = self._read_rows()

        for row in rows:
            pid = row.get("patient_id", "").strip()
            if not pid:
                continue  # 空 patient_id 跳过

            visit = self._build_visit(row)
            if visit is None:
                continue  # 无效行（如日期无法解析）跳过

            patient = self.patients.get(pid)
            if patient is None:
                patient = Patient(pid)
                patient.metadata["gender"] = self._safe(row, "gender")
                patient.metadata["age"] = self._to_int(row.get("age"))
                self.patients[pid] = patient
                self._unique_patient_ids.append(pid)

            patient.add_visit(visit)

        # 每个患者的 visit 按时间排序
        for patient in self.patients.values():
            patient.visits.sort(key=lambda v: v.timestamp)

    def _read_rows(self) -> List[dict]:
        """读取 CSV 为 dict 列表（空行/空字段按原样保留，由调用方处理）。"""
        with open(self.csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return [row for row in reader]

    def _build_visit(self, row: dict) -> Optional[Visit]:
        """将一行 CSV 解析为一次 Visit，含 diagnosis/procedure/prescription/lab 事件。"""
        visit_id = self._safe(row, "visit_id")
        ts = self._parse_date(row.get("visit_date"))
        if ts is None:
            return None

        visit = Visit(visit_id, ts)
        visit.metadata["age"] = self._to_int(row.get("age"))
        visit.metadata["gender"] = self._safe(row, "gender")

        for col, event_type in self.EVENT_TYPE_MAP.items():
            if event_type == "lab":
                # lab 两列分别作为一个 Event，value 存数值
                self._add_lab_event(visit, col, row.get(col), ts)
            else:
                self._add_code_events(visit, col, event_type, row.get(col), ts)

        return visit

    def _add_code_events(self, visit: Visit, col: str, event_type: str,
                         raw: Optional[str], ts: datetime) -> None:
        """拆分 ``|`` 分隔的多码，逐码生成一个 Event；空值跳过。"""
        if raw is None:
            return
        codes = [c.strip() for c in str(raw).split("|") if c.strip()]
        for code in codes:
            event = _EventCls(event_type, timestamp=ts, code=code)
            visit.add_event(event)

    def _add_lab_event(self, visit: Visit, col: str, raw: Optional[str],
                       ts: datetime) -> None:
        """lab 值作为数值型 Event，同时写入 visit.labs；空值/非法值跳过。"""
        val = self._to_float(raw)
        if val is None:
            return
        lab_name = self.LAB_NAMES[col]
        event = _EventCls("lab", timestamp=ts, lab_name=lab_name, value=val)
        visit.add_event(event)
        visit.labs[lab_name] = val

    # ------------------------------------------------------------------ #
    # 工具函数
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_date(raw) -> Optional[datetime]:
        if raw is None:
            return None
        raw = str(raw).strip()
        if not raw:
            return None
        try:
            return datetime.strptime(raw, "%Y-%m-%d")
        except ValueError:
            return None

    @staticmethod
    def _safe(row: dict, key: str) -> str:
        v = row.get(key)
        return "" if v is None else str(v).strip()

    @staticmethod
    def _to_int(raw) -> Optional[int]:
        try:
            return int(float(str(raw).strip()))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_float(raw) -> Optional[float]:
        try:
            return float(str(raw).strip())
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------ #
    # 统计与查询
    # ------------------------------------------------------------------ #
    def _print_stats(self) -> None:
        """打印数据集统计：患者数、就诊总数、诊断码种类数、药物码种类数。"""
        n_patients = len(self._unique_patient_ids)
        n_visits = sum(len(p.visits) for p in self.patients.values())

        diag_codes: set = set()
        drug_codes: set = set()
        for p in self.patients.values():
            for v in p.visits:
                for e in v.events:
                    if e.event_type == "diagnosis":
                        diag_codes.add(e.code)
                    elif e.event_type == "prescription":
                        drug_codes.add(e.code)

        print("=" * 50)
        print(f"[SyntheticEHRDataset] {self.dataset_name}")
        print(f"  患者数:       {n_patients}")
        print(f"  就诊总数:     {n_visits}")
        print(f"  诊断码种类数: {len(diag_codes)}")
        print(f"  药物码种类数: {len(drug_codes)}")
        print("=" * 50)

    def stats(self) -> None:
        """兼容 BaseDataset.stats() 接口。"""
        self._print_stats()

    def describe_patient(self, patient_id: str) -> str:
        """返回单个患者结构的人类可读文本，便于测试块打印。"""
        p = self.get_patient(patient_id)
        lines = [f"患者 {p.patient_id}  (gender={p.metadata.get('gender')}, "
                 f"age={p.metadata.get('age')}, visits={len(p.visits)})"]
        for v in p.visits:
            lines.append(f"  ├─ {v}")
            for e in v.events:
                if e.event_type == "lab":
                    lines.append(f"  │    └─ {e.event_type:12s} {e.lab_name:<10s} = {e.value}")
                else:
                    lines.append(f"  │    └─ {e.event_type:12s} {e.code}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 测试块
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    # 兼容：既可从命令行传路径，也可默认当前目录
    root = sys.argv[1] if len(sys.argv) > 1 else "."

    ds = SyntheticEHRDataset(root=root)

    print()
    print("患者 ID 列表:", ds.unique_patient_ids)
    print()

    # 打印第一个患者的结构
    first_pid = ds.unique_patient_ids[0]
    print(ds.describe_patient(first_pid))

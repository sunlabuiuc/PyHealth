"""
生成教学用合成 EHR 数据集 (PyHealth)

字段:
    patient_id, visit_id, visit_date, age, gender,
    diagnosis_code (ICD-9), procedure_code (ICD-9-CM),
    drug_code (NDC), lab_glucose, lab_creatinine

临床合理性设计:
    - 每个患者被赋予一个"主诊断"(primary condition)，
      该诊断决定其用药与实验室值的倾向。
    - 糖尿病患者 (250.00): 高血糖倾向 + 更可能用二甲双胍(860975)
    - 高血压/冠心病/心衰患者: 更可能用赖诺普利(003127)、阿司匹林(003127)、阿托伐他汀(551110)
    - 体检患者 (V70.0): 实验室值接近正常，通常不用药
"""

import random
import csv
from datetime import date, timedelta

random.seed(42)  # 可复现

# ---- 诊断码与药物码定义 ----
DIAGNOSES = {
    "250.00": "糖尿病",
    "401.9": "高血压",
    "414.01": "冠心病",
    "428.0": "心衰",
    "530.81": "GERD",
    "V70.0": "体检",
}

DRUGS = {
    "860975": "二甲双胍",       # metformin
    "003127": "赖诺普利",       # lisinopril
    "551110": "阿托伐他汀",     # atorvastatin
    "504580": "阿司匹林",       # aspirin
}

# ---- 每位患者的主诊断, 用于指导临床合理性 ----
# 10 位患者: 4 糖尿病, 2 高血压, 1 冠心病, 1 心衰, 1 GERD, 1 体检
PRIMARY_CONDITIONS = [
    "250.00", "250.00", "250.00", "250.00",  # 4 糖尿病
    "401.9", "401.9",                          # 2 高血压
    "414.01",                                 # 1 冠心病
    "428.0",                                  # 1 心衰
    "530.81",                                 # 1 GERD
    "V70.0",                                  # 1 体检
]


def pick_primary_diagnosis(condition: str) -> str:
    """根据主诊断返回本次就诊的诊断码, 有时会附带并发症诊断。"""
    return condition


def pick_secondary_diagnoses(condition: str) -> list:
    """返回可能伴随的诊断码列表(不含主诊断)。"""
    secondary = []
    if condition == "250.00" and random.random() < 0.5:
        secondary.append("401.9")  # 糖尿病常合并高血压
    if condition == "414.01" and random.random() < 0.6:
        secondary.append("401.9")  # 冠心病常合并高血压
    if condition == "428.0" and random.random() < 0.5:
        secondary.append("401.9")
    if condition == "401.9" and random.random() < 0.3:
        secondary.append("250.00")
    return secondary


def pick_drugs(condition: str) -> list:
    """根据主诊断选择药物, 体现临床合理性。"""
    drugs = []
    if condition == "250.00":
        if random.random() < 0.9:
            drugs.append("860975")          # 二甲双胍
        if random.random() < 0.4:
            drugs.append("003127")          # 赖诺普利 (合并高血压时)
    elif condition == "401.9":
        if random.random() < 0.9:
            drugs.append("003127")          # 赖诺普利
        if random.random() < 0.4:
            drugs.append("551110")          # 阿托伐他汀
    elif condition == "414.01":
        if random.random() < 0.8:
            drugs.append("003127")          # 阿司匹林
        if random.random() < 0.7:
            drugs.append("551110")          # 阿托伐他汀
    elif condition == "428.0":
        if random.random() < 0.8:
            drugs.append("003127")          # 赖诺普利
    elif condition == "530.81":
        if random.random() < 0.5:
            pass  # GERD 有时不用所列药物
    # V70.0 体检通常不用药
    if not drugs:
        # 保证非空, 给一个低概率安慰用药或留空由调用方处理
        if random.random() < 0.3:
            drugs.append("551110")
    return drugs


def gen_glucose(condition: str) -> float:
    """根据诊断生成合理血糖值。"""
    if condition == "250.00":
        # 糖尿病: 偏高 (110-300)
        return round(random.uniform(110, 300), 1)
    else:
        # 非糖尿病: 正常偏低 (70-130)
        return round(random.uniform(70, 130), 1)


def gen_creatinine(condition: str) -> float:
    """根据诊断生成合理肌酐值 (心衰/糖尿病肾功能略差)。"""
    base = random.uniform(0.5, 1.2)
    if condition == "428.0":
        base = random.uniform(0.9, 2.5)   # 心衰肾功能可能受损
    elif condition == "250.00":
        base = random.uniform(0.6, 2.0)   # 糖尿病肾病风险
    return round(min(base, 3.0), 2)


def gen_procedure(condition: str) -> str:
    """简单流程码: 用诊断码简化替代 ICD-9-CM 流程 (教学用途)。"""
    # 常见配套检查流程 (ICD-9-CM 示例)
    procedures = {
        "250.00": "89.52",   # 心电图? 此处仅为示例码
        "401.9": "89.65",    # 动脉血气/血压监测示例
        "414.01": "88.72",   # 诊断性心导管
        "428.0": "89.52",    # 心电图
        "530.81": "87.62",   # 上消化道造影示例
        "V70.0": "89.01",    # 常规体检问诊
    }
    return procedures[condition]


rows = []
visit_id_counter = 1

for pid_idx, condition in enumerate(PRIMARY_CONDITIONS, start=1):
    patient_id = f"P{pid_idx:03d}"
    age = random.randint(30, 85)
    gender = "M" if pid_idx <= 5 else "F"  # 前 5 位 M, 后 5 位 F

    n_visits = random.randint(1, 4)

    # 就诊日期: 在 2022-01 ~ 2023-12 内递增分布
    start = date(2022, 1, 1)
    end = date(2023, 12, 31)
    total_days = (end - start).days

    visit_dates = []
    # 生成 n_visits 个递增日期
    offsets = sorted(random.sample(range(total_days + 1), n_visits))
    for off in offsets:
        visit_dates.append(start + timedelta(days=off))

    for vi in range(n_visits):
        visit_id = f"V{visit_id_counter:04d}"
        visit_id_counter += 1

        # 诊断码: 主诊断 + 可能的伴随诊断, 用 '|' 分隔
        diag_list = [pick_primary_diagnosis(condition)]
        diag_list += pick_secondary_diagnoses(condition)
        # 去重保持顺序
        seen = set()
        diag_codes = []
        for d in diag_list:
            if d not in seen:
                seen.add(d)
                diag_codes.append(d)
        diagnosis_code = "|".join(diag_codes)

        drugs = pick_drugs(condition)
        drug_code = "|".join(drugs) if drugs else ""

        rows.append({
            "patient_id": patient_id,
            "visit_id": visit_id,
            "visit_date": visit_dates[vi].isoformat(),
            "age": age,
            "gender": gender,
            "diagnosis_code": diagnosis_code,
            "procedure_code": gen_procedure(condition),
            "drug_code": drug_code,
            "lab_glucose": gen_glucose(condition),
            "lab_creatinine": gen_creatinine(condition),
        })

# ---- 保存 CSV ----
fieldnames = [
    "patient_id", "visit_id", "visit_date", "age", "gender",
    "diagnosis_code", "procedure_code", "drug_code",
    "lab_glucose", "lab_creatinine",
]

with open("synthetic_ehr.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# ---- 打印前 10 行确认 ----
print("生成完成: synthetic_ehr.csv")
print("总记录数:", len(rows))
print("-" * 100)
print(",".join(fieldnames))
for r in rows[:10]:
    print(",".join(str(r[k]) for k in fieldnames))

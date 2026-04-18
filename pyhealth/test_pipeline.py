from pyhealth.datasets import MIMIC3CirculatoryFailureDataset
from pyhealth.tasks import CirculatoryFailurePredictionTask


def main():
    # 1. 初始化資料集（請確保路徑正確）
    dataset = MIMIC3CirculatoryFailureDataset(
        root="/mimic_test"
    )

    # 2. 初始化任務（12小時預警）
    task = CirculatoryFailurePredictionTask(prediction_window_hours=12)

    # 3. 讀取 cohort，找到第一個真的能產生 samples 的 ICU stay
    cohort = dataset.load_cohort()

    samples = None
    chosen_icustay_id = None

    for row in cohort:
        icustay_id = row["icustay_id"]
        patient = dataset.get_patient_by_icustay_id(icustay_id)
        samples = task(patient)

        if samples:
            chosen_icustay_id = icustay_id
            break

    # 4. 檢查結果
    if samples:
        print(f"--- 成功測試 ICU Stay ID: {chosen_icustay_id} ---")
        print(f"成功產生樣本數: {len(samples)}")
        print(
            f"其中 Label=1 (未來12小時內會衰竭) 的數量: "
            f"{sum(s['label'] for s in samples)}"
        )
        print(f"第一筆樣本特徵: {samples[0]['features']}")
        print(f"第一筆完整樣本: {samples[0]}")
    else:
        print("未找到任何可產生樣本的 ICU stay，請檢查資料與路徑。")


if __name__ == "__main__":
    main()
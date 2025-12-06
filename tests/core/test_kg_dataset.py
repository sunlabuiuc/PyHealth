import os
import pandas as pd
from pyhealth.datasets.custom_radiology import CustomRadiologyDataset


def test_custom_radiology_dataset(tmp_path):
    # Create a temporary CSV
    csv_path = tmp_path / "mimic_reports.csv"
    df = pd.DataFrame({
        "report_id": ["1", "2"],
        "report": ["Normal chest radiograph.", "Left lower lobe opacity."]
    })
    df.to_csv(csv_path, index=False)

    dataset = CustomRadiologyDataset(root=str(tmp_path), csv_name="mimic_reports.csv")

    # dataset should load both samples
    assert len(dataset) == 2
    sample = dataset[0]

    assert "patient_id" in sample
    assert "visit_id" in sample
    assert "text" in sample

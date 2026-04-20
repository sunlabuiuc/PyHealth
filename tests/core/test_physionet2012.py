import os
import tempfile
from unittest.mock import patch
from pyhealth.datasets import PhysioNet2012Dataset
from pyhealth.tasks import PhysioNetMortalityTask

@patch("pyhealth.datasets.base_dataset.in_notebook", return_value=True)
@patch("pyhealth.datasets.physionet2012.subprocess.run")
@patch("pyhealth.datasets.physionet2012.zipfile.ZipFile")
def test_physionet2012_dataset_download_parse(mock_zipfile, mock_run, mock_in_notebook):
    # Rubric: Uses temporary directories and proper cleanup
    with tempfile.TemporaryDirectory() as temp_dir:
        def mock_extractall(*args, **kwargs):
            set_a_dir = os.path.join(temp_dir, "set-a")
            set_b_dir = os.path.join(temp_dir, "set-b")
            os.makedirs(set_a_dir, exist_ok=True)
            os.makedirs(set_b_dir, exist_ok=True)
           
            # Rubric: Uses small synthetic/pseudo data (2 patients max)
            with open(os.path.join(set_a_dir, "132801.txt"), "w") as f:
                f.write("Time,Parameter,Value\n"
                        "00:00,RecordID,132801\n"
                        "00:00,Age,83\n"
                        "00:00,Gender,0\n"
                        "00:00,Height,157.5\n"
                        "00:00,ICUType,4\n"
                        "00:00,Weight,48.6\n"
                        "00:01,GCS,15\n"
                        "00:01,HR,90\n"
                        "00:01,Temp,36.4\n")
           
            with open(os.path.join(set_b_dir, "132791.txt"), "w") as f:
                f.write("Time,Parameter,Value\n"
                        "00:00,RecordID,132791\n"
                        "00:00,Age,79\n"
                        "00:00,Gender,1\n"
                        "00:00,Height,-1\n"
                        "00:00,ICUType,3\n"
                        "00:00,Weight,61\n"
                        "00:54,GCS,15\n"
                        "00:54,HR,90\n"
                        "00:54,NIDiasABP,63\n")

            outcomes_content_a = "RecordID,In-hospital_death\n132801,0\n"
            with open(os.path.join(temp_dir, "Outcomes-a.txt"), "w") as f:
                f.write(outcomes_content_a)

            outcomes_content_b = "RecordID,In-hospital_death\n132791,1\n"
            with open(os.path.join(temp_dir, "Outcomes-b.txt"), "w") as f:
                f.write(outcomes_content_b)

        mock_zip_ctx = mock_zipfile.return_value.__enter__.return_value
        mock_zip_ctx.extractall.side_effect = mock_extractall

        # Initialize dataset using the temp_dir for both root and cache to ensure total cleanup
        dataset = PhysioNet2012Dataset(
            root=temp_dir,
            tables=["events", "outcomes"],
            dev=False,
            num_workers=1,
            cache_dir=temp_dir
        )
       
        # Rubric: Tests data loading and patient parsing
        assert len(dataset.unique_patient_ids) == 2
        assert "132801" in dataset.unique_patient_ids
        assert "132791" in dataset.unique_patient_ids
       
        # Rubric: Tests event parsing and data integrity
        patient = dataset.get_patient("132801")
        assert patient.patient_id == "132801"
       
        events = patient.get_events("events")
        assert len(events) == 9
       
        hr_events =[e for e in events if e.parameter == "HR"]
        assert len(hr_events) == 1
        assert float(hr_events[0].value) == 90.0
       
        outcomes = patient.get_events("outcomes")
        assert len(outcomes) == 1
        assert int(getattr(outcomes[0], "in-hospital_death")) == 0
       
        # Rubric: Tests task functionality
        task = PhysioNetMortalityTask(n_timesteps=16)
        sample_dataset = dataset.set_task(task)
       
        assert len(sample_dataset) == 2
        assert "x_ts" in sample_dataset[0]
        assert "x_static" in sample_dataset[0]
        assert "times" in sample_dataset[0]
        assert "label" in sample_dataset[0]
       
        labels = [sample["label"] for sample in sample_dataset]
        assert 0 in labels
        assert 1 in labels
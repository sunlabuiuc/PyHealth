from pyhealth.datasets.mimic3 import MIMIC3NursingNotesDataset

if __name__ == "__main__":
    dataset_text = MIMIC3NursingNotesDataset(root="physionet.org/files/deidentifiedmedicaltext/1.0", nursing_notes_filename="id.text")

    print("### Origin data without masks ###")
    print(dataset_text.text_records[0])
    print("### Notes with mask ###")
    print(dataset_text.res_records[0])
    print("### Masks information ###")
    print(dataset_text.masks[0])
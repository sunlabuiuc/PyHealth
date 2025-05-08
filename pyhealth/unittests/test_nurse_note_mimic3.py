from pyhealth.datasets.mimic3 import MIMIC3NursingNotesDataset

if __name__ == "__main__":
    dataset = MIMIC3NursingNotesDataset(root="physionet.org/files/deidentifiedmedicaltext/1.0", nursing_notes_filename="id.text")
    if len(dataset.records) == 0:
        print("No data found")
        print("Please download data from http://physionet.org/content/deidentifiedmedicaltext/1.0/")
    else:
        print("### Origin data without masks ###")
        print(dataset.records[0].text_record)
        print("### Notes with mask ###")
        print(dataset.records[0].res_record)
        print("### Masks information ###")
        print(dataset.records[0].mask_info)
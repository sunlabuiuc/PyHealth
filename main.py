from pyhealth.datasets import TUSZDataset
from pyhealth.tasks import TUSZTask
from pyhealth.sampler import TUSZSampler
import torch
from torch.utils.data import DataLoader

BATCH_SIZE = 32
LABEL_GROUP = 'label'

def eeg_collate_fn(samples):
    batch = []

    for sample in samples:
        signals = sample['signal']
        labels = sample[LABEL_GROUP]
        batch.append((signals, labels))

    batch_size = len(batch)
    max_seq_len = max(s[0].shape[1] for s in batch)
    num_channels = batch[0][0].shape[0]
    max_label_len = max(len(s[1]) for s in batch)

    seqs = torch.zeros(batch_size, max_seq_len, num_channels)
    targets = torch.zeros(batch_size, max_label_len, dtype=torch.long)

    seq_lengths = []
    target_lengths = []
    for i, (signals, labels) in enumerate(batch):
        seq_len = signals.shape[1]

        signals = signals.permute(1, 0)

        seqs[i, :seq_len] = signals
        targets[i, :len(labels)] = torch.tensor(labels)

        seq_lengths.append(seq_len)
        target_lengths.append(len(labels))

    return seqs, targets, seq_lengths, target_lengths


if __name__ == "__main__":
    train_dataset = TUSZDataset(
        root = '../datasets/tuh_eeg_v2.0.5',
        subset = 'dev'
    )

    # # get patients
    # patient_ids = dataset.unique_patient_ids
    # for patient_id in patient_ids:
    #     patient = dataset.get_patient(patient_id)
    #     print(f"Patient ID: {patient_id}")
    #     for i, event in enumerate(patient.get_events()):
    #         if i < 2:
    #             print(event)
    #         else:
    #             break

    # # get events
    # eval_events = dataset.get_patient('aaaaaaaq').get_events("eval")
    # for eval_event in eval_events:
    #     # print(eval_event['event_type'])
    #     # print(eval_event['timestamp'])
    #     # print(eval_event['signal_file'])
    #     print(eval_event['record_id'])

    print(train_dataset.stats())

    task = TUSZTask() # dataset.default_task
    samples = train_dataset.set_task(task)
    sampler = TUSZSampler()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
        sampler=sampler, # only in train dataset
        collate_fn=eeg_collate_fn,
        shuffle = True, # True in train and val/dev dataset
    )
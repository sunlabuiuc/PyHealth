from pyhealth.datasets import TUSZDataset, TUSZSamplerDataset
from pyhealth.tasks import TUSZTask
from pyhealth.sampler import TUSZSampler
import torch
from torch.utils.data import DataLoader
from pyhealth.datasets import get_dataloader

BATCH_SIZE = 32
LABEL_GROUP = 'label'

def eeg_collate_fn(samples):
    batch = []
    for sample in samples:
        signals = sample['signal']
        labels = sample[LABEL_GROUP]
        batch.append((signals, labels))

    batch_size = len(batch)
    max_seq_len = max(s[0].shape[1] for s in batch) # 6000
    num_channels = batch[0][0].shape[0]             # 20
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

    ##################################################
    # DATASET
    ##################################################
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


    ##################################################
    # TASK
    ##################################################
    task = train_dataset.default_task      # TUSZTask()
    sample_dataset = train_dataset.set_task(task)


    ##################################################
    # DATALOADER
    ##################################################
    # option 1: dataloader with sampler
    sampler_dataset = TUSZSamplerDataset(
        dataset=sample_dataset,
        is_training_set=True,
        buffer_size=32
    )
    train_loader = DataLoader(
        sampler_dataset,
        batch_size=BATCH_SIZE,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=eeg_collate_fn,
        shuffle = False, # True in train and val/dev dataset
    )
    
    # option 2: dataloader without sampler
    # train_loader = get_dataloader(sample_dataset, batch_size=BATCH_SIZE, shuffle=False)


    ##################################################
    # DESCRIBE
    ##################################################
    first_batch = next(iter(train_loader))

    def describe(value):
        if hasattr(value, 'shape'):
            return f"{type(value).__name__}(shape={tuple(value.shape)})"
        if isinstance(value, (list, tuple)):
            return f"{type(value).__name__}(len={len(value)})"
        return type(value).__name__

    x, y, seq_lengths, target_lengths = first_batch
    print('Batch structure:')
    print(f"  x: {describe(x)}")
    print(f"  y: {describe(y)}")
    print(f"  seq_lengths: {describe(seq_lengths)}")
    print(f"  target_lengths: {describe(target_lengths)}")


    ##################################################
    # LOOP
    ##################################################
    for train_batch in train_loader:
        train_x, train_y, seq_lengths, target_lengths = train_batch
        print('Batch structure:')
        print(f"  x: {describe(x)}")
        print(f"  y: {describe(y)}")
        print(f"  seq_lengths: {describe(seq_lengths)}")
        print(f"  target_lengths: {describe(target_lengths)}")
        break

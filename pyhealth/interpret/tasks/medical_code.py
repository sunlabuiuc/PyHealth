import torch
import torch.nn as nn
import torch.nn.functional as F
from pyhealth.models import Transformer
from pyhealth.interpret.tasks.methods.chefer import CheferRelevance




class MedicalCodeInterpreter():
    def __init__(self, model):
        self.model = model 
        self.relevance= CheferRelevance(model)   

    def scale_rel_scores(self, input, out_length, method="", vis = False):
        # interpolate such that we can get a bigger 1D vector of weights or scores for each chunk
        # since attn is a square matrix, our temporal grad attn is a vector
        # since we know that sequence is C x L, divide and ceiling it to recreate the "convolved zones" 
        attribution_scores = None
        input = input.unsqueeze(1) # so we can get B x 1 x T dimensions for interpolation of temporal attention
        if method == "linear":
            attribution_scores = F.interpolate(input, size=out_length, mode="linear")    
        else: # naive interpolation where every chunk is simply just what we want
            attribution_scores = F.interpolate(input, size=out_length, mode="nearest")
        
        # attribution_scores = attribution_scores.squeeze().squeeze()# squeeze it back to a normal dim
        if vis:
            attribution_scores = attribution_scores.detach().cpu().numpy()

        return attribution_scores


    # input is PyHealth data sample with respective keys.

    def get_cam(self, input, class_index= None, method = "linear"):
        if class_index != None:
            if not isinstance(class_index, torch.Tensor):
                class_index = torch.tensor(class_index)
            if len(class_index.size()) < 1:
                class_index = class_index.unsqueeze(0)
        channel_length = sequence.size()[2]
        n_channels = sequence.size()[1]
        sequence = sequence.to(self.device)
        # transformer_attribution = torch.zeros(sequence.size())
        transformer_attribution = self.get_relevance_matrix(sequence, index=class_index).detach()
        transformer_attribution = self.scale_rel_scores(transformer_attribution, channel_length, method=method)
        # want to make sure it matches the sequence dimensions!

        transformer_attribution = transformer_attribution.repeat(1, n_channels, 1)
        
        # get channel attention
        channel_attribution = self.get_channel_relevance(sequence, index=class_index)
        
        # channel_attribution = torch.zeros(sequence.size())
        channel_attribution = channel_attribution.unsqueeze(-1)# reshape into columnwise addition

        # combine channel attribution to transformer attribution across each channel.
        transformer_attribution += channel_attribution
        # normalize again
        transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
        self.model.eval() # disable autograd again after done with gradient computations
        return transformer_attribution 


    # get visualization and convert to numpy while doing it
    def visualize(self, input):
        return None 
    

if __name__ == "__main__":
    print("HELLO")

    from pyhealth.datasets import MIMIC3Dataset

    mimic3_ds = MIMIC3Dataset(
            root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            dev=True,
    )

    print (mimic3_ds.stat())
    # data format
    mimic3_ds.info()
    from pyhealth.tasks import length_of_stay_prediction_mimic3_fn

    mimic3_ds = mimic3_ds.set_task(task_fn=length_of_stay_prediction_mimic3_fn)
    # stats info
    print (mimic3_ds.stat())


    {
        "patient_id": "p001",
        "visit_id": "v001",
        "diagnoses": [...],
        "labs": [...],
        "procedures": [...],
        "label": 1,
    }

    from pyhealth.datasets.splitter import split_by_patient
    from pyhealth.datasets import split_by_patient, get_dataloader

    # data split
    train_dataset, val_dataset, test_dataset = split_by_patient(mimic3_ds, [0.8, 0.1, 0.1])

    # create dataloaders (they are <torch.data.DataLoader> object)
    train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)
    mimic3_ds.samples[0].keys()

    from pyhealth.models import Transformer
    model = Transformer(
            dataset=mimic3_ds,
            # look up what are available for "feature_keys" and "label_keys" in dataset.samples[0]
            feature_keys=["conditions", "procedures", "drugs"],
            label_key="label",
            mode="multiclass",
        )
    
    print("Testing MIMIC3 STUFF")
    sample = test_loader.dataset[0]

    print(sample)

    print("----")
    print(model)
    # exit(0)
    from pyhealth.trainer import Trainer

    trainer = Trainer(
        model=model,
        metrics=["accuracy", "f1_weighted"], # the metrics that we want to log
        )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=30,
        monitor="accuracy",
        monitor_criterion="max",
    )
    data_iterator = iter(test_loader)
    data = next(data_iterator)
    print(data)
    model(**data)

    relevance = CheferRelevance(model)
    # returns a list ofr now
    # interpretability code here!
    rel_scores = relevance.get_relevance_matrix(**data)

    # weigh and plot these scores and their corresponding feature list
    print(rel_scores)
    

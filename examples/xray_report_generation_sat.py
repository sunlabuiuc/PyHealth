import os
import argparse
import pickle
import torch
from collections import OrderedDict
from torchvision import transforms
from pyhealth.datasets import BaseImageCaptionDataset
from pyhealth.tasks.xray_report_generation import biview_multisent_fn
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.tokenizer import Tokenizer
from pyhealth.models import WordSAT, SentSAT
from pyhealth.trainer import Trainer

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, default=".")
    parser.add_argument('--encoder-chkpt-fname', type=str, default=None)
    parser.add_argument('--tokenizer-fname', type=str, default=None)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--model-type', type=str, default="wordsat")

    args = parser.parse_args()
    return args

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# STEP 1: load data
def load_data(root):
    base_dataset = BaseImageCaptionDataset(root=root,dataset_name='IU_XRay')
    return base_dataset

# STEP 2: set task
def set_task(base_dataset):
    sample_dataset = base_dataset.set_task(biview_multisent_fn)
    transform = transforms.Compose([
                transforms.RandomAffine(degrees=30),
                transforms.Resize((512,512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],
                                std=[0.229,0.224,0.225]),
            ])
    sample_dataset.set_transform(transform)
    return sample_dataset

# STEP 3: get dataloaders
def get_dataloaders(sample_dataset):
    train_dataset, val_dataset, test_dataset = split_by_patient(
                                                  sample_dataset,[0.8,0.1,0.1])
    
    train_dataloader = get_dataloader(train_dataset,batch_size=8,shuffle=True)
    val_dataloader = get_dataloader(val_dataset,batch_size=1,shuffle=False)
    test_dataloader = get_dataloader(test_dataset,batch_size=1,shuffle=False)
    
    return train_dataloader,val_dataloader,test_dataloader

# STEP 4: get tokenizer
def get_tokenizer(root,sample_dataset=None,tokenizer_fname=None):
    if tokenizer_fname:
        with open(os.path.join(root,tokenizer_fname), 'wb') as f:
            tokenizer = pickle.load(f)
    else:
        # <pad> should always be first element in the list of special tokens
        special_tokens = ['<pad>','<start>','<end>','<unk>']
        tokenizer = Tokenizer(
                    sample_dataset.get_all_tokens(key='caption'),
                    special_tokens=special_tokens,
                )
    return tokenizer

# STEP 5: get encoder pretrained state dictionary
def extract_encoder_state_dict(root,chkpt_fname):
    checkpoint = torch.load(os.path.join(root,chkpt_fname) )
    state_dict = OrderedDict()
    for k,v in checkpoint['state_dict'].items():
        if 'classifier' in k: continue
        if k[:7] == 'module.' :
            name = k[7:]
        else:
            name = k

        name = name.replace('classifier.0.','classifier.')
        state_dict[name] = v
    return state_dict

# STEP 6: define model
def define_model(
            sample_dataset,
            tokenizer,
            encoder_weights,
            model_type='wordsat'):

    if model_type == 'wordsat':
        model=WordSAT(
              dataset = sample_dataset,
              n_input_images = 2,
              label_key = 'caption',
              tokenizer = tokenizer,
              encoder_pretrained_weights = encoder_weights,
              encoder_freeze_weights = True,
              save_generated_caption = True
             )
    else:
        model=SentSAT(
              dataset = sample_dataset,
              n_input_images = 2,
              label_key = 'caption',
              tokenizer = tokenizer,
              encoder_pretrained_weights = encoder_weights,
              encoder_freeze_weights = True,
              save_generated_caption = True
             )

    return model

# STEP 7: run trainer
def run_trainer(output_path, train_dataloader, val_dataloader, model,n_epochs):
    trainer = Trainer(
            model=model, 
            output_path=output_path
            )
    
    trainer.train(
        train_dataloader = train_dataloader,
        val_dataloader = val_dataloader,
        optimizer_params = {"lr": 1e-4},
        weight_decay = 1e-5,
        epochs = n_epochs,
        monitor = 'Bleu_1'
    )
    return trainer

# STEP 8: evaluate
def evaluate(trainer,test_dataloader):
    print(trainer.evaluate(test_dataloader))
    return None

if __name__ == '__main__':
    args = get_args()
    seed_everything(42)
    base_dataset = load_data(args.root)
    sample_dataset = set_task(base_dataset)
    train_dataloader,val_dataloader,test_dataloader = get_dataloaders(
                                                            sample_dataset)
    tokenizer = get_tokenizer(args.root,sample_dataset)
    encoder_weights = extract_encoder_state_dict(args.root,
                                                 args.encoder_chkpt_fname)

    model = define_model(
                sample_dataset,
                tokenizer,
                encoder_weights,
                args.model_type)

    trainer = run_trainer(
                    args.root, 
                    train_dataloader, 
                    val_dataloader, 
                    model,
                    args.num_epochs)

    print("\n===== Evaluating Test Data ======\n")
    evaluate(trainer,test_dataloader)




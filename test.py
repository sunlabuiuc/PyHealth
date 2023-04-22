import pickle
from torchvision import transforms
from pyhealth.datasets import BaseImageCaptionDataset
from pyhealth.tasks.xray_report_generation import biview_multisent_fn
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.tokenizer import Tokenizer
from pyhealth.models import WordSAT
from pyhealth.trainer import Trainer
from pyhealth.datasets.utils import list_nested_levels, flatten_list

import torch
from collections import OrderedDict
def extract_state_dict(chkpt_pth):
    checkpoint = torch.load(chkpt_pth + 'model_ones_3epoch_densenet.tar')
    new_state_dict = OrderedDict()
    for k,v in checkpoint['state_dict'].items():
        if 'classifier' in k: continue
        if k[:7] == 'module.' :
            name = k[7:]
        else:
            name = k

        name = name.replace('classifier.0.','classifier.')
        new_state_dict[name] = v
    return new_state_dict

chkpt_pth = '/home/keshari2/ChestXrayReporting/IU_XRay/src/models/pretrained/' 
state_dict = extract_state_dict(chkpt_pth)

#####
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
    
seed_everything(42)

root = '/home/keshari2/ChestXrayReporting/IU_XRay/src/data'
sample_dataset = BaseImageCaptionDataset(root=root,dataset_name='IU_XRay')
sample_dataset = sample_dataset.set_task(biview_multisent_fn)
transform = transforms.Compose([
                transforms.RandomAffine(degrees=30),
                transforms.Resize((512,512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],
                                std=[0.229,0.224,0.225]),
            ])
sample_dataset.set_transform(transform)

"""
special_tokens = ['<pad>','<start>','<end>','<unk>']
tokenizer = Tokenizer(
            sample_dataset.get_all_tokens(key='caption'),
            special_tokens=special_tokens,
        )

with open(root+'/pyhealth_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
"""
with open(root+'/pyhealth_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
#print(sample_dataset[0]['caption'])

train_dataset, val_dataset, test_dataset = split_by_patient(
    sample_dataset,[0.8,0.1,0.1]
)

train_dataloader = get_dataloader(train_dataset,batch_size=8,shuffle=True)
val_dataloader = get_dataloader(val_dataset,batch_size=1,shuffle=False)
test_dataloader = get_dataloader(test_dataset,batch_size=1,shuffle=False)

print(len(train_dataset),len(val_dataset),len(test_dataset))

model=WordSAT(dataset=sample_dataset,
              feature_keys=['image_1','image_2'],
              label_key='caption',
              tokenizer=tokenizer,
              mode='sequence',
              encoder_pretrained_weights=state_dict,
              save_generated_caption = True
             )
#model.eval()
#data = next(iter(val_dataloader))
#print(model(**data))
"""
output_path = '/home/keshari2/ChestXrayReporting/IU_XRay/src/output/pyhealth'
ckpt_path = '/home/keshari2/ChestXrayReporting/IU_XRay/src/output/pyhealth/20230422-005011/best.ckpt'

trainer = Trainer(
            model=model, 
            output_path=output_path,
            checkpoint_path = ckpt_path
            )
trainer.train(
    train_dataloader = train_dataloader,
    val_dataloader = val_dataloader,
    optimizer_params = {"lr": 1e-4},
    weight_decay = 1e-5,
    max_grad_norm = 1,
    epochs = 5,
    monitor = 'Bleu_1'
)
"""
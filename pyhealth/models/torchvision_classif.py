from typing import List, Dict

import torch
import torch.nn as nn
from torchvision import models

from pyhealth.datasets.sample_dataset_v2 import SampleDataset
from pyhealth.models import BaseModel


class TorchvisionClassification(BaseModel):
    """Torchvision model for image classification.
    -----------------------------------ResNet---------------------------------------------
    Paper: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition. CVPR 2016.
    -----------------------------------DenseNet-------------------------------------------
    Paper: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.
    Densely Connected Convolutional Networks. CVPR 2017.
    ----------------------------Vision Transformer (ViT)----------------------------------
    Paper: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, 
    Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, 
    Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.
    ----------------------------Swin Transformer (and V2)---------------------------------
    Paper: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, 
    Baining Guo.
    Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows. ICCV 2021.
    Paper: Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, 
    Yue Cao, Zheng Zhang, Li Dong, Furu Wei, Baining Guo.
    Swin Transformer V2: Scaling Up Capacity and Resolution. CVPR 2022.    
    --------------------------------------------------------------------------------------
    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        pretrained: whether to use pretrained weights. Default is False.
        model_parameters: dict, {"name" : str, 
                                 "num_layers": int,
                                 "model_size": str,
                                 "patch_size": int}
    
        Note that for different models, the items in model_parameters vary!
        model_parameters['name'] is one of "resnet", "densenet", "vit", "swin", "swin_v2"                             
        For ResNet:
            model_parameters = {"name": "resnet", 
                                "num_layers": int}
            "num_layers" is one of 18, 34, 50, 101, 152
        For DenseNet:
            model_parameters = {"name": "densenet", 
                                "num_layers": int}
            "num_layers" is one of 121, 161, 169, 201
        For Vision Transformer:
            model_parameters = {"name": "vit", 
                                "model_config": str}
            "model_config" is one of 'b_16', 'b_32', 'l_16', 'l_32', 'h_14'
        For Swin Transformer:
            model_parameters = {"name": "swin", 
                                "model_size": str}
            "model_config" is one of 't', 's', 'b' 
        For Swin Transformer V2:
            model_parameters = {"name": "swin_v2", 
                                "model_config": str}
            "model_config" is one of 't', 's', 'b'
    --------------------------------------------------------------------------------------
    Reference:
        Torchvision: https://github.com/mlverse/torchvision                   
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        model_parameters: dict,
        pretrained=False,
        **kwargs,
    ):
        super(TorchvisionClassification, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.model_name = model_parameters['name']
        
        if self.model_name == 'resnet':
            num_layers = model_parameters['num_layers']
            supported_num_layers = [18, 34, 50, 101, 152]
            try:
                supported_num_layers.index(num_layers)
                dnn_name = f"{self.model_name}{num_layers}"
                self.dnn = models.__dict__[dnn_name](pretrained=pretrained)
                hidden_dim = self.dnn.fc.in_features
                self.label_tokenizer = self.get_label_tokenizer()
                output_size = self.get_output_size(self.label_tokenizer)
                self.dnn.fc = nn.Linear(hidden_dim, output_size)  
            except:
                raise SystemExit('PyTorch does not provide this number of learnable layers for ResNet\
                                  \nThe candidate number is one of 18, 34, 50, 101, 152')          
        elif self.model_name == 'densenet':
            num_layers = model_parameters['num_layers']
            supported_num_layers = [121, 161, 169, 201]
            try: 
                supported_num_layers.index(num_layers)
                dnn_name = f"{self.model_name}{num_layers}"
                self.dnn = models.__dict__[dnn_name](pretrained=pretrained)
                num_ftrs = self.dnn.classifier.in_features 
                self.label_tokenizer = self.get_label_tokenizer()
                output_size = self.get_output_size(self.label_tokenizer)
                self.dnn.classifier = nn.Linear(num_ftrs, output_size)                
            except:
                raise SystemExit('PyTorch does not provide this number of learnable layers for DenseNet\
                                  \nThe candidate number is one of 121, 161, 169, 201')
        elif self.model_name == 'vit':
            model_config = model_parameters['model_config']
            supported_model_config = ['b_16', 'b_32', 'l_16', 'l_32', 'h_14']
            try:
                supported_model_config.index(model_config)
                dnn_name = f"{self.model_name}_{model_config}"
                self.dnn = models.__dict__[dnn_name](pretrained=pretrained)
                num_ftrs = self.dnn.heads.head.in_features
                self.label_tokenizer = self.get_label_tokenizer()
                output_size = self.get_output_size(self.label_tokenizer)
                self.dnn.heads.head = nn.Linear(num_ftrs, output_size)
            except:
                raise SystemExit('PyTorch does not provide this model configration for Vision Transformer\
                                  \nThe candidate is one of \'b_16\', \'b_32\', \'l_16\', \'l_32\', \'h_14\'')
        elif self.model_name == 'swin' or self.model_name == 'swin_v2':
            model_size = model_parameters['model_size']
            supported_model_size = ['t', 's', 'b']
            try:
                supported_model_size.index(model_size)
                dnn_name = f"{self.model_name}_{model_size}"
                self.dnn = models.__dict__[dnn_name](pretrained=pretrained)
                num_ftrs = self.dnn.head.in_features
                self.label_tokenizer = self.get_label_tokenizer()
                output_size = self.get_output_size(self.label_tokenizer)
                self.dnn.head = nn.Linear(num_ftrs, output_size)                                
            except:
                raise SystemExit('PyTorch does not provide this model size for Swin Transformer and Swin Transformer V2\
                                  \nThe candidate is one of \'t\', \'s\', \'b\'')                
        else:
            raise SystemExit(f'ERROR: PyHealth does not currently include {self.model_name} model!')        

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation."""
        # concat the info within one batch (batch, channel, length)
        x = kwargs[self.feature_keys[0]]
        x = torch.stack(x, dim=0).to(self.device)
        if x.shape[1] == 1:
            x = x.repeat((1, 3, 1, 1))
        logits = self.dnn(x)
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
        }


if __name__ == "__main__":
    from pyhealth.datasets.utils import get_dataloader
    from torchvision import transforms
    from pyhealth.datasets import COVID19CXRDataset

    base_dataset = COVID19CXRDataset(
        root="/home/wuzijian1231/Datasets/COVID-19_Radiography_Dataset",
    )

    sample_dataset = base_dataset.set_task()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5862785803043838], std=[0.27950088968644304])
    ])
    def encode(sample):
        sample["path"] = transform(sample["path"])
        return sample

    sample_dataset.set_transform(encode)

    train_loader = get_dataloader(sample_dataset, batch_size=16, shuffle=True)

    # model_parameters = {'name':'resnet', 'num_layers':18}
    # model_parameters = {'name':'densenet', 'num_layers':121}
    # model_parameters = {'name':'vit', 'model_config':'b_16'}
    # model_parameters = {'name':'swin', 'model_size':'t'}
    model_parameters = {'name':'swin_v2', 'model_size':'t'}
    
    model = TorchvisionClassification(
        dataset=sample_dataset,
        feature_keys=[
            "path",
        ],
        label_key="label",
        mode="multiclass",
        model_parameters=model_parameters,
    )

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()
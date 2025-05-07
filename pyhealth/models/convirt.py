import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from transformers import AutoModel

class BaselineModel(nn.Module):
    def __init__(self, init_type='random'):
        super().__init__()
        self.init_type = init_type
        if init_type == 'random':
            self.model = resnet50(pretrained=False)
        elif init_type == 'imagenet':
            self.model = resnet50(pretrained=True)
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")

        # Remove the classification head
        self.encoder = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        return self.encoder(x)

    def get_image_encoder(self):
        return self.encoder

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


class ConVIRTModel(nn.Module):
    def __init__(
            self,
            img_encoder_name="resnet50",
            txt_encoder_name="emilyalsentzer/Bio_ClinicalBERT",
            proj_dim=512,
            temperature=0.1,
            lambda_weight=0.75
    ):
        super().__init__()

        if img_encoder_name == "resnet50":
            self.img_encoder = resnet50(pretrained=True)
            self.img_encoder = nn.Sequential(*list(self.img_encoder.children())[:-1])
            self.img_feat_dim = 2048
        else:
            raise ValueError(f"Unknown encoder: {img_encoder_name}")

        self.txt_encoder = AutoModel.from_pretrained(txt_encoder_name)
        self.txt_feat_dim = self.txt_encoder.config.hidden_size

        ############################################################################################
        # "freezing the embeddings and the first 6 transformer layers of this BERT encoder"
        for param in self.txt_encoder.embeddings.parameters():
            param.requires_grad = False

        for i in range(6):
            for param in self.txt_encoder.encoder.layer[i].parameters():
                param.requires_grad = False
        ############################################################################################

        self.vis_projection = ProjectionHead(self.img_feat_dim, proj_dim)
        self.txt_projection = ProjectionHead(self.txt_feat_dim, proj_dim)

        self.temp = temperature
        self.lambda_w = lambda_weight

    def encode_image(self, images):
        features = self.img_encoder(images)

        if isinstance(features, tuple):
            features = features[0]

        if len(features.shape) == 4:
            features = features.squeeze(-1).squeeze(-1)
        return features

    def encode_text(self, input_ids, attention_mask):
        outputs = self.txt_encoder(input_ids=input_ids, attention_mask=attention_mask)

        masked = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
        masked[masked == 0] = -1e9
        return torch.max(masked, dim=1)[0]

    def info_nce_loss(self, query, keys):
        batch_size = query.shape[0]

        sim = torch.matmul(query, keys.transpose(0, 1)) / self.temp
        labels = torch.arange(batch_size, device=query.device)

        return F.cross_entropy(sim, labels)

    def forward(self, **kwargs):
        imgs = kwargs.get('images')
        ids = kwargs.get('input_ids')
        mask = kwargs.get('attention_mask')

        img_feat = self.encode_image(imgs)
        txt_feat = self.encode_text(ids, mask)

        img_proj = F.normalize(self.vis_projection(img_feat), p=2, dim=1)
        txt_proj = F.normalize(self.txt_projection(txt_feat), p=2, dim=1)

        img2txt = self.info_nce_loss(img_proj, txt_proj)
        txt2img = self.info_nce_loss(txt_proj, img_proj)

        loss = self.lambda_w * img2txt + (1 - self.lambda_w) * txt2img

        return {
            'vis_proj': img_proj,
            'txt_proj': txt_proj,
            'loss': loss
        }

    def predict(self, **kwargs):
        out = {}

        if 'images' in kwargs:
            img_feat = self.encode_image(kwargs['images'])
            img_proj = self.vis_projection(img_feat)
            out['vis_proj'] = F.normalize(img_proj, p=2, dim=1)

        if 'input_ids' in kwargs and 'attention_mask' in kwargs:
            txt_feat = self.encode_text(
                kwargs['input_ids'],
                kwargs['attention_mask']
            )
            txt_proj = self.txt_projection(txt_feat)
            out['txt_proj'] = F.normalize(txt_proj, p=2, dim=1)

        return out

    def get_image_encoder(self):
        return self.img_encoder

    def get_text_encoder(self):
        return self.txt_encoder


class ClassificationModel(nn.Module):
    def __init__(self, encoder, num_classes, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        try:
            with torch.no_grad():
                device = next(encoder.parameters()).device
                dummy = torch.randn(1, 3, 224, 224).to(device)
                out = encoder(dummy)
                input_dim = out.view(1, -1).shape[1]
        except:
            input_dim = 2048

        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, images):
        feat = self.encoder(images)

        if len(feat.shape) == 4:
            feat = feat.squeeze(-1).squeeze(-1)

        feat = self.dropout(feat)
        return self.classifier(feat)
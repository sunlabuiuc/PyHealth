import torch
import torch.nn as nn
import torch.nn.functional as F
from pyhealth.models import BaseModel

class MedFuse(BaseModel):
    def __init__(self, dataset, hidden_dim=128):
        super(MedFuse, self).__init__(dataset)
        
        # 1. Determine Input Size
        feature_key = self.feature_keys[0]
        processor = dataset.input_processors[feature_key]
        
        # Check if it's categorical (needs embedding) or already a vector
        if hasattr(processor, 'vocab_size'):
            self.input_size = processor.vocab_size() if callable(processor.vocab_size) else processor.vocab_size
            self.use_embedding = True
            self.embedding = nn.Embedding(self.input_size + 10, hidden_dim, padding_idx=0)
            lstm_input_dim = hidden_dim
        else:
            # If it's a vector (like the 27 we see in your error)
            self.input_size = 27 # Force match the 27 from your error log
            self.use_embedding = False
            lstm_input_dim = self.input_size
        
        # 2. EHR Encoder (LSTM)
        self.ehr_encoder = nn.LSTM(
            input_size=lstm_input_dim, 
            hidden_size=hidden_dim, 
            batch_first=True
        )
        
        # 3. Image Encoder (Linear)
        self.image_encoder = nn.Linear(512, hidden_dim)
        
        # 4. Fusion Layer
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def get_loss_function(self):
        return F.binary_cross_entropy_with_logits

    def forward(self, **kwargs):
        # 1. Extract EHR
        ehr_key = self.feature_keys[0]
        ehr_data = kwargs[ehr_key].float() 
        
        if self.use_embedding:
            # Process categorical codes
            ehr_feat = self.embedding(ehr_data.long())
            if ehr_feat.dim() == 4: # (batch, visit, code, dim)
                ehr_feat = torch.mean(ehr_feat, dim=2)
        else:
            # Data is already a vector (batch, visit, 27)
            ehr_feat = ehr_data
            
        # Ensure 3D for LSTM: (batch, seq, feature)
        if ehr_feat.dim() == 2:
            ehr_feat = ehr_feat.unsqueeze(1)

        # 2. Mock CXR
        if "cxr" in kwargs:
            cxr_data = kwargs["cxr"].float()
        else:
            cxr_data = torch.zeros(ehr_feat.shape[0], 512).to(self.device)

        # 3. Forward Pass
        _, (hn, _) = self.ehr_encoder(ehr_feat)
        ehr_out = hn[-1]
        cxr_out = self.image_encoder(cxr_data)
        
        fused = torch.cat([ehr_out, cxr_out], dim=-1)
        logits = self.fc(fused)
        
        y_true = kwargs[self.label_keys[0]].float().view(-1, 1)
        return {
            "loss": self.get_loss_function()(logits, y_true),
            "y_prob": torch.sigmoid(logits),
            "y_true": y_true
        }
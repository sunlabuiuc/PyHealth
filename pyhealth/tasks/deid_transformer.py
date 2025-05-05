
from transformers import AutoModelForTokenClassification
from pyhealth.tasks import TokenClassificationTask
class DeidTransformer(TokenClassificationTask):
    def __init__(self,model_name="bert-base-uncased",num_labels=2,**kw):
        super().__init__(**kw); self.model=AutoModelForTokenClassification.from_pretrained(model_name,num_labels=num_labels)
    def forward(self,input_ids,attention_mask,labels=None):
        out=self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels); return {"loss":out.loss,"logits":out.logits}

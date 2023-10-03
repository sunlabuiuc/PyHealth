"""
Transformer Feature Attribution Method for Self-Attention

Implementation based on https://github.com/hila-chefer/Transformer-Explainability

:param model: A trained base model.
:type model: PyHealth Transformer 
"""



import torch
import torch.nn.functional as F
from pyhealth.models import Transformer

"""Compute Chefer's Self Attention Rules for Relevance Maps """
def apply_self_attention_rules(R_ss, cam_ss):
    return torch.matmul(cam_ss, R_ss)

""" Average Attention Scores Weighed by their Gradients """
def avg_heads(cam, grad):
    # force shapes of cam and grad to be the same order

    if len(cam.size()) < 4 and len(grad.size()) < 4: # check if no averaging needed. i.e single head
        return (grad * cam).clamp(min=0)
    cam = grad * cam # elementwise mult
    cam = cam.clamp(min=0).mean(dim=1) # average across heads
    return cam.clone()

   
# STT transformer assumes no cls token so we average it at the end
class CheferRelevance():
    def __init__(self, model : Transformer):
        self.model = model
       
    """ Returns a list of F (# of different types of feature) tensors of batch size x number of tokens """
    def get_relevance_matrix(self, **data):
        input = data
        input['register_hook'] = True
        index = data.get('class_index')

        logits = self.model(**input)['logit']
        if index == None:
            index= torch.argmax(logits, dim=-1)

        # create one_hot matrix of n x c, one_hot vecs, for graph computation
        one_hot = F.one_hot(torch.tensor(index), logits.size()[1]).float()
        one_hot = one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot.to(logits.device) * logits)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        feature_keys = self.model.feature_keys 
        num_tokens = 0
        for key in feature_keys:
            feature_transformer = self.model.transformer[key].transformer
            for block in feature_transformer:
                num_tokens = block.attention.get_attn_map().shape[-1] # self.model.transformer[key].transformer.transformer_blocks[0].mhattn.get_attn_map().shape[-1]

        attn  = {}
        for key in feature_keys:
            R = torch.eye(num_tokens).unsqueeze(0).repeat(input.size()[0], 1, 1).to(logits.device) # initialize identity matrix, but batched
            for blk in self.model.transformer[key].transformer:
                grad = blk.attention.get_attn_grad()
                cam = blk.attention.get_attn_map()
                cam = avg_heads(cam, grad)
                R += apply_self_attention_rules(R, cam).detach()

            # R -= torch.eye(num_tokens).unsqueeze(0).expand(logits.size()[0], -1, -1).to(logits.device)
            attn[key] = R[0]
        # return 3 Rs for each feature_key
        return attn # Assume CLS token is first row of attention score matrix 









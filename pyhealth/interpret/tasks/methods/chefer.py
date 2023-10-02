"""
Transformer Interpretability

Implementation based on https://github.com/hila-chefer/Transformer-Explainability
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
       
    
    def get_relevance_matrix(self, input, index):

        output = self.model(input, register_hook=True)
        if index == None:
            index= torch.argmax(output, dim=-1)

        # create one_hot matrix of n x c, one_hot vecs, for graph computation
        one_hot = F.one_hot(index, output.size()[1]).float()
        one_hot = one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot.to(input.device) * output)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        num_tokens = self.model.transformer.transformer_blocks[0].mhattn.get_attn_map().shape[-1]
     
        R = torch.eye(num_tokens).unsqueeze(0).repeat(index.size()[0], 1, 1).to(input.device) # initialize identity matrix, but batched
        for blk in self.model.transformer.transformer_blocks:
            grad = blk.mhattn.get_attn_grad()
            cam = blk.mhattn.get_attn_map()
            cam = avg_heads(cam, grad)
            R = apply_self_attention_rules(R, cam).detach()

        R -= torch.eye(num_tokens).unsqueeze(0).expand(index.size()[0], -1, -1).to(input.device)
        return torch.mean(R, dim=1).detach() # Assume no CLS token, 









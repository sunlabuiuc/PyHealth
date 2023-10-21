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

   

class CheferRelevance():
    def __init__(self, model : Transformer):
        self.model = model
       
    """ Transformer Self Attention Token Relevance Computation
    Compute the relevance of each token in the input sequence for a given class index.
    The relevance is computed using the Chefer's Self Attention Rules.
    Paper:
        Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers
        Hila Chefer, Shir Gur, Lior Wolf
        https://arxiv.org/abs/2103.15679
        Implementation based on https://github.com/hila-chefer/Transformer-Explainability

    :param model: A trained transformer model.
    :type model: Transformer

    Examples:
        >>> from pyhealth.models import Transformer
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import length_of_stay_prediction_mimic3_fn
        >>> from pyhealth.interpret.methods.chefer import CheferRelevance
        >>> mimic3_ds = MIMIC3Dataset("/srv/scratch1/data/MIMIC3").set_task(length_of_stay_prediction_mimic3_fn)
        >>> model = Transformer(dataset=mimic3_ds,feature_keys=["conditions", "procedures", "drugs"], label_key="label",mode="multiclass",)
        >>> train_data, val_data, test_data = split_by_patient(sleep_ds, [0.6, 0.2, 0.2])
        >>> train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
        >>> val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
        >>> test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)
        >>> # ... Train the model here ...
        >>> # Get Relevance Scores for Tokens
        >>> relevance = CheferRelevance(model)
        >>> data_iterator = iter(test_loader)
        >>> data = next(data_iterator)
        >>> data['class_index'] = 0 # define class index
        >>> scores = relevance.get_relevance_matrix(**data)
        >>> print(scores)
        {'conditions': tensor([[1.2210]], device='cuda:0'), 'procedures': tensor([[1.0865]], device='cuda:0'), 'drugs': tensor([[1.]], device='cuda:0')}
    """
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

        # get how many tokens we see per modality
        num_tokens = {}
        for key in feature_keys:
            feature_transformer = self.model.transformer[key].transformer
            for block in feature_transformer:
                num_tokens[key] = block.attention.get_attn_map().shape[-1] 

        attn  = {}
        for key in feature_keys:
            R = torch.eye(num_tokens[key]).unsqueeze(0).repeat(len(input[key]), 1, 1).to(logits.device) # initialize identity matrix, but batched
            for blk in self.model.transformer[key].transformer:
                grad = blk.attention.get_attn_grad()
                cam = blk.attention.get_attn_map()
                cam = avg_heads(cam, grad)
                R += apply_self_attention_rules(R, cam).detach()
            
            attn[key] = R[:,0] # get CLS Token

        # return Rs for each feature_key
        return attn # Assume CLS token is first row of attention score matrix 









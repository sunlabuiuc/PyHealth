import torch
import torch.nn as nn
import torch.nn.functional as F
from pyhealth.interpret.tasks.methods.chefer import CheferRelevance

class VisionInterpreter():

    def __init__(self, model, method="chefer"):
        self.model = model
        if method == "chefer":
            self.relevance= CheferRelevance(model)   


    # if images, we know that input 
    def scale_rel_scores(self, relevance, out_size, vis = False):
        # interpolate such that we can get a bigger 1D vector of weights or scores for each chunk
        # since attn is a square matrix, our temporal grad attn is a vector
        # since we know that sequence is C x L, divide and ceiling it to recreate the "convolved zones" 
        attribution_scores = None
        input = input.unsqueeze(1) # so we can get B x 1 x T dimensions for interpolation of temporal attention

        attribution_scores = F.interpolate(relevance, size=out_size, mode="bilinear")    
      
        
        # attribution_scores = attribution_scores.squeeze().squeeze()# squeeze it back to a normal dim
        if vis:
            attribution_scores = attribution_scores.detach().cpu().numpy()

        return attribution_scores

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
    

    def visualize():
        vis = 0
        return vis


    

if __name__ == "__main__":
    print("HELLO")
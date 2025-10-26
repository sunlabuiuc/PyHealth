import torch
import numpy as np

class SaliencyMaps:
    """ Base class for saliency map methods """
    def __init__(self, model, dataloader, batches=1, image_key='image', label_key='disease'):
        self.model = model
        self.dataloader = dataloader
        self.batches = batches
        self.image_key = image_key
        self.label_key = label_key
        self.gradient_saliency_maps = None
        
    def get_gradient_saliency_maps(self):
        """Get the gradient saliency maps. Compute if not already computed."""
        self.init_gradient_saliency_maps()
        return self.gradient_saliency_maps
    
    def init_gradient_saliency_maps(self):
        if self.gradient_saliency_maps is None:
            self.GradientSaliencyMapping()

    def GradientSaliencyMapping(self):
        """ Create gradient saliency maps against batches from the inputted dataloader
        
        Args:
            self.model: standard pyhealth model
            self.dataloader: standard pyhealth dataloader
            self.batches: number of batches to process (default 1)
            self.image_key: adjust the image_key based on what the model expects (default 'image')
            self.label_key: adjust the image_key based on what the model expects (default 'disease')

        Updates:
            self.gradient_saliency_maps: list of saliency maps for each batch
        """
        self.model.eval()
        batch_count = 0
        self.gradient_saliency_maps = []
        for inputs in self.dataloader:
            if batch_count == self.batches:
                break
            imgs = inputs[self.image_key]
            batch_images = imgs.clone().detach().requires_grad_()
            batch_labels = inputs[self.label_key]
            output = self.model(image=batch_images, disease=batch_labels)
            y_prob = output['y_prob']
            target_class = y_prob.argmax(dim=1)
            scores = y_prob.gather(1, target_class.unsqueeze(1)).squeeze()

            self.model.zero_grad()
            scores.sum().backward()

            sal = batch_images.grad.abs()
            sal, _ = torch.max(sal, dim=1)

            self.gradient_saliency_maps.append({'saliency': sal, 'image': batch_images, 'label': batch_labels})
            batch_count += 1

    def imshowSaliencyCompFromDict(self, plt, batch_index, image_index, title, id2label, alpha=0.3):
        self.init_gradient_saliency_maps()
        img = self.gradient_saliency_maps[batch_index]['image'][image_index]
        saliency = self.gradient_saliency_maps[batch_index]['saliency'][image_index]
        label = self.gradient_saliency_maps[batch_index]['label'][image_index]
        new_title = str(title + " " + id2label[label.item()])
        self._imshowSaliencyComp(plt, img, saliency, new_title, alpha)

    def _imshowSaliencyComp(self, plt, img, saliency, title, alpha=0.3):
        npimg = img.detach().numpy()
        plt.figure(figsize=(15, 7))
        plt.axis('off')
        plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
        plt.imshow(saliency, cmap='hot', alpha=alpha)
        plt.title(title)
        plt.show()
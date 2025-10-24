import torch
import numpy as np


def GradientSaliencyMapping(model, dataloader, batches=1, image_key='image', label_key='disease'):
    """ Create gradient saliency maps against batches from the inputted dataloader

    Args:
        model: standard pyhealth model
        dataloader: standard pyhealth dataloader
        batches: number of batches to process (default 1)
        image_key: adjust the image_key based on what the model expects (default 'image')
        label_key: adjust the image_key based on what the model expects (default 'disease')

    Returns:
        saliency: [{'saliency': [batch_size], 'image': [batch_size], 'label': [batch_size]} batches]
    """
    model.eval()
    saliency = []
    batch_count = 0
    for inputs in dataloader:
        if batch_count == batches:
            break
        imgs = inputs[image_key]
        batch_images = imgs.clone().detach().requires_grad_()
        batch_labels = inputs[label_key]
        output = model(image=batch_images, disease=batch_labels)
        y_prob = output['y_prob']
        target_class = y_prob.argmax(dim=1)
        scores = y_prob.gather(1, target_class.unsqueeze(1)).squeeze()

        model.zero_grad()
        scores.sum().backward()

        sal = batch_images.grad.abs()
        sal, _ = torch.max(sal, dim=1)

        saliency.append({'saliency': sal, 'image': batch_images, 'label': batch_labels})
        batch_count += 1
    return saliency

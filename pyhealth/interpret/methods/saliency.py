import torch
import numpy as np

class SaliencyMaps:
    """Base class for computing and visualizing saliency maps.
    
    This class provides methods to generate aliency maps for image
    classification models, helping visualize which parts of an input image most
    influence the model's predictions.
    """
    def __init__(self, model, dataloader, batches=1, image_key='image', label_key='disease'):
        """Initialize the saliency map generator.
        
        Args:
            model: PyHealth model with forward method expecting image and disease kwargs
            dataloader: DataLoader providing batches of images and labels
            batches: Number of batches to process (default: 1)
            image_key: Key for accessing images in dataloader samples (default: 'image')
            label_key: Key for accessing labels in dataloader samples (default: 'disease')
        """
        self.model = model
        self.dataloader = dataloader
        self.batches = batches
        self.image_key = image_key
        self.label_key = label_key
        self.gradient_saliency_maps = None
        
    def init_gradient_saliency_maps(self):
        """Init gradient saliency maps, generating them if needed."""
        if self.gradient_saliency_maps is None:
            self._compute_saliency_maps()
            
    def get_gradient_saliency_maps(self):
        """Retrieve gradient saliency maps, generating them if needed."""
        self.init_gradient_saliency_maps()
        return self.gradient_saliency_maps

    
    def _compute_saliency_maps(self):
        """Compute gradient saliency maps for the given batches of images.
        
        For each image, computes the gradients of the predicted class score
        with respect to the input image pixels. The absolute values of these
        gradients indicate which pixels most strongly influence the model's
        prediction.
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

    def visualize_saliency_map(self, plt, batch_index, image_index, title, id2label, alpha=0.3):
        """Visualize a saliency map overlaid on its corresponding image.
        
        Args:
            plt: matplotlib.pyplot instance for visualization
            batch_index: Index of the batch containing the desired image
            image_index: Index of the image within the batch
            title: Base title for the plot
            id2label: Dictionary mapping class IDs to human-readable labels
            alpha: Transparency of the saliency overlay (default: 0.3)
        """
        # Ensure saliency maps are computed
        if self.gradient_saliency_maps is None:
            self._compute_saliency_maps()
            
        # Get the image and its saliency map
        img = self.gradient_saliency_maps[batch_index]['image'][image_index]
        saliency = self.gradient_saliency_maps[batch_index]['saliency'][image_index]
        label = self.gradient_saliency_maps[batch_index]['label'][image_index]
        
        # Create title with class label
        full_title = f"{title} - {id2label[label.item()]}"
        self._plot_saliency_overlay(plt, img, saliency, full_title, alpha)

    def _plot_saliency_overlay(self, plt, img, saliency, title, alpha=0.3):
        """Plot an image with its saliency map overlay.
        
        Args:
            plt: matplotlib.pyplot instance
            img: Input image tensor
            saliency: Saliency map tensor
            title: Plot title
            alpha: Transparency of saliency overlay
        """
        npimg = img.detach().numpy()
        plt.figure(figsize=(15, 7))
        plt.axis('off')
        plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
        plt.imshow(saliency, cmap='hot', alpha=alpha)
        plt.title(title)
        plt.show()
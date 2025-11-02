import torch
import numpy as np

class BasicGradientSaliencyMaps:
    """Base class for computing and visualizing saliency maps.
    
    This class provides methods to generate singular pytorch gradient-based saliency maps for image
    classification models, supporting both dataloader and direct batch inputs.
    Results from each source are stored separately and can be accessed independently.
    """
    def __init__(self, model, dataloader=None, input_batch=None, batches=1, 
                 image_key='image', label_key='disease'):
        """Initialize the saliency map generator.
        
        Args:
            model: PyHealth model with forward method expecting image and disease kwargs
            dataloader: Optional DataLoader providing batches of images and labels
            input_batch: Optional single batch of data formatted like dataloader output
            batches: Number of batches to process from dataloader (default: 1)
            image_key: Key for accessing images in samples (default: 'image')
            label_key: Key for accessing labels in samples (default: 'disease')
        """
        if dataloader is None and input_batch is None:
            raise ValueError("At least one of dataloader or input_batch must be provided")
            
        # Validate that input_batch is either a dictionary, list, or tensor if provided
        if input_batch is not None and not isinstance(input_batch, (dict, list, torch.Tensor)):
            raise ValueError("input_batch must be a dictionary, list, or tensor")
            
        self.Model = model
        self.Dataloader = dataloader
        self.Input_batch = input_batch
        self.Batches = batches
        self.Image_key = image_key
        self.Label_key = label_key
        # Store results separately for each source
        self.Batch_saliency_maps = None
        self.Dataloader_saliency_maps = None
        
    def init_gradient_saliency_maps(self):
        """Init gradient saliency maps, generating them if needed."""
        if self.Batch_saliency_maps is None and self.Dataloader_saliency_maps is None:
            self._compute_saliency_maps()
            
    def get_batch_saliency_maps(self):
        """Retrieve saliency maps from the input batch."""
        self.init_gradient_saliency_maps()
        return self.Batch_saliency_maps
    
    def get_dataloader_saliency_maps(self):
        """Retrieve saliency maps from the dataloader batches."""
        self.init_gradient_saliency_maps()
        return self.Dataloader_saliency_maps
    
    def get_gradient_saliency_maps(self):
        """Retrieve all gradient saliency maps, generating them if needed.
        
        Returns:
            tuple: (batch_maps, dataloader_maps) containing results from both sources
        """
        self.init_gradient_saliency_maps()
        return self.Batch_saliency_maps, self.Dataloader_saliency_maps
    
    def _compute_saliency_maps(self):
        """Compute gradient saliency maps for input data.
        
        Processes input batch and dataloader inputs separately if provided. 
        Stores results from each source in separate lists.
        """
        self.Model.eval()
        
        # Process input batch if provided
        if self.Input_batch is not None:
            self.Batch_saliency_maps = []
            if isinstance(self.Input_batch, (list, torch.Tensor)):
                # If input_batch is a list or tensor, wrap it in a dictionary
                batch_dict = {
                    self.Image_key: self.Input_batch[0] if isinstance(self.Input_batch, list) else self.Input_batch,
                    self.Label_key: self.Input_batch[1] if isinstance(self.Input_batch, list) else None
                }
                self._process_batch(batch_dict, is_dataloader=False)
            else:
                # Assume it's already a dictionary
                self._process_batch(self.Input_batch, is_dataloader=False)
        
        # Process dataloader batches if provided
        if self.Dataloader is not None:
            self.Dataloader_saliency_maps = []
            batch_count = 0
            for inputs in self.Dataloader:
                if batch_count == self.Batches:
                    break
                self._process_batch(inputs, is_dataloader=True)
                batch_count += 1
                
    def _process_batch(self, batch, is_dataloader=True):
        """Process a single batch of inputs to generate saliency maps.
        
        Args:
            batch: Dictionary containing image and label tensors with keys
                  matching self.Image_key and self.Label_key
            is_dataloader: Whether this batch is from the dataloader (True)
                         or direct input batch (False)
        """
        # Prepare input tensors
        imgs = batch[self.Image_key]
        batch_images = imgs.clone().detach().requires_grad_()
        batch_labels = batch[self.Label_key]
        
        # Get model predictions
        output = self.Model(image=batch_images, disease=batch_labels)
        y_prob = output['y_prob']
        target_class = y_prob.argmax(dim=1)
        scores = y_prob.gather(1, target_class.unsqueeze(1)).squeeze()

        # Compute gradients
        self.Model.zero_grad()
        scores.sum().backward()

        # Process gradients into saliency map
        sal = batch_images.grad.abs()
        sal, _ = torch.max(sal, dim=1)  # Max across channels

        # Store results in appropriate list
        result = {
            'saliency': sal,
            'image': batch_images,
            'label': batch_labels
        }
        
        if is_dataloader:
            self.Dataloader_saliency_maps.append(result)
        else:
            self.Batch_saliency_maps.append(result)

    def visualize_saliency_map(self, plt, *, image_index, title=None, id2label=None, alpha=0.3, batch_index=None):
        """Display an image with its saliency map overlay.
        
        This method supports visualizing saliency maps from either pre-computed results
        or from a new batch, determined by whether batch_index is provided.
        
        Args:
            plt: matplotlib.pyplot instance
            image_index: Index of image within batch
            title: Optional title for the plot
            id2label: Optional dictionary mapping class indices to labels
            alpha: Transparency of saliency overlay (default: 0.3)
            batch_index: Optional batch index for pre-computed maps. If None,
                        uses the most recent batch from the dataloader
        """
        if plt is None:
            import matplotlib.pyplot as plt

        # Determine if we're using pre-computed maps or the latest batch
        if batch_index is not None:
            # Use pre-computed results from dataloader
            if self.Dataloader_saliency_maps is None:
                raise ValueError("No pre-computed saliency maps available. Run init_gradient_saliency_maps() first.")
                
            if batch_index >= len(self.Dataloader_saliency_maps):
                raise ValueError(f"Batch index {batch_index} out of range")
                
            batch_result = self.Dataloader_saliency_maps[batch_index]
            if image_index >= len(batch_result['image']):
                raise ValueError(f"Image index {image_index} out of range for batch {batch_index}")
                
            img_tensor = batch_result['image'][image_index]
            saliency = batch_result['saliency'][image_index]
            label = batch_result['label'][image_index].item()
            
            # Compute prediction for this image to compare with true label
            img_for_pred = img_tensor.unsqueeze(0) if img_tensor.dim() == 3 else img_tensor
            with torch.no_grad():
                dummy_label = torch.zeros(1, dtype=torch.long, device=img_tensor.device)
                output = self.Model(image=img_for_pred, disease=dummy_label)
                pred_class = torch.argmax(output['y_prob']).item()
            
            # Add both true label and predicted class to title
            if id2label is not None:
                true_label_str = id2label[label]
                pred_label_str = id2label[pred_class]
                if title is None:
                    title = f"True: {true_label_str}, Predicted: {pred_label_str}"
                else:
                    title = f"{title} - True: {true_label_str}, Predicted: {pred_label_str}"
                
        else:
            # Use the latest batch from input_batch
            if self.Input_batch is None:
                raise ValueError("No input batch available. Initialize with input_batch or provide batch_index.")
                
            img_tensor = self.Input_batch[self.Image_key][image_index]
            true_label = self.Input_batch[self.Label_key][image_index].item()
            
            # Ensure input is a tensor with correct shape
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            if img_tensor.dim() != 4:
                raise ValueError(f"Expected 4D tensor (batch, channels, height, width), got shape {img_tensor.shape}")
                
            # Compute saliency
            img_tensor = img_tensor.clone().requires_grad_(True)
            
            # Create a dummy label tensor of zeros for the forward pass
            dummy_label = torch.zeros(img_tensor.size(0), dtype=torch.long, device=img_tensor.device)
            
            output = self.Model(image=img_tensor, disease=dummy_label)
            pred_class = torch.argmax(output['y_prob']).item()
            
            # Backward pass
            self.Model.zero_grad()
            output['y_prob'][:, pred_class].backward()
            
            # Get saliency map
            saliency = torch.max(img_tensor.grad.abs(), dim=1)[0]
            
            # Add both true label and predicted class to title
            if id2label is not None:
                true_label_str = id2label[true_label]
                pred_label_str = id2label[pred_class]
                if title is None:
                    title = f"True: {true_label_str}, Predicted: {pred_label_str}"
                else:
                    title = f"{title} - True: {true_label_str}, Predicted: {pred_label_str}"

        # Convert image to numpy for display
        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0]
        img_np = img_tensor.detach().cpu().numpy()
        if img_np.shape[0] in [1, 3]:  # CHW to HWC
            img_np = np.transpose(img_np, (1, 2, 0))
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)
            
        # Convert saliency to numpy
        if saliency.dim() > 2:
            saliency = saliency[0]
        saliency_np = saliency.detach().cpu().numpy()
        
        # Create visualization
        plt.figure(figsize=(15, 7))
        plt.axis('off')
        plt.imshow(img_np, cmap='gray')
        plt.imshow(saliency_np, cmap='hot', alpha=alpha)
        if title:
            plt.title(title)
        plt.show()

    def _compute_gradients(self, img_tensor, target_label=None):
        """Compute gradients of model output with respect to input image.
        
        Args:
            img_tensor: Input image tensor
            target_label: Optional target class. If None, uses predicted class.
            
        Returns:
            torch.Tensor: Gradient of target class output with respect to input
        """
        # Forward pass
        self.Model.zero_grad()
        output = self.Model(image=img_tensor, disease=None)['y_prob']
        
        if target_label is None:
            target_label = output.argmax(dim=1)
            
        # One-hot encoding of target class
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, target_label.unsqueeze(1), 1.0)
        
        # Backward pass
        output.backward(gradient=one_hot)
        return img_tensor.grad
        
    def _compute_saliency(self, gradients):
        """Convert raw gradients to saliency map.
        
        Args:
            gradients: Gradient tensor from _compute_gradients
            
        Returns:
            torch.Tensor: Saliency map tensor
        """
        # Take maximum magnitude across channels
        saliency, _ = torch.max(gradients.abs(), dim=1)
        return saliency

    def _plot_saliency_overlay(self, plt, img, saliency, title, alpha=0.3):
        """Plot an image with its saliency map overlay.
        
        Args:
            plt: matplotlib.pyplot instance
            img: Input image tensor
            saliency: Saliency map tensor
            title: Plot title
            alpha: Transparency of saliency overlay
        """
        # Get the first image if this is a batch
        if img.dim() == 4:
            img = img[0]
            
        # Convert to numpy, handling both CHW and HWC formats
        npimg = img.detach().cpu().numpy()
        if npimg.shape[0] in [1, 3]:  # If channels first (CHW)
            npimg = np.transpose(npimg, (1, 2, 0))
            
        # Handle single channel images
        if npimg.shape[-1] == 1:
            npimg = npimg.squeeze(-1)
            
        # Convert saliency to numpy, taking first item if it's a batch
        if saliency.dim() > 2:
            saliency = saliency[0]
        npsaliency = saliency.detach().cpu().numpy()
        
        plt.figure(figsize=(15, 7))
        plt.axis('off')
        plt.imshow(npimg, cmap='gray')
        plt.imshow(npsaliency, cmap='hot', alpha=alpha)
        plt.title(title)
        plt.show()
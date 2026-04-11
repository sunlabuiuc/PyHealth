import torch
import numpy as np
from typing import Dict
from pyhealth.interpret.methods.base_interpreter import BaseInterpreter

class BasicGradientSaliencyMaps(BaseInterpreter):
    """Compute gradient-based saliency maps for image classification models.
    
    This class generates saliency maps by computing gradients of model predictions with respect
    to input pixels, highlighting regions that most influenced the prediction. The saliency is
    computed by taking the maximum absolute gradient across color channels.
    
    The method is particularly useful for:
    
    - **Clinical interpretability**: Understanding which image regions drove a diagnosis
    - **Model debugging**: Verifying the model focuses on clinically relevant features
    - **Trust and transparency**: Providing visual explanations for predictions
    - **Error analysis**: Comparing saliency maps for correct vs. incorrect predictions
    
    Algorithm:
        1. Forward pass: Compute model predictions for input batch
        2. Target selection: Use predicted class (argmax of probabilities)
        3. Backward pass: Compute gradients with respect to input pixels
        4. Saliency map: Take absolute value and max across color channels
    
    Mathematical formula:
        saliency(x, y) = max_c |∂score_predicted / ∂pixel_{x,y,c}|
        where c iterates over color channels (RGB or grayscale)
    
    Examples:
        Basic usage with a batch::
        
            from pyhealth.interpret.methods.basic_gradient import BasicGradientSaliencyMaps
            import matplotlib.pyplot as plt
            
            # Create batch
            batch = {
                'image': torch.randn(2, 3, 224, 224),
                'disease': torch.tensor([0, 1])
            }
            
            # Compute saliency maps
            saliency = BasicGradientSaliencyMaps(model, input_batch=batch)
            
            # Visualize
            saliency.visualize_saliency_map(
                plt, 
                image_index=0,
                title="Saliency Map",
                id2label={0: "Normal", 1: "COVID"}
            )
        
        Using the attribute() interface::
        
            # Initialize without batch
            saliency = BasicGradientSaliencyMaps(model)
            
            # Compute attributions for new data
            attributions = saliency.attribute(**batch)
            # Returns: {'image': tensor with saliency maps}
            
            # Save to batch history
            attributions = saliency.attribute(save_to_batch=True, **batch)
    
    Note:
        - Do not use within ``torch.no_grad()`` context as gradients are required
        - Works with any PyHealth image classification model
        - For best results, normalize input images consistently with training
    
    See Also:
        - ``examples/ChestXrayClassificationWithSaliency.ipynb``: Complete tutorial
        - :class:`~pyhealth.interpret.methods.IntegratedGradients`: Alternative attribution method
    """
    def __init__(self, model, input_batch=None, image_key='image', label_key='disease'):
        """Initialize the saliency map generator.
        
        Args:
            model: PyHealth model with forward method expecting image and disease kwargs
            input_batch: Optional batch of data as dictionary, list, or tensor. 
                        If None, use attribute() method to compute saliency maps.
            image_key: Key for accessing images in samples (default: 'image')
            label_key: Key for accessing labels in samples (default: 'disease')
        """
        # Validate that input_batch is either a dictionary, list, tensor, or None
        if input_batch is not None and not isinstance(input_batch, (dict, list, torch.Tensor)):
            raise ValueError("input_batch must be a dictionary, list, tensor, or None")
        
        # Call parent constructor
        super().__init__(model)
        
        # Store additional attributes specific to this class
        self.Model = model  # Keep for backward compatibility
        self.Input_batch = input_batch
        self.Image_key = image_key
        self.Label_key = label_key
        self.Batch_saliency_maps = []
        
        # Compute saliency maps if input_batch was provided
        if input_batch is not None:
            self._compute_saliency_maps()
    
    def attribute(self, save_to_batch=False, **data) -> Dict[str, torch.Tensor]:
        """Compute attribution scores for input features.
        
        This method implements the BaseInterpreter interface by computing
        gradient-based saliency maps for the input images.
        
        Args:
            save_to_batch: If True, save results to Batch_saliency_maps (default: False)
            **data: Input data dictionary containing 'image' and optionally 'disease' keys
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'saliency' key mapping to saliency map tensor
        """
        # Process the batch
        if isinstance(data, (list, torch.Tensor)):
            batch_dict = {
                self.Image_key: data[0] if isinstance(data, list) else data,
                self.Label_key: data[1] if isinstance(data, list) else None
            }
        else:
            batch_dict = data
        
        # Prepare input tensors
        imgs = batch_dict[self.Image_key]
        batch_images = imgs.clone().detach().requires_grad_()
        batch_labels = batch_dict.get(self.Label_key, None)
        
        # Get model predictions
        output = self.model(image=batch_images, disease=batch_labels)
        y_prob = output['y_prob']
        target_class = y_prob.argmax(dim=1)
        scores = y_prob.gather(1, target_class.unsqueeze(1)).squeeze()

        # Compute gradients
        self.model.zero_grad()
        scores.sum().backward()

        # Process gradients into saliency map
        sal = batch_images.grad.abs()
        sal, _ = torch.max(sal, dim=1)  # Max across channels
        
        # Save to Batch_saliency_maps if requested
        if save_to_batch:
            result = {
                'saliency': sal,
                'image': batch_images,
                'label': batch_labels
            }
            self.Batch_saliency_maps.append(result)
        
        return {self.Image_key: sal}
            
    def get_gradient_saliency_maps(self):
        """Retrieve gradient saliency maps.
        
        Returns:
            list: Batch saliency map results
        """
        return self.Batch_saliency_maps
    
    def _compute_saliency_maps(self):
        """Compute gradient saliency maps for input batch."""
        if self.Input_batch is None:
            return  # Nothing to compute
            
        self.Model.eval()
        
        if isinstance(self.Input_batch, (list, torch.Tensor)):
            # If input_batch is a list or tensor, wrap it in a dictionary
            batch_dict = {
                self.Image_key: self.Input_batch[0] if isinstance(self.Input_batch, list) else self.Input_batch,
                self.Label_key: self.Input_batch[1] if isinstance(self.Input_batch, list) else None
            }
            self._process_batch(batch_dict)
        else:
            # Assume it's already a dictionary
            self._process_batch(self.Input_batch)
                
    def _process_batch(self, batch):
        """Process a batch of inputs to generate saliency maps.
        
        This method wraps the attribute() method to maintain backward compatibility
        with the original batch processing API.
        
        Args:
            batch: Dictionary containing image and label tensors with keys
                  matching self.Image_key and self.Label_key
        """
        # Use attribute method to compute saliency
        attributions = self.attribute(**batch)
        
        # Extract the saliency map from attributions
        sal = attributions[self.Image_key]
        
        # Prepare input tensors for storing complete results
        imgs = batch[self.Image_key]
        batch_images = imgs.clone().detach().requires_grad_()
        batch_labels = batch[self.Label_key]
        
        # Store results in the original format for backward compatibility
        result = {
            'saliency': sal,
            'image': batch_images,
            'label': batch_labels
        }
        
        self.Batch_saliency_maps.append(result)

    def visualize_saliency_map(self, plt, *, image_index, title=None, id2label=None, alpha=0.3):
        """Display an image with its saliency map overlay.
        
        Args:
            plt: matplotlib.pyplot instance
            image_index: Index of image within batch
            title: Optional title for the plot
            id2label: Optional dictionary mapping class indices to labels
            alpha: Transparency of saliency overlay (default: 0.3)
        """
        if plt is None:
            import matplotlib.pyplot as plt

        # Check if input_batch is available
        if self.Input_batch is None:
            raise ValueError("Cannot visualize: no input_batch was provided during initialization")

        # Get image from input batch
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
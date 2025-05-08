"""
Accuracy Extrapolation Module

This module provides functionality to predict model performance when trained on
larger datasets based on small pilot data, implementing the APEx-GP approach with
improvements including Matern kernels and Beta priors.

Based on the paper "A Probabilistic Method to Predict Classifier Accuracy on
Larger Datasets given Small Pilot Data" with additional improvements.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional
from ..utils import tensor_to_numpy

try:
    import gpytorch
except ImportError:
    raise ImportError(
        "You need to install the gpytorch package to use the extrapolation module. "
        "Install it with: pip install gpytorch"
    )

# Set default tensor type for consistency
torch.set_default_dtype(torch.float32)

class GPKernelBase(gpytorch.models.ExactGP):
    """Base class for Gaussian Process models with different kernels."""
    
    def __init__(
        self, 
        train_x: torch.Tensor, 
        train_y: torch.Tensor, 
        likelihood: gpytorch.likelihoods.Likelihood,
        mean_type: str = "constant",
        kernel_type: str = "rbf",
        epsilon_min: float = 0.0, 
        with_priors: bool = True,
    ):
        """
        Initialize a GP model with specified mean and kernel functions.
        
        Args:
            train_x: Training inputs
            train_y: Training targets
            likelihood: GP likelihood function
            mean_type: Type of mean function ("constant", "powerlaw", "arctan")
            kernel_type: Type of kernel function ("rbf", "matern12", "matern32", "matern52")
            epsilon_min: Minimum value for epsilon in powerlaw/arctan models
            with_priors: Whether to use priors on model parameters
        """
        # Ensure consistent tensor types
        train_x = train_x.to(torch.float32)
        train_y = train_y.to(torch.float32)
        
        super(GPKernelBase, self).__init__(train_x, train_y, likelihood)
        self.max_y = torch.max(train_y).item()
        
        # Set up mean module
        if mean_type == "constant":
            self.mean_module = gpytorch.means.ConstantMean()
        elif mean_type == "powerlaw":
            self.mean_module = PowerLawMean(self.max_y, epsilon_min)
        elif mean_type == "arctan":
            self.mean_module = ArctanMean(self.max_y, epsilon_min)
        else:
            raise ValueError(f"Unknown mean type: {mean_type}")
        
        # Set up kernel module
        if kernel_type == "rbf":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel_type == "matern12":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        elif kernel_type == "matern32":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        elif kernel_type == "matern52":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        # Set priors if requested
        if with_priors:
            # Set reasonable priors for the parameters
            if mean_type == "constant":
                self.mean_module.constant.constraint = gpytorch.constraints.Interval(0.5, 1.0)
            
            self.covar_module.base_kernel.lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
            self.covar_module.outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15)
            likelihood.noise_constraint = gpytorch.constraints.GreaterThan(1e-5)
            
            if hasattr(self.mean_module, "epsilon"):
                self.register_prior(
                    'epsilon_prior',
                    gpytorch.priors.UniformPrior(epsilon_min, (1.0 - self.max_y)),
                    lambda module: module.mean_module.epsilon_min + 
                                   (1.0 - module.mean_module.max_y - module.mean_module.epsilon_min) * 
                                   torch.sigmoid(module.mean_module.epsilon)
                )
    
    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Forward pass for the GP model"""
        # Ensure input is float32
        x = x.to(torch.float32)
        
        mean_x = self.mean_module(x)
        # Use logarithmic transformation for input space
        covar_x = self.covar_module(torch.log10(x + 1e-8))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class PowerLawMean(gpytorch.means.Mean):
    """Power law mean function for Gaussian Process"""
    
    def __init__(self, max_y: float, epsilon_min: float = 0.0):
        """
        Initialize power law mean function
        
        Args:
            max_y: Maximum observed value in training data
            epsilon_min: Minimum value for epsilon parameter
        """
        super(PowerLawMean, self).__init__()
        self.max_y = max_y
        self.epsilon_min = epsilon_min
        self.epsilon = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.theta1 = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.theta2 = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.softplus = torch.nn.Softplus()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for power law mean function"""
        # Ensure input is float32
        x = x.to(torch.float32)
        
        epsilon = self.epsilon_min + (1.0 - self.max_y - self.epsilon_min) * torch.sigmoid(self.epsilon)
        theta1 = self.softplus(self.theta1)
        theta2 = -torch.sigmoid(self.theta2)
        return (1.0 - epsilon) - (theta1 * torch.pow(x.ravel(), theta2))


class ArctanMean(gpytorch.means.Mean):
    """Arctan mean function for Gaussian Process"""
    
    def __init__(self, max_y: float, epsilon_min: float = 0.0):
        """
        Initialize arctan mean function
        
        Args:
            max_y: Maximum observed value in training data
            epsilon_min: Minimum value for epsilon parameter
        """
        super(ArctanMean, self).__init__()
        self.max_y = max_y
        self.epsilon_min = epsilon_min
        self.epsilon = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.theta1 = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.theta2 = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.softplus = torch.nn.Softplus()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for arctan mean function"""
        # Ensure input is float32
        x = x.to(torch.float32)
        
        epsilon = self.epsilon_min + (1.0 - self.max_y - self.epsilon_min) * torch.sigmoid(self.epsilon)
        theta1 = self.softplus(self.theta1)
        theta2 = self.softplus(self.theta2)
        return 2/np.pi * torch.atan(theta1 * np.pi/2 * x.ravel() + theta2) - epsilon


class GPMatern(GPKernelBase):
    """Gaussian Process with Matern kernel implementation"""
    
    def __init__(
        self, 
        train_x: torch.Tensor, 
        train_y: torch.Tensor, 
        likelihood: gpytorch.likelihoods.Likelihood,
        nu: float = 2.5,
        mean_type: str = "powerlaw",
        epsilon_min: float = 0.0, 
        with_priors: bool = True
    ):
        """
        Initialize GP with Matern kernel
        
        Args:
            train_x: Training inputs
            train_y: Training targets
            likelihood: GP likelihood function
            nu: Smoothness parameter (0.5, 1.5, or 2.5)
            mean_type: Type of mean function
            epsilon_min: Minimum value for epsilon
            with_priors: Whether to use priors
        """
        assert nu in [0.5, 1.5, 2.5], "nu must be one of 0.5, 1.5, or 2.5"
        kernel_mapping = {0.5: "matern12", 1.5: "matern32", 2.5: "matern52"}
        super(GPMatern, self).__init__(
            train_x, train_y, likelihood, 
            mean_type=mean_type, 
            kernel_type=kernel_mapping[nu],
            epsilon_min=epsilon_min,
            with_priors=with_priors
        )


class BetaPriorGP(GPKernelBase):
    """GP model with Beta prior for bounded outputs"""
    
    def __init__(
        self, 
        train_x: torch.Tensor, 
        train_y: torch.Tensor, 
        likelihood: gpytorch.likelihoods.BetaLikelihood = None,
        mean_type: str = "powerlaw",
        kernel_type: str = "rbf",
        epsilon_min: float = 0.0,
        with_priors: bool = True
    ):
        """
        Initialize GP with Beta prior
        
        Args:
            train_x: Training inputs
            train_y: Training targets
            likelihood: Beta likelihood function
            mean_type: Type of mean function
            kernel_type: Type of kernel function
            epsilon_min: Minimum value for epsilon
            with_priors: Whether to use priors
        """
        # Ensure targets are in (0,1) range
        train_y = train_y.to(torch.float32)
        assert torch.all(train_y > 0) and torch.all(train_y < 1), "Beta prior requires targets in range (0,1)"
        
        # Create custom Beta likelihood if not provided
        if likelihood is None:
            likelihood = gpytorch.likelihoods.BetaLikelihood()
            
        super(BetaPriorGP, self).__init__(
            train_x, train_y, likelihood,
            mean_type=mean_type,
            kernel_type=kernel_type,
            epsilon_min=epsilon_min,
            with_priors=with_priors
        )


class AccuracyExtrapolation:
    """Accuracy extrapolation for predicting model performance with more data"""
    
    def __init__(
        self, 
        model_type: str = "matern", 
        nu: float = 2.5,
        mean_type: str = "powerlaw",
        use_beta_prior: bool = False,
        epsilon_min: float = 0.0,
        with_priors: bool = True
    ):
        """
        Initialize accuracy extrapolation
        
        Args:
            model_type: Type of model ("rbf", "matern")
            nu: Smoothness parameter for Matern kernels
            mean_type: Type of mean function
            use_beta_prior: Whether to use Beta prior
            epsilon_min: Minimum value for epsilon
            with_priors: Whether to use priors
        """
        self.model_type = model_type
        self.nu = nu
        self.mean_type = mean_type
        self.use_beta_prior = use_beta_prior
        self.epsilon_min = epsilon_min
        self.with_priors = with_priors
        self.model = None
        self.likelihood = None
        
    def fit(
        self, 
        train_sizes: Union[List[int], np.ndarray, torch.Tensor],
        accuracies: Union[List[float], np.ndarray, torch.Tensor],
        max_iter: int = 1000,
        lr: float = 0.01,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Fit model to training data
        
        Args:
            train_sizes: List of training set sizes
            accuracies: List of corresponding accuracies
            max_iter: Maximum iterations for optimization
            lr: Learning rate
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training losses
        """
        # Convert inputs to tensors if they aren't already
        if not isinstance(train_sizes, torch.Tensor):
            train_sizes = torch.tensor(train_sizes, dtype=torch.float32)
        if not isinstance(accuracies, torch.Tensor):
            accuracies = torch.tensor(accuracies, dtype=torch.float32)
        
        # Choose appropriate likelihood based on prior
        if self.use_beta_prior:
            # Ensure accuracies are in (0,1) range for Beta prior
            accuracies = torch.clamp(accuracies, 1e-6, 1-1e-6)
            self.likelihood = gpytorch.likelihoods.BetaLikelihood()
        else:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        # Create model based on type
        if self.model_type == "matern":
            self.model = GPMatern(
                train_sizes, accuracies, self.likelihood,
                nu=self.nu, mean_type=self.mean_type,
                epsilon_min=self.epsilon_min, with_priors=self.with_priors
            )
        elif self.model_type == "rbf":
            self.model = GPKernelBase(
                train_sizes, accuracies, self.likelihood,
                mean_type=self.mean_type, kernel_type="rbf",
                epsilon_min=self.epsilon_min, with_priors=self.with_priors
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        # Train the model
        losses = self._train_model(max_iter, lr, verbose)
        
        return {"losses": losses}
        
    def _train_model(self, max_iter: int, lr: float, verbose: bool) -> List[float]:
        """Train the GP model"""
        self.likelihood.train()
        self.model.train()
        losses = []
        
        # Use GPU if available
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.likelihood = self.likelihood.to(device)
        
        # Setup optimization
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # Move data to device
        train_x = self.model.train_inputs[0].to(device)
        train_y = self.model.train_targets.to(device)
        
        # Training loop
        for i in range(max_iter):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if verbose and (i+1) % 100 == 0:
                print(f"Iter {i+1}/{max_iter} - Loss: {loss.item():.6f}")
        
        # Move back to CPU for inference
        self.model = self.model.to('cpu')
        self.likelihood = self.likelihood.to('cpu')
        self.model.eval()
        self.likelihood.eval()
        
        return losses
    
    def predict(
        self, 
        test_sizes: Union[List[int], np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict accuracies for given test sizes
        
        Args:
            test_sizes: List of test sizes to predict for
            
        Returns:
            Tuple of (mean predictions, lower bounds, upper bounds)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
            
        # Convert to tensor if not already
        if not isinstance(test_sizes, torch.Tensor):
            test_sizes = torch.tensor(test_sizes, dtype=torch.float32)
        else:
            test_sizes = test_sizes.to(torch.float32)
        
        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(test_sizes))
            
            # Get mean and confidence intervals
            if hasattr(predictions, 'mean'):
                mean = predictions.mean
                lower, upper = predictions.confidence_region()
            else:
                # For Beta likelihood which returns a Beta distribution
                mean = predictions.mean
                # Get approximate confidence intervals
                variance = predictions.variance
                lower = torch.clamp(mean - 2 * torch.sqrt(variance), 0.0, 1.0)
                upper = torch.clamp(mean + 2 * torch.sqrt(variance), 0.0, 1.0)
                
        # Convert to numpy arrays
        return tensor_to_numpy(mean), tensor_to_numpy(lower), tensor_to_numpy(upper)
        
    def plot(
        self, 
        train_sizes: Union[List[int], np.ndarray, torch.Tensor],
        accuracies: Union[List[float], np.ndarray, torch.Tensor],
        extrapolate_to: int = None,
        num_points: int = 100,
        log_scale: bool = True,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot training data and extrapolation predictions
        
        Args:
            train_sizes: Training set sizes
            accuracies: Corresponding accuracies
            extrapolate_to: Maximum size to extrapolate to
            num_points: Number of points to use for prediction curve
            log_scale: Whether to use log scale for x-axis
            save_path: Path to save the figure
            show: Whether to show the figure
            
        Returns:
            Matplotlib figure
        """
        if not isinstance(train_sizes, np.ndarray):
            train_sizes = np.array(train_sizes)
        if not isinstance(accuracies, np.ndarray):
            accuracies = np.array(accuracies)
            
        # Set extrapolation limit if not provided
        if extrapolate_to is None:
            extrapolate_to = int(train_sizes.max() * 10)
            
        # Generate prediction points
        if log_scale:
            pred_sizes = np.logspace(
                np.log10(train_sizes.min()), 
                np.log10(extrapolate_to),
                num_points
            )
        else:
            pred_sizes = np.linspace(
                train_sizes.min(),
                extrapolate_to,
                num_points
            )
            
        # Make predictions
        mean, lower, upper = self.predict(pred_sizes)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(train_sizes, accuracies, c='black', s=50, label='Training data')
        plt.plot(pred_sizes, mean, 'b-', label='Prediction')
        plt.fill_between(pred_sizes, lower, upper, alpha=0.3, color='b', label='95% Confidence')
        
        if log_scale:
            plt.xscale('log')
            
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Extrapolation (Model: {self.model_type}, Mean: {self.mean_type})')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        return plt.gcf()


def extrapolate_accuracy(
    train_sizes: Union[List[int], np.ndarray, torch.Tensor],
    accuracies: Union[List[float], np.ndarray, torch.Tensor],
    target_sizes: Union[List[int], np.ndarray, torch.Tensor] = None,
    model_type: str = "matern",
    nu: float = 2.5,
    mean_type: str = "powerlaw",
    use_beta_prior: bool = False,
    return_std: bool = False,
    max_iter: int = 1000,
    plot: bool = False,
    plot_path: Optional[str] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience function to extrapolate model accuracy to larger dataset sizes
    
    Args:
        train_sizes: List of training set sizes used
        accuracies: List of corresponding accuracies
        target_sizes: Target sizes to extrapolate to
        model_type: Type of model ("rbf", "matern")
        nu: Smoothness parameter for Matern kernels
        mean_type: Type of mean function
        use_beta_prior: Whether to use Beta prior
        return_std: Whether to return standard deviations
        max_iter: Maximum training iterations
        plot: Whether to generate a plot
        plot_path: Path to save the plot
        
    Returns:
        Predicted accuracies or tuple of (predicted accuracies, standard deviations)
    """
    # Create and fit model
    extrapolator = AccuracyExtrapolation(
        model_type=model_type,
        nu=nu,
        mean_type=mean_type,
        use_beta_prior=use_beta_prior
    )
    
    extrapolator.fit(train_sizes, accuracies, max_iter=max_iter, verbose=False)
    
    # Set target sizes if not provided
    if target_sizes is None:
        max_size = np.max(train_sizes) if isinstance(train_sizes, np.ndarray) else max(train_sizes)
        target_sizes = [int(max_size * 2), int(max_size * 5), int(max_size * 10)]
    
    # Get predictions
    mean, lower, upper = extrapolator.predict(target_sizes)
    std = (upper - lower) / 4  # Approximate standard deviation from 95% CI
    
    # Generate plot if requested
    if plot:
        extrapolator.plot(
            train_sizes, accuracies,
            extrapolate_to=int(np.max(target_sizes) * 1.2),
            save_path=plot_path,
            show=(plot_path is None)
        )
    
    if return_std:
        return mean, std
    else:
        return mean 
"""
Example script for using the PyHealth Accuracy Extrapolation Module

This example demonstrates how to use the accuracy extrapolation functionality
to predict model performance on larger datasets based on small pilot data.

The example:
1. Creates a synthetic dataset with learning curve data
2. Trains different GP models to extrapolate performance
3. Plots and compares predictions from RBF and Matern kernels
4. Shows how to use Beta priors for bounded accuracy metrics
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pyhealth.metrics.extrapolation import AccuracyExtrapolation, extrapolate_accuracy

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def generate_learning_curve_data(min_size=100, max_size=1000, num_points=10, noise=0.02):
    """Generate synthetic learning curve data following a power law"""
    sizes = np.logspace(np.log10(min_size), np.log10(max_size), num_points).astype(int)
    
    # Power law function: y = 1 - α * n^β where n is the training set size
    alpha = 0.6
    beta = -0.3
    
    # Calculate true accuracies
    true_accuracies = 1 - alpha * np.power(sizes, beta)
    
    # Add noise to simulate real data
    accuracies = true_accuracies + np.random.normal(0, noise, size=num_points)
    
    # Ensure values are in valid range
    accuracies = np.clip(accuracies, 0.01, 0.99)
    
    return sizes, accuracies, true_accuracies

def run_simple_example():
    """Example using the convenience function for quick extrapolation"""
    print("Running simple example with convenience function...")
    
    # Generate data
    sizes, accuracies, true_values = generate_learning_curve_data(min_size=100, max_size=1000, num_points=8)
    
    # Define target sizes for extrapolation
    target_sizes = [2000, 5000, 10000, 20000]
    
    print(f"Training sizes: {sizes}")
    print(f"Observed accuracies: {accuracies.round(3)}")
    
    # Use convenience function with Matern kernel
    predictions_matern = extrapolate_accuracy(
        train_sizes=sizes,
        accuracies=accuracies,
        target_sizes=target_sizes,
        model_type="matern",
        nu=2.5,
        mean_type="powerlaw",
        use_beta_prior=False,
        plot=True
    )
    
    # Calculate true values for comparison
    alpha = 0.6
    beta = -0.3
    true_target_accs = 1 - alpha * np.power(np.array(target_sizes), beta)
    
    print("\nExtrapolation results:")
    print(f"Target sizes: {target_sizes}")
    print(f"Predicted accuracies: {predictions_matern.round(3)}")
    print(f"True accuracies: {true_target_accs.round(3)}")
    print(f"Mean absolute error: {np.mean(np.abs(predictions_matern - true_target_accs)):.4f}")

def compare_models():
    """Compare different models for accuracy extrapolation"""
    print("\nComparing different extrapolation models...")
    
    # Generate data
    sizes, accuracies, true_values = generate_learning_curve_data(
        min_size=100, max_size=2000, num_points=10, noise=0.01
    )
    
    # Define extrapolation range
    extrapolate_to = 50000
    
    # Define models to compare
    models = [
        {
            "name": "RBF Kernel",
            "params": {
                "model_type": "rbf",
                "mean_type": "powerlaw"
            }
        },
        {
            "name": "Matern 1/2",
            "params": {
                "model_type": "matern",
                "nu": 0.5,
                "mean_type": "powerlaw"
            }
        },
        {
            "name": "Matern 5/2",
            "params": {
                "model_type": "matern",
                "nu": 2.5,
                "mean_type": "powerlaw"
            }
        },
        {
            "name": "Beta Prior",
            "params": {
                "model_type": "matern",
                "nu": 2.5,
                "mean_type": "powerlaw",
                "use_beta_prior": True
            }
        }
    ]
    
    # Setup plot
    plt.figure(figsize=(12, 8))
    
    # Plot training data
    plt.scatter(sizes, accuracies, color='black', s=50, label='Training data')
    
    # True curve for comparison
    alpha = 0.6
    beta = -0.3
    x_true = np.logspace(np.log10(min(sizes)), np.log10(extrapolate_to), 100)
    y_true = 1 - alpha * np.power(x_true, beta)
    plt.plot(x_true, y_true, 'k--', label='True function')
    
    # Colors for different models
    colors = ['blue', 'red', 'green', 'purple']
    
    # Train and plot each model
    for i, model_config in enumerate(models):
        name = model_config["name"]
        params = model_config["params"]
        
        print(f"Training {name}...")
        
        # Create and train model
        model = AccuracyExtrapolation(**params)
        model.fit(sizes, accuracies, max_iter=1000, verbose=False)
        
        # Generate prediction points
        pred_sizes = np.logspace(np.log10(min(sizes)), np.log10(extrapolate_to), 100)
        
        # Get predictions
        mean, lower, upper = model.predict(pred_sizes)
        
        # Plot results
        plt.plot(pred_sizes, mean, color=colors[i], label=f'{name} prediction')
        plt.fill_between(pred_sizes, lower, upper, alpha=0.2, color=colors[i])
    
    # Formatting
    plt.xscale('log')
    plt.xlabel('Training set size')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Accuracy Extrapolation Models')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comparison plot saved as 'model_comparison.png'")

def real_world_example():
    """Simulate a real-world case with multiple model architectures"""
    print("\nSimulating real-world example with multiple architectures...")
    
    # Define architectures and their characteristics
    architectures = {
        "CNN": {"alpha": 0.55, "beta": -0.28, "noise": 0.015},
        "Transformer": {"alpha": 0.65, "beta": -0.32, "noise": 0.02},
        "LSTM": {"alpha": 0.58, "beta": -0.25, "noise": 0.018},
        "MLP": {"alpha": 0.62, "beta": -0.22, "noise": 0.025}
    }
    
    # Generate data for each architecture
    sizes = np.array([100, 200, 500, 1000, 2000, 5000])
    target_size = 50000
    all_data = {}
    
    plt.figure(figsize=(12, 8))
    markers = ['o', 's', '^', 'x']
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, (arch, params) in enumerate(architectures.items()):
        # Generate data points
        alpha = params["alpha"]
        beta = params["beta"]
        noise = params["noise"]
        
        true_acc = 1 - alpha * np.power(sizes, beta)
        acc_with_noise = true_acc + np.random.normal(0, noise, size=len(sizes))
        acc_with_noise = np.clip(acc_with_noise, 0.01, 0.99)
        
        all_data[arch] = {
            "sizes": sizes,
            "accuracies": acc_with_noise,
            "true_value": 1 - alpha * np.power(target_size, beta)
        }
        
        # Plot data points
        plt.scatter(
            sizes, acc_with_noise, 
            color=colors[i], marker=markers[i], s=50, 
            label=f'{arch} data'
        )
        
        # Train model using Matern kernel
        model = AccuracyExtrapolation(
            model_type="matern",
            nu=2.5,
            mean_type="powerlaw"
        )
        
        model.fit(sizes, acc_with_noise, max_iter=1000, verbose=False)
        
        # Generate prediction curve
        pred_sizes = np.logspace(np.log10(min(sizes)), np.log10(target_size*2), 100)
        mean, lower, upper = model.predict(pred_sizes)
        
        # Plot extrapolation
        plt.plot(pred_sizes, mean, color=colors[i], ls='-')
        plt.fill_between(pred_sizes, lower, upper, alpha=0.1, color=colors[i])
        
        # Predict and print for target size
        target_pred = model.predict([target_size])[0][0]
        print(f"{arch} - True accuracy at {target_size}: {all_data[arch]['true_value']:.4f}, "
              f"Predicted: {target_pred:.4f}, "
              f"Error: {abs(all_data[arch]['true_value'] - target_pred):.4f}")
    
    # Formatting
    plt.xscale('log')
    plt.xlabel('Training set size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Extrapolation for Different Model Architectures')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.savefig('architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Architecture comparison plot saved as 'architecture_comparison.png'")

def main():
    """Main function to run all examples"""
    print("PyHealth Accuracy Extrapolation Examples")
    print("=" * 50)
    
    # Run examples
    run_simple_example()
    compare_models()
    real_world_example()
    
    print("\nAll examples completed!")

if __name__ == "__main__":
    main() 
import matplotlib.pyplot as plt
import numpy as np

# --- LATEX & MATPLOTLIB CONFIGURATION ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

# --- DATA PARSING ---
alphas = [0.01, 0.05, 0.1, 0.2]
datasets = ["TUEV", "TUAB"]
methods = ["KDE CP", "KMeans CP", "Naive CP", "NCP"]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 

# Extracted directly from your logs
full_data = {
    "TUEV": {
        "KDE CP":    {0.01: (0.9723, 0.0088, 1.90, 0.56), 0.05: (0.8718, 0.0161, 1.37, 0.22), 0.1: (0.7457, 0.0072, 1.11, 0.14), 0.2: (0.5415, 0.0297, 0.89, 0.10)},
        "KMeans CP": {0.01: (0.9721, 0.0095, 1.79, 0.44), 0.05: (0.8549, 0.0219, 1.33, 0.27), 0.1: (0.7488, 0.0152, 1.23, 0.24), 0.2: (0.6328, 0.0114, 0.93, 0.16)},
        "Naive CP":  {0.01: (0.9744, 0.0069, 1.79, 0.22), 0.05: (0.8749, 0.0139, 1.31, 0.22), 0.1: (0.7461, 0.0334, 1.11, 0.21), 0.2: (0.5275, 0.0249, 0.90, 0.12)},
        "NCP":       {0.01: (0.9617, 0.0057, 1.35, 0.16), 0.05: (0.9365, 0.0083, 1.28, 0.15), 0.1: (0.9152, 0.0117, 1.25, 0.13), 0.2: (0.8722, 0.0052, 1.22, 0.17)}
    },
    "TUAB": {
        "KDE CP":    {0.01: (0.9729, 0.0065, 1.82, 0.36), 0.05: (0.8656, 0.0163, 1.30, 0.21), 0.1: (0.7447, 0.0217, 1.11, 0.19), 0.2: (0.5517, 0.0162, 0.87, 0.08)},
        "KMeans CP": {0.01: (0.9703, 0.0110, 1.89, 0.48), 0.05: (0.8652, 0.0072, 1.36, 0.25), 0.1: (0.7262, 0.0178, 1.17, 0.25), 0.2: (0.6161, 0.0307, 0.94, 0.15)},
        "Naive CP":  {0.01: (0.9715, 0.0090, 1.97, 0.37), 0.05: (0.8691, 0.0189, 1.36, 0.27), 0.1: (0.7532, 0.0165, 1.11, 0.18), 0.2: (0.5380, 0.0248, 0.90, 0.10)},
        "NCP":       {0.01: (0.9659, 0.0050, 1.38, 0.20), 0.05: (0.9348, 0.0113, 1.30, 0.18), 0.1: (0.9155, 0.0094, 1.24, 0.18), 0.2: (0.8713, 0.0098, 1.18, 0.13)}
    }
}

models = ["ContraWR", "TFM-Tokenizer"] # TFM uses ContraWR data for mapping

def plot_all_methods(metric_idx, ylabel, filename, include_target=False):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey='row')
    
    for i, dset in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[i, j]
            
            for m_idx, method in enumerate(methods):
                means = np.array([full_data[dset][method][a][metric_idx] for a in alphas])
                stds = np.array([full_data[dset][method][a][metric_idx + 1] for a in alphas])
                
                # Highlight ribbon
                ax.fill_between(alphas, means - stds, means + stds, 
                                color=colors[m_idx], alpha=0.15)
                # Primary line
                ax.plot(alphas, means, marker='o', markersize=3, 
                        color=colors[m_idx], linewidth=1.2, label=method)
            
            if include_target:
                target_cov = [1 - a for a in alphas]
                ax.plot(alphas, target_cov, linestyle='--', color='black', 
                        alpha=0.7, label=r"Target ($1-\alpha$)")

            if i == 0: ax.set_title(rf"\textbf{{{model}}}")
            if j == 0: ax.set_ylabel(rf"\textbf{{{dset}}}\\[0.5em]{ylabel}")
            if i == 1: ax.set_xlabel(r"Significance Level ($\alpha$)")
            
            ax.set_xticks(alphas)
            if i == 0 and j == 1:
                ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"File saved to: {filename}")

# Create the two figures
plot_all_methods(0, "Empirical Coverage", "coverage_plot.png", include_target=True)
plot_all_methods(2, "Avg. Prediction Set Size", "set_size_plot.png", include_target=False)

plt.show()
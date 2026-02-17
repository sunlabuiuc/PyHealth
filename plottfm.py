import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ==========================================
# 1. Configuration and Data Setup
# ==========================================

# --- LATEX & MATPLOTLIB STYLE ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 14,
    "font.size": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.titlesize": 16,
    "legend.fontsize": 13,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
})

# --- DATA POPULATION ---
alphas = [0.01, 0.05, 0.1, 0.2]
datasets_ordered = ["TUEV", "TUAB"]
methods_ordered = ["NCP", "KDE CP", "Naive CP", "KMeans CP"]

# Extracted from provided logs
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

# --- STYLING (Solid lines for methods, dashed for target) ---
style_map = {
    "NCP":       {'color': '#4e6386', 'marker': 'o'}, # Blue-Gray
    "KDE CP":    {'color': '#7393B3', 'marker': 's'}, # Lighter Blue-Gray
    "Naive CP":  {'color': '#D55E00', 'marker': '^'}, # Vermillion Orange
    "KMeans CP": {'color': '#E69F00', 'marker': 'D'}  # Light Orange
}
target_style = {'color': 'black', 'ls': '--', 'lw': 2.0}


# ==========================================
# 2. Main Plotting Function
# ==========================================
def generate_1x2_plot(metric_idx, ylabel, filename, include_target=False):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True, sharex=True)
    
    for i, dset in enumerate(datasets_ordered):
        ax = axes[i]
        
        # 1. Target Line (Dashed)
        if include_target:
             target_cov = [1 - a for a in alphas]
             ax.plot(alphas, target_cov, **target_style, zorder=1)

        # 2. CP Methods (All Solid)
        for method in methods_ordered:
            s = style_map[method]
            means = np.array([full_data[dset][method][a][metric_idx] for a in alphas])
            stds = np.array([full_data[dset][method][a][metric_idx + 1] for a in alphas])
            
            ax.fill_between(alphas, means - stds, means + stds, 
                            color=s['color'], alpha=0.15, zorder=2)
            
            # Using linestyle='-' for all methods
            ax.plot(alphas, means, color=s['color'], linestyle='-', 
                    marker=s['marker'], label=method, zorder=3)
            
        ax.set_title(rf"\textbf{{{dset} (ContraWR)}}")
        if i == 0: ax.set_ylabel(ylabel)
        ax.set_xlabel(r"Significance Level ($\alpha$)")
        ax.set_xticks(alphas)

    # 3. Legend Header
    handles = []
    if include_target:
        handles.append(mlines.Line2D([], [], **target_style, label=r'Target ($1-\alpha$)'))
    
    for method in methods_ordered:
        s = style_map[method]
        handles.append(mlines.Line2D([], [], color=s['color'], linestyle='-', 
                                     marker=s['marker'], markersize=10, 
                                     linewidth=3, label=method))

    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.91), 
               ncol=len(handles), frameon=False, handlelength=3)

    plt.tight_layout(rect=[0, 0.0, 1, 0.91])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Generated: {filename}")

if __name__ == "__main__":
    # Coverage Plot
    generate_1x2_plot(0, "Empirical Coverage", "contrawr_coverage_solid.png", True)
    # Set Size Plot
    generate_1x2_plot(2, "Avg. Prediction Set Size", "contrawr_setsize_solid.png", False)

    plt.show()
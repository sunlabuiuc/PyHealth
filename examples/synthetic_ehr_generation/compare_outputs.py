"""
Compare synthetic EHR outputs from original baselines.py vs PyHealth implementation.

This script compares the outputs from the original reproducible_synthetic_ehr
baselines with the PyHealth implementation to verify correctness.

Usage:
    python compare_outputs.py \
        --original_csv /path/to/original/great_synthetic_flattened_ehr.csv \
        --pyhealth_csv /path/to/pyhealth/great_synthetic_flattened_ehr.csv \
        --output_report comparison_report.txt
"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def load_synthetic_data(csv_path):
    """Load synthetic data CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    return df


def detect_format(df):
    """Detect if data is in long-form (sequential) or flattened (tabular) format.

    Returns:
        'long-form' if sequential format (SUBJECT_ID, HADM_ID, ICD9_CODE)
        'flattened' if tabular format (patient x codes matrix)
    """
    # Check for long-form columns
    has_subject = 'SUBJECT_ID' in df.columns
    has_hadm = 'HADM_ID' in df.columns
    has_code = 'ICD9_CODE' in df.columns

    if has_subject and has_hadm and has_code and len(df.columns) == 3:
        return 'long-form'
    else:
        return 'flattened'


def convert_longform_to_flattened(df):
    """Convert long-form EHR data to flattened patient x codes matrix."""
    # Create crosstab: count occurrences of each code per patient
    flattened = pd.crosstab(df['SUBJECT_ID'], df['ICD9_CODE'])
    return flattened


def compare_basic_statistics(original_df, pyhealth_df):
    """Compare basic statistical properties."""
    print("\n" + "=" * 80)
    print("BASIC STATISTICS COMPARISON")
    print("=" * 80)

    # Detect formats
    orig_format = detect_format(original_df)
    pyh_format = detect_format(pyhealth_df)

    print(f"\nOriginal format: {orig_format}")
    print(f"PyHealth format: {pyh_format}")

    # Convert to flattened if needed for comparison
    if orig_format == 'long-form':
        print("Converting original to flattened format...")
        original_flat = convert_longform_to_flattened(original_df)
    else:
        original_flat = original_df

    if pyh_format == 'long-form':
        print("Converting PyHealth to flattened format...")
        pyhealth_flat = convert_longform_to_flattened(pyhealth_df)
    else:
        pyhealth_flat = pyhealth_df

    stats_comparison = {
        "Metric": [],
        "Original": [],
        "PyHealth": [],
        "Difference": [],
    }

    # For long-form data, also show raw statistics
    if orig_format == 'long-form' or pyh_format == 'long-form':
        stats_comparison["Metric"].append("Total records (rows)")
        stats_comparison["Original"].append(len(original_df))
        stats_comparison["PyHealth"].append(len(pyhealth_df))
        stats_comparison["Difference"].append(abs(len(original_df) - len(pyhealth_df)))

        stats_comparison["Metric"].append("Unique patients")
        orig_patients = original_df['SUBJECT_ID'].nunique() if 'SUBJECT_ID' in original_df.columns else len(original_flat)
        pyh_patients = pyhealth_df['SUBJECT_ID'].nunique() if 'SUBJECT_ID' in pyhealth_df.columns else len(pyhealth_flat)
        stats_comparison["Original"].append(orig_patients)
        stats_comparison["PyHealth"].append(pyh_patients)
        stats_comparison["Difference"].append(abs(orig_patients - pyh_patients))

        stats_comparison["Metric"].append("Unique codes")
        orig_codes = original_df['ICD9_CODE'].nunique() if 'ICD9_CODE' in original_df.columns else len(original_flat.columns)
        pyh_codes = pyhealth_df['ICD9_CODE'].nunique() if 'ICD9_CODE' in pyhealth_df.columns else len(pyhealth_flat.columns)
        stats_comparison["Original"].append(orig_codes)
        stats_comparison["PyHealth"].append(pyh_codes)
        stats_comparison["Difference"].append(abs(orig_codes - pyh_codes))

    # Number of patients (rows in flattened)
    stats_comparison["Metric"].append("Patients (flattened rows)")
    stats_comparison["Original"].append(len(original_flat))
    stats_comparison["PyHealth"].append(len(pyhealth_flat))
    stats_comparison["Difference"].append(abs(len(original_flat) - len(pyhealth_flat)))

    # Number of features
    stats_comparison["Metric"].append("Codes (flattened cols)")
    stats_comparison["Original"].append(len(original_flat.columns))
    stats_comparison["PyHealth"].append(len(pyhealth_flat.columns))
    stats_comparison["Difference"].append(abs(len(original_flat.columns) - len(pyhealth_flat.columns)))

    # Mean values (on flattened data)
    stats_comparison["Metric"].append("Overall mean")
    orig_mean = original_flat.mean().mean()
    pyh_mean = pyhealth_flat.mean().mean()
    stats_comparison["Original"].append(f"{orig_mean:.4f}")
    stats_comparison["PyHealth"].append(f"{pyh_mean:.4f}")
    stats_comparison["Difference"].append(f"{abs(orig_mean - pyh_mean):.4f}")

    # Standard deviation
    stats_comparison["Metric"].append("Overall std")
    orig_std = original_flat.std().mean()
    pyh_std = pyhealth_flat.std().mean()
    stats_comparison["Original"].append(f"{orig_std:.4f}")
    stats_comparison["PyHealth"].append(f"{pyh_std:.4f}")
    stats_comparison["Difference"].append(f"{abs(orig_std - pyh_std):.4f}")

    # Sparsity
    stats_comparison["Metric"].append("Sparsity (% zeros)")
    orig_sparsity = (original_flat == 0).sum().sum() / (original_flat.shape[0] * original_flat.shape[1]) * 100
    pyh_sparsity = (pyhealth_flat == 0).sum().sum() / (pyhealth_flat.shape[0] * pyhealth_flat.shape[1]) * 100
    stats_comparison["Original"].append(f"{orig_sparsity:.2f}%")
    stats_comparison["PyHealth"].append(f"{pyh_sparsity:.2f}%")
    stats_comparison["Difference"].append(f"{abs(orig_sparsity - pyh_sparsity):.2f}%")

    # Print table
    comparison_df = pd.DataFrame(stats_comparison)
    print("\n" + comparison_df.to_string(index=False))

    return comparison_df


def compare_distributions(original_df, pyhealth_df):
    """Compare distributions using statistical tests."""
    print("\n" + "=" * 80)
    print("DISTRIBUTION COMPARISON")
    print("=" * 80)

    # Convert to flattened if needed
    orig_format = detect_format(original_df)
    pyh_format = detect_format(pyhealth_df)

    if orig_format == 'long-form':
        original_flat = convert_longform_to_flattened(original_df)
    else:
        original_flat = original_df

    if pyh_format == 'long-form':
        pyhealth_flat = convert_longform_to_flattened(pyhealth_df)
    else:
        pyhealth_flat = pyhealth_df

    # Find common columns
    common_cols = set(original_flat.columns) & set(pyhealth_flat.columns)
    print(f"\nCommon features: {len(common_cols)}")
    print(f"Original-only features: {len(set(original_flat.columns) - common_cols)}")
    print(f"PyHealth-only features: {len(set(pyhealth_flat.columns) - common_cols)}")

    # Sample some columns for detailed comparison
    sample_cols = list(common_cols)[:10] if len(common_cols) > 10 else list(common_cols)

    print("\n" + "-" * 80)
    print("Kolmogorov-Smirnov Test (per feature)")
    print("-" * 80)
    print(f"Testing {len(sample_cols)} sampled features...")

    ks_results = []
    for col in sample_cols:
        orig_vals = original_flat[col].values
        pyh_vals = pyhealth_flat[col].values

        # KS test
        ks_stat, ks_pval = stats.ks_2samp(orig_vals, pyh_vals)

        ks_results.append({
            "Feature": col,
            "KS Statistic": f"{ks_stat:.4f}",
            "P-value": f"{ks_pval:.4f}",
            "Significant": "Yes" if ks_pval < 0.05 else "No"
        })

    ks_df = pd.DataFrame(ks_results)
    print(ks_df.to_string(index=False))

    return ks_df


def compare_code_frequencies(original_df, pyhealth_df):
    """Compare frequency of codes."""
    print("\n" + "=" * 80)
    print("CODE FREQUENCY COMPARISON")
    print("=" * 80)

    # Convert to flattened if needed
    orig_format = detect_format(original_df)
    pyh_format = detect_format(pyhealth_df)

    if orig_format == 'long-form':
        original_flat = convert_longform_to_flattened(original_df)
    else:
        original_flat = original_df

    if pyh_format == 'long-form':
        pyhealth_flat = convert_longform_to_flattened(pyhealth_df)
    else:
        pyhealth_flat = pyhealth_df

    # Get frequencies
    orig_freq = original_flat.sum().sort_values(ascending=False)
    pyh_freq = pyhealth_flat.sum().sort_values(ascending=False)

    # Find common codes
    common_codes = set(orig_freq.index) & set(pyh_freq.index)

    print(f"\nTop 10 codes (Original):")
    print(orig_freq.head(10))

    print(f"\nTop 10 codes (PyHealth):")
    print(pyh_freq.head(10))

    # Calculate correlation of frequencies for common codes
    if len(common_codes) > 0:
        orig_common = orig_freq[list(common_codes)]
        pyh_common = pyh_freq[list(common_codes)]

        # Align by index
        combined = pd.DataFrame({
            'original': orig_common,
            'pyhealth': pyh_common
        }).fillna(0)

        correlation = combined['original'].corr(combined['pyhealth'])
        print(f"\nFrequency correlation (Pearson): {correlation:.4f}")

        return correlation

    return None


def create_visualizations(original_df, pyhealth_df, output_dir):
    """Create comparison visualizations."""
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    import os
    os.makedirs(output_dir, exist_ok=True)

    # Convert to flattened if needed
    orig_format = detect_format(original_df)
    pyh_format = detect_format(pyhealth_df)

    if orig_format == 'long-form':
        original_flat = convert_longform_to_flattened(original_df)
    else:
        original_flat = original_df

    if pyh_format == 'long-form':
        pyhealth_flat = convert_longform_to_flattened(pyhealth_df)
    else:
        pyhealth_flat = pyhealth_df

    # 1. Distribution of column means
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    orig_means = original_flat.mean()
    pyh_means = pyhealth_flat.mean()

    axes[0].hist(orig_means, bins=50, alpha=0.7, label='Original')
    axes[0].hist(pyh_means, bins=50, alpha=0.7, label='PyHealth')
    axes[0].set_xlabel('Column Mean')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Column Means')
    axes[0].legend()

    # 2. Q-Q plot of overall distributions
    orig_vals_flat = original_flat.values.flatten()
    pyh_vals_flat = pyhealth_flat.values.flatten()

    # Sample for efficiency
    sample_size = min(10000, len(orig_vals_flat), len(pyh_vals_flat))
    orig_sample = np.random.choice(orig_vals_flat, sample_size, replace=False)
    pyh_sample = np.random.choice(pyh_vals_flat, sample_size, replace=False)

    stats.probplot(orig_sample, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Original)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'), dpi=150)
    print(f"Saved: {os.path.join(output_dir, 'distribution_comparison.png')}")

    # 3. Heatmap of correlation between top codes
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top 20 codes by frequency
    top_codes_orig = original_flat.sum().nlargest(20).index
    top_codes_pyh = pyhealth_flat.sum().nlargest(20).index

    # Find common top codes
    common_top = list(set(top_codes_orig) & set(top_codes_pyh))[:15]

    if len(common_top) > 1:
        sns.heatmap(original_flat[common_top].corr(), ax=axes[0], cmap='coolwarm', center=0, vmin=-1, vmax=1)
        axes[0].set_title('Code Correlation (Original)')

        sns.heatmap(pyhealth_flat[common_top].corr(), ax=axes[1], cmap='coolwarm', center=0, vmin=-1, vmax=1)
        axes[1].set_title('Code Correlation (PyHealth)')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_comparison.png'), dpi=150)
        print(f"Saved: {os.path.join(output_dir, 'correlation_comparison.png')}")

    plt.close('all')


def generate_report(original_df, pyhealth_df, output_file):
    """Generate comprehensive comparison report."""
    print("\n" + "=" * 80)
    print("GENERATING REPORT")
    print("=" * 80)

    # Convert to flattened if needed
    orig_format = detect_format(original_df)
    pyh_format = detect_format(pyhealth_df)

    if orig_format == 'long-form':
        original_flat = convert_longform_to_flattened(original_df)
    else:
        original_flat = original_df

    if pyh_format == 'long-form':
        pyhealth_flat = convert_longform_to_flattened(pyhealth_df)
    else:
        pyhealth_flat = pyhealth_df

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SYNTHETIC EHR COMPARISON REPORT\n")
        f.write("Original baselines.py vs PyHealth Implementation\n")
        f.write("=" * 80 + "\n\n")

        # Basic info
        f.write("Dataset Information:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Original format: {orig_format}\n")
        f.write(f"PyHealth format: {pyh_format}\n")
        f.write(f"Original shape (raw): {original_df.shape}\n")
        f.write(f"PyHealth shape (raw): {pyhealth_df.shape}\n")
        f.write(f"Original shape (flattened): {original_flat.shape}\n")
        f.write(f"PyHealth shape (flattened): {pyhealth_flat.shape}\n\n")

        # Statistics
        f.write("Statistical Summary (Flattened):\n")
        f.write("-" * 80 + "\n")
        f.write("Original:\n")
        f.write(original_flat.describe().to_string() + "\n\n")
        f.write("PyHealth:\n")
        f.write(pyhealth_flat.describe().to_string() + "\n\n")

        # Validation checks
        f.write("Validation Checks:\n")
        f.write("-" * 80 + "\n")

        # Check 1: Similar dimensions
        dim_check = "✓ PASS" if abs(original_flat.shape[0] - pyhealth_flat.shape[0]) / original_flat.shape[0] < 0.01 else "✗ FAIL"
        f.write(f"{dim_check} - Similar number of rows (within 1%)\n")

        # Check 2: Similar sparsity
        orig_sparsity = (original_flat == 0).sum().sum() / (original_flat.shape[0] * original_flat.shape[1])
        pyh_sparsity = (pyhealth_flat == 0).sum().sum() / (pyhealth_flat.shape[0] * pyhealth_flat.shape[1])
        sparsity_check = "✓ PASS" if abs(orig_sparsity - pyh_sparsity) < 0.1 else "✗ FAIL"
        f.write(f"{sparsity_check} - Similar sparsity (within 10%)\n")

        # Check 3: Similar mean
        orig_mean = original_flat.mean().mean()
        pyh_mean = pyhealth_flat.mean().mean()
        mean_check = "✓ PASS" if abs(orig_mean - pyh_mean) / orig_mean < 0.2 else "✗ FAIL"
        f.write(f"{mean_check} - Similar overall mean (within 20%)\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Report generated successfully.\n")
        f.write("=" * 80 + "\n")

    print(f"Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare synthetic EHR outputs from original vs PyHealth"
    )
    parser.add_argument(
        "--original_csv",
        type=str,
        required=True,
        help="Path to original synthetic data CSV"
    )
    parser.add_argument(
        "--pyhealth_csv",
        type=str,
        required=True,
        help="Path to PyHealth synthetic data CSV"
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="comparison_report.txt",
        help="Output report file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./comparison_outputs",
        help="Directory for output visualizations"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("SYNTHETIC EHR COMPARISON")
    print("=" * 80)

    # Load data
    original_df = load_synthetic_data(args.original_csv)
    pyhealth_df = load_synthetic_data(args.pyhealth_csv)

    # Run comparisons
    basic_stats = compare_basic_statistics(original_df, pyhealth_df)
    distributions = compare_distributions(original_df, pyhealth_df)
    correlation = compare_code_frequencies(original_df, pyhealth_df)

    # Create visualizations
    create_visualizations(original_df, pyhealth_df, args.output_dir)

    # Generate report
    generate_report(original_df, pyhealth_df, args.output_report)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nReport: {args.output_report}")
    print(f"Visualizations: {args.output_dir}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze random sampling results and generate plots.

Reads the CSV from results/ directory and generates:
- Bar plots for all 200 features across layers
- Cross-layer statistics
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


def create_bar_plot(
    results_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    as_percentage: bool = False
):
    """Create bar plot with confidence intervals across layers."""
    layers = []
    means = []
    cis = []

    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]
        values = layer_df[metric_col].values

        # Convert to percentage if requested
        if as_percentage:
            values = values * 100

        # Calculate mean and 95% CI
        mean = values.mean()
        ci = stats.t.interval(
            confidence=0.95,
            df=len(values)-1,
            loc=mean,
            scale=stats.sem(values)
        )
        ci_error = mean - ci[0]

        layers.append(layer)
        means.append(mean)
        cis.append(ci_error)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(layers, means, yerr=cis, capsize=5, alpha=0.7,
                   color='steelblue', ecolor='black', linewidth=1.5)

    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path.name}")


def print_cross_layer_stats(results_df: pd.DataFrame):
    """Print cross-layer statistics for all 200 features."""
    print("\n" + "="*80)
    print("CROSS-LAYER ANALYSIS (ALL 200 FEATURES)")
    print("="*80)

    print("\nTOP-10 TOKEN APPROACH:")
    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]
        mean_rel = layer_df['relative_change_top10'].mean()
        mean_abs = layer_df['absolute_change_top10'].mean()
        print(f"  Layer {layer}: mean_rel={mean_rel:.4f} ({mean_rel*100:.1f}%), mean_abs={mean_abs:.6f}")

    print("\nALL POSITIVE TOKENS APPROACH:")
    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]
        mean_rel = layer_df['relative_change_allpos'].mean()
        mean_abs = layer_df['absolute_change_allpos'].mean()
        print(f"  Layer {layer}: mean_rel={mean_rel:.4f} ({mean_rel*100:.1f}%), mean_abs={mean_abs:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze random sampling results")
    parser.add_argument("--csv", type=str, default="results/random_sampling_results.csv",
                       help="Path to results CSV")
    parser.add_argument("--output_dir", type=str, default="results/plots",
                       help="Directory to save plots")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Layers: {sorted(df['layer'].unique())}")
    print(f"Unique features per layer: {df.groupby('layer')['feature_id'].nunique().iloc[0]}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving plots to: {output_dir}")

    # Print statistics
    print_cross_layer_stats(df)

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS (ALL 200 FEATURES)")
    print("="*80)

    # All 200 features
    print("\nAll 200 Features - Top-10 Token Approach:")
    create_bar_plot(
        df, 'relative_change_top10',
        'Mean Relative Change (%)',
        'All 200 Features: Relative Change (Top-10 Tokens)',
        output_dir / 'all200_top10_relative.png',
        as_percentage=True
    )
    create_bar_plot(
        df, 'absolute_change_top10',
        'Mean Absolute Change',
        'All 200 Features: Absolute Change (Top-10 Tokens)',
        output_dir / 'all200_top10_absolute.png'
    )

    print("\nAll 200 Features - All Positive Tokens Approach:")
    create_bar_plot(
        df, 'relative_change_allpos',
        'Mean Relative Change (%)',
        'All 200 Features: Relative Change (All Positive Tokens)',
        output_dir / 'all200_allpos_relative.png',
        as_percentage=True
    )
    create_bar_plot(
        df, 'absolute_change_allpos',
        'Mean Absolute Change',
        'All 200 Features: Absolute Change (All Positive Tokens)',
        output_dir / 'all200_allpos_absolute.png'
    )

    # Top-10 by metric value (per layer)
    print("\nTop-10 by Metric Value (per layer) - Top-10 Token Approach:")
    top10_metric_df = pd.concat([
        df[df['layer'] == layer].nlargest(10, 'relative_change_top10')
        for layer in range(1, 12)
    ])
    create_bar_plot(
        top10_metric_df, 'relative_change_top10',
        'Mean Relative Change (%)',
        'Top-10 by Metric: Relative Change (Top-10 Tokens)',
        output_dir / 'top10_bymetric_top10_relative.png',
        as_percentage=True
    )
    create_bar_plot(
        top10_metric_df, 'absolute_change_top10',
        'Mean Absolute Change',
        'Top-10 by Metric: Absolute Change (Top-10 Tokens)',
        output_dir / 'top10_bymetric_top10_absolute.png'
    )

    print("\nTop-10 by Metric Value (per layer) - All Positive Tokens Approach:")
    top10_metric_allpos_df = pd.concat([
        df[df['layer'] == layer].nlargest(10, 'relative_change_allpos')
        for layer in range(1, 12)
    ])
    create_bar_plot(
        top10_metric_allpos_df, 'relative_change_allpos',
        'Mean Relative Change (%)',
        'Top-10 by Metric: Relative Change (All Positive Tokens)',
        output_dir / 'top10_bymetric_allpos_relative.png',
        as_percentage=True
    )
    create_bar_plot(
        top10_metric_allpos_df, 'absolute_change_allpos',
        'Mean Absolute Change',
        'Top-10 by Metric: Absolute Change (All Positive Tokens)',
        output_dir / 'top10_bymetric_allpos_absolute.png'
    )

    # Top-10 by activation (per layer)
    print("\nTop-10 by Activation (per layer) - Top-10 Token Approach:")
    top10_activation_df = pd.concat([
        df[df['layer'] == layer].nlargest(10, 'activation')
        for layer in range(1, 12)
    ])
    create_bar_plot(
        top10_activation_df, 'relative_change_top10',
        'Mean Relative Change (%)',
        'Top-10 by Activation: Relative Change (Top-10 Tokens)',
        output_dir / 'top10_byactivation_top10_relative.png',
        as_percentage=True
    )
    create_bar_plot(
        top10_activation_df, 'absolute_change_top10',
        'Mean Absolute Change',
        'Top-10 by Activation: Absolute Change (Top-10 Tokens)',
        output_dir / 'top10_byactivation_top10_absolute.png'
    )

    print("\nTop-10 by Activation (per layer) - All Positive Tokens Approach:")
    create_bar_plot(
        top10_activation_df, 'relative_change_allpos',
        'Mean Relative Change (%)',
        'Top-10 by Activation: Relative Change (All Positive Tokens)',
        output_dir / 'top10_byactivation_allpos_relative.png',
        as_percentage=True
    )
    create_bar_plot(
        top10_activation_df, 'absolute_change_allpos',
        'Mean Absolute Change',
        'Top-10 by Activation: Absolute Change (All Positive Tokens)',
        output_dir / 'top10_byactivation_allpos_absolute.png'
    )

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"Generated 12 plots in: {output_dir}")
    print("\nPlots generated:")
    print("  All 200 features: 4 plots")
    print("  Top-10 by metric: 4 plots")
    print("  Top-10 by activation: 4 plots")


if __name__ == "__main__":
    main()

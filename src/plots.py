"""Plotting utilities for visualization."""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class PlotGenerator:
    """Generate plots for intervention results."""

    def __init__(
        self,
        output_dir: str,
        logger: logging.Logger = None
    ):
        """Initialize plot generator.

        Args:
            output_dir: Directory to save plots
            logger: Logger instance
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("sae_interventions")

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10

    def plot_layer_vs_metric(
        self,
        df: pd.DataFrame,
        metric_col: str,
        alpha_value: float,
        ylabel: str,
        title: str,
        output_filename: str,
        show_std: bool = True
    ) -> str:
        """Plot layer vs metric with error bars.

        Args:
            df: DataFrame with per-feature metrics
            metric_col: Column name for metric
            alpha_value: Alpha value to filter by
            ylabel: Y-axis label
            title: Plot title
            output_filename: Output filename
            show_std: Whether to show standard deviation as error bars

        Returns:
            Path to saved plot
        """
        # Filter by alpha
        df_alpha = df[df['alpha'] == alpha_value].copy()

        # Group by layer and compute mean/std
        layer_stats = df_alpha.groupby('layer')[metric_col].agg(['mean', 'std', 'count']).reset_index()

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        if show_std and 'std' in layer_stats.columns:
            ax.errorbar(
                layer_stats['layer'],
                layer_stats['mean'],
                yerr=layer_stats['std'],
                marker='o',
                linestyle='-',
                linewidth=2,
                markersize=8,
                capsize=5,
                label=f'α={alpha_value}'
            )
        else:
            ax.plot(
                layer_stats['layer'],
                layer_stats['mean'],
                marker='o',
                linestyle='-',
                linewidth=2,
                markersize=8,
                label=f'α={alpha_value}'
            )

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set integer ticks for layer axis
        ax.set_xticks(layer_stats['layer'])

        plt.tight_layout()

        # Save
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved plot: {output_path}")

        return str(output_path)

    def plot_dose_response(
        self,
        df: pd.DataFrame,
        layer: int,
        feature_id: int,
        metric_col: str,
        ylabel: str,
        title: str,
        output_filename: str
    ) -> str:
        """Plot dose-response curve (metric vs alpha) for a specific feature.

        Args:
            df: DataFrame with per-feature metrics
            layer: Layer index
            feature_id: Feature ID
            metric_col: Column name for metric
            ylabel: Y-axis label
            title: Plot title
            output_filename: Output filename

        Returns:
            Path to saved plot
        """
        # Filter for specific feature
        df_feature = df[
            (df['layer'] == layer) & (df['feature_id'] == feature_id)
        ].sort_values('alpha')

        if len(df_feature) == 0:
            self.logger.warning(
                f"No data for layer {layer}, feature {feature_id}"
            )
            return None

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(
            df_feature['alpha'],
            df_feature[metric_col],
            marker='o',
            linestyle='-',
            linewidth=2,
            markersize=8
        )

        ax.set_xlabel('Alpha (α)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add horizontal line at y=0 if applicable
        if df_feature[metric_col].min() < 0 < df_feature[metric_col].max():
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()

        # Save
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved plot: {output_path}")

        return str(output_path)

    def plot_all_dose_responses(
        self,
        df: pd.DataFrame,
        metric_col: str,
        ylabel: str,
        title: str,
        output_filename: str,
        max_features: int = 10
    ) -> str:
        """Plot dose-response curves for multiple features.

        Args:
            df: DataFrame with per-feature metrics
            metric_col: Column name for metric
            ylabel: Y-axis label
            title: Plot title
            output_filename: Output filename
            max_features: Maximum number of features to plot

        Returns:
            Path to saved plot
        """
        # Get unique (layer, feature_id) combinations
        features = df[['layer', 'feature_id']].drop_duplicates()

        # Limit number
        if len(features) > max_features:
            features = features.sample(n=max_features, random_state=42)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        for _, row in features.iterrows():
            layer = row['layer']
            feature_id = row['feature_id']

            df_feature = df[
                (df['layer'] == layer) & (df['feature_id'] == feature_id)
            ].sort_values('alpha')

            ax.plot(
                df_feature['alpha'],
                df_feature[metric_col],
                marker='o',
                linestyle='-',
                linewidth=1.5,
                markersize=5,
                label=f"L{layer}F{feature_id}",
                alpha=0.7
            )

        ax.set_xlabel('Alpha (α)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()

        # Save
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved plot: {output_path}")

        return str(output_path)

    def plot_heatmap(
        self,
        df: pd.DataFrame,
        metric_col: str,
        alpha_value: float,
        title: str,
        output_filename: str
    ) -> str:
        """Plot heatmap of metric across layers and features.

        Args:
            df: DataFrame with per-feature metrics
            metric_col: Column name for metric
            alpha_value: Alpha value to filter by
            title: Plot title
            output_filename: Output filename

        Returns:
            Path to saved plot
        """
        # Filter by alpha
        df_alpha = df[df['alpha'] == alpha_value].copy()

        # Pivot to create matrix
        pivot = df_alpha.pivot_table(
            values=metric_col,
            index='layer',
            columns='feature_id',
            aggfunc='mean'
        )

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(
            pivot,
            annot=False,
            fmt='.3f',
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': metric_col},
            ax=ax
        )

        ax.set_xlabel('Feature ID', fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved plot: {output_path}")

        return str(output_path)

    def create_summary_plots(
        self,
        per_feature_df: pd.DataFrame,
        alpha_value: float = 2.0
    ) -> List[str]:
        """Create all summary plots.

        Args:
            per_feature_df: DataFrame with per-feature metrics
            alpha_value: Alpha value for layer plots

        Returns:
            List of paths to saved plots
        """
        self.logger.info("Generating summary plots...")

        plot_paths = []

        # Layer vs Delta Loss
        path = self.plot_layer_vs_metric(
            per_feature_df,
            metric_col='d_loss',
            alpha_value=alpha_value,
            ylabel='Δ Loss (edited - clean)',
            title=f'Layer vs Delta Loss (α={alpha_value})',
            output_filename=f'layer_vs_dloss_alpha{int(alpha_value)}.png'
        )
        plot_paths.append(path)

        # Layer vs KL Divergence
        path = self.plot_layer_vs_metric(
            per_feature_df,
            metric_col='kl_avg',
            alpha_value=alpha_value,
            ylabel='KL Divergence',
            title=f'Layer vs KL Divergence (α={alpha_value})',
            output_filename=f'layer_vs_kl_alpha{int(alpha_value)}.png'
        )
        plot_paths.append(path)

        # Layer vs Mean Logit Delta
        path = self.plot_layer_vs_metric(
            per_feature_df,
            metric_col='mean_logit_delta',
            alpha_value=alpha_value,
            ylabel='Mean Logit Delta',
            title=f'Layer vs Mean Logit Delta (α={alpha_value})',
            output_filename=f'layer_vs_logit_delta_alpha{int(alpha_value)}.png'
        )
        plot_paths.append(path)

        # Dose-response for sample features
        path = self.plot_all_dose_responses(
            per_feature_df,
            metric_col='d_loss',
            ylabel='Δ Loss',
            title='Dose-Response Curves (Sample Features)',
            output_filename='dose_response_dloss.png',
            max_features=10
        )
        plot_paths.append(path)

        self.logger.info(f"Generated {len(plot_paths)} summary plots")

        return plot_paths

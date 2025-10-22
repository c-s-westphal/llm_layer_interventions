"""Markdown report generation."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class ReportGenerator:
    """Generate markdown report for experiment results."""

    def __init__(
        self,
        output_path: str,
        logger: logging.Logger = None
    ):
        """Initialize report generator.

        Args:
            output_path: Path to output markdown file
            logger: Logger instance
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("sae_interventions")

        self.sections = []

    def add_header(self, config: Dict[str, Any], env_info: Dict[str, Any]) -> None:
        """Add report header with experiment configuration.

        Args:
            config: Configuration dictionary
            env_info: Environment information dictionary
        """
        header = [
            "# SAE Intervention Experiment Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Experiment Configuration",
            "",
            f"- **Model**: {config.get('model_name', 'N/A')}",
            f"- **SAE Release**: {config.get('sae_release', 'N/A')}",
            f"- **Hook Point**: {config.get('hook', 'N/A')}",
            f"- **Layers**: {config.get('layers', 'N/A')}",
            f"- **Alpha**: {config.get('alpha', 'N/A')}",
            f"- **Alpha Sweep**: {config.get('alpha_sweep', 'N/A')}",
            f"- **Live Percentile**: P{config.get('live_percentile', 90)}",
            f"- **Corpus**: {config.get('corpus_name', 'N/A')}",
            f"- **Max Passages**: {config.get('max_passages', 'N/A')}",
            f"- **Max Length**: {config.get('max_len', 'N/A')} tokens",
            f"- **Batch Size**: {config.get('batch_size', 'N/A')}",
            f"- **Seed**: {config.get('seed', 'N/A')}",
            "",
            "## Environment",
            "",
            f"- **Python**: {env_info.get('python_version', 'N/A')}",
            f"- **PyTorch**: {env_info.get('torch_version', 'N/A')}",
            f"- **TransformerLens**: {env_info.get('transformer_lens_version', 'N/A')}",
            f"- **SAE-Lens**: {env_info.get('sae_lens_version', 'N/A')}",
            f"- **Device**: {env_info.get('device_name', 'N/A')}",
            "",
        ]

        if env_info.get('cuda_available'):
            header.append(f"- **GPU Memory**: {env_info.get('gpu_memory_total_gb', 0):.1f} GB")
            header.append("")

        self.sections.append('\n'.join(header))

    def add_feature_selection_table(self, features_df: pd.DataFrame) -> None:
        """Add table of selected features.

        Args:
            features_df: DataFrame with selected features
        """
        section = [
            "## Selected Features",
            "",
            f"Total features: {len(features_df)} across {features_df['layer'].nunique()} layers",
            "",
        ]

        # Show counts per layer
        layer_counts = features_df.groupby('layer').size()
        section.append("**Features per layer**:")
        section.append("")
        for layer, count in layer_counts.items():
            section.append(f"- Layer {layer}: {count} features")
        section.append("")

        # Show sample features
        section.append("**Sample features** (top 5 by confidence):")
        section.append("")

        top_features = features_df.nlargest(5, 'label_confidence')

        section.append("| Layer | Feature ID | Label | Confidence |")
        section.append("|-------|------------|-------|------------|")

        for _, row in top_features.iterrows():
            section.append(
                f"| {row['layer']} | {row['feature_id']} | "
                f"{row['label'][:40]} | {row['label_confidence']:.3f} |"
            )

        section.append("")

        self.sections.append('\n'.join(section))

    def add_summary_statistics(
        self,
        per_feature_df: pd.DataFrame,
        per_layer_df: pd.DataFrame,
        alpha_value: float = 2.0
    ) -> None:
        """Add summary statistics section.

        Args:
            per_feature_df: Per-feature metrics DataFrame
            per_layer_df: Per-layer summary DataFrame
            alpha_value: Alpha value for summary
        """
        # Filter to specific alpha
        df_alpha = per_feature_df[per_feature_df['alpha'] == alpha_value]

        section = [
            f"## Summary Statistics (α={alpha_value})",
            "",
            "### Overall",
            "",
            f"- **Mean Δ Loss**: {df_alpha['d_loss'].mean():.4f} ± {df_alpha['d_loss'].std():.4f}",
            f"- **Mean KL Divergence**: {df_alpha['kl_avg'].mean():.4f} ± {df_alpha['kl_avg'].std():.4f}",
            f"- **Mean Logit Delta**: {df_alpha['mean_logit_delta'].mean():.4f} ± {df_alpha['mean_logit_delta'].std():.4f}",
            f"- **Mean Firing Rate**: {df_alpha['firing_rate'].mean():.3f} ± {df_alpha['firing_rate'].std():.3f}",
            "",
            "### By Layer",
            "",
        ]

        # Per-layer summary table
        layer_summary = per_layer_df[per_layer_df['alpha'] == alpha_value].sort_values('layer')

        section.append("| Layer | Δ Loss | KL Div | Logit Δ | Firing Rate |")
        section.append("|-------|--------|--------|---------|-------------|")

        for _, row in layer_summary.iterrows():
            section.append(
                f"| {row['layer']} | "
                f"{row['mean_d_loss']:.4f} | "
                f"{row['mean_kl_avg']:.4f} | "
                f"{row['mean_logit_delta']:.4f} | "
                f"{row['mean_firing_rate']:.3f} |"
            )

        section.append("")

        self.sections.append('\n'.join(section))

    def add_plots(self, plot_paths: List[str]) -> None:
        """Add plots to report.

        Args:
            plot_paths: List of paths to plot images
        """
        section = [
            "## Visualizations",
            "",
        ]

        for plot_path in plot_paths:
            plot_name = Path(plot_path).stem.replace('_', ' ').title()
            # Use relative path from report location
            rel_path = Path(plot_path).relative_to(self.output_path.parent.parent)

            section.append(f"### {plot_name}")
            section.append("")
            section.append(f"![{plot_name}]({rel_path})")
            section.append("")

        self.sections.append('\n'.join(section))

    def add_snippets(self, snippets: List[Dict[str, Any]], max_snippets: int = 10) -> None:
        """Add example snippets to report.

        Args:
            snippets: List of snippet dictionaries
            max_snippets: Maximum number of snippets to include
        """
        section = [
            "## Example Snippets",
            "",
            "Below are examples of text windows with high and low KL divergence where features were active.",
            "",
        ]

        # Take sample snippets
        sample_snippets = snippets[:max_snippets]

        for snippet in sample_snippets:
            section.append(
                f"### Layer {snippet['layer']}, Feature {snippet['feature_id']} "
                f"(α={snippet['alpha']:.1f}, {snippet['snippet_type']})"
            )
            section.append("")
            section.append(f"**KL Divergence**: {snippet['kl_value']:.4f}")
            section.append(f"**Target Token**: `{snippet['target_token']}`")
            section.append("")
            section.append("```")
            section.append(snippet['text_window'])
            section.append("```")
            section.append("")

        self.sections.append('\n'.join(section))

    def add_key_findings(self, per_feature_df: pd.DataFrame, alpha_value: float = 2.0) -> None:
        """Add key findings section.

        Args:
            per_feature_df: Per-feature metrics DataFrame
            alpha_value: Alpha value for analysis
        """
        df_alpha = per_feature_df[per_feature_df['alpha'] == alpha_value]

        # Find most impactful features
        top_dloss = df_alpha.nlargest(3, 'd_loss')
        top_kl = df_alpha.nlargest(3, 'kl_avg')

        section = [
            "## Key Findings",
            "",
            f"### Most Impactful Features (α={alpha_value})",
            "",
            "**By Δ Loss (largest increase)**:",
            "",
        ]

        for _, row in top_dloss.iterrows():
            section.append(
                f"- Layer {row['layer']}, Feature {row['feature_id']}: "
                f"Δ Loss = {row['d_loss']:.4f}, KL = {row['kl_avg']:.4f}"
            )

        section.append("")
        section.append("**By KL Divergence (largest distribution shift)**:")
        section.append("")

        for _, row in top_kl.iterrows():
            section.append(
                f"- Layer {row['layer']}, Feature {row['feature_id']}: "
                f"KL = {row['kl_avg']:.4f}, Δ Loss = {row['d_loss']:.4f}"
            )

        section.append("")

        # Layer-wise trends
        layer_means = df_alpha.groupby('layer')['d_loss'].mean()
        max_layer = layer_means.idxmax()
        min_layer = layer_means.idxmin()

        section.append("### Layer-wise Trends")
        section.append("")
        section.append(
            f"- **Most sensitive layer**: Layer {max_layer} "
            f"(mean Δ Loss = {layer_means[max_layer]:.4f})"
        )
        section.append(
            f"- **Least sensitive layer**: Layer {min_layer} "
            f"(mean Δ Loss = {layer_means[min_layer]:.4f})"
        )
        section.append("")

        self.sections.append('\n'.join(section))

    def add_footer(self) -> None:
        """Add report footer."""
        footer = [
            "---",
            "",
            "Report generated by SAE Intervention Pipeline",
            ""
        ]

        self.sections.append('\n'.join(footer))

    def save(self) -> None:
        """Save report to file."""
        content = '\n'.join(self.sections)

        with open(self.output_path, 'w') as f:
            f.write(content)

        self.logger.info(f"Saved report to {self.output_path}")

    def generate_full_report(
        self,
        config: Dict[str, Any],
        env_info: Dict[str, Any],
        features_df: pd.DataFrame,
        per_feature_df: pd.DataFrame,
        per_layer_df: pd.DataFrame,
        plot_paths: List[str],
        snippets: List[Dict[str, Any]],
        alpha_value: float = 2.0
    ) -> None:
        """Generate complete report.

        Args:
            config: Configuration dictionary
            env_info: Environment information
            features_df: Selected features DataFrame
            per_feature_df: Per-feature metrics DataFrame
            per_layer_df: Per-layer summary DataFrame
            plot_paths: List of plot paths
            snippets: List of snippet dictionaries
            alpha_value: Alpha value for summaries
        """
        self.logger.info("Generating report...")

        self.add_header(config, env_info)
        self.add_feature_selection_table(features_df)
        self.add_summary_statistics(per_feature_df, per_layer_df, alpha_value)
        self.add_key_findings(per_feature_df, alpha_value)
        self.add_plots(plot_paths)
        self.add_snippets(snippets, max_snippets=10)
        self.add_footer()

        self.save()

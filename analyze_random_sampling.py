#!/usr/bin/env python3
"""
Random Feature Sampling Ablation Analysis

For each layer:
1. Randomly sample 60 features (same 60 for all positions in that layer)
2. Randomly sample 25% of positions in the corpus
3. Ablate each of the 60 features at the sampled positions
4. Measure both relative and absolute probability change
5. Track activation magnitude and co-activation count

Reports per layer:
- Mean across all 60 random features at all sampled positions
- Mean of top-K by metric value (relative change) for K=5,10,20
- Mean of top-K by activation value for K=5,10,20
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import yaml
from tqdm import tqdm

from src.data import CorpusLoader, collate_batch
from src.model import ModelLoader


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to both file and console."""
    logger = logging.getLogger("random_sampling")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(output_dir / "experiment.log")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def create_reconstruction_hook(sae):
    """Create hook for SAE reconstruction."""
    def hook_fn(activations, hook):
        return sae.decode(sae.encode(activations))
    return hook_fn


def create_ablation_hook(sae, feature_id: int):
    """Create hook for ablating a specific feature."""
    def hook_fn(activations, hook):
        sae_acts = sae.encode(activations)
        sae_acts[:, :, feature_id] = 0
        return sae.decode(sae_acts)
    return hook_fn


def sample_positions(
    data: List[torch.Tensor],
    batch_size: int,
    sample_fraction: float,
    logger: logging.Logger
) -> List[Tuple[int, int, int]]:
    """
    Randomly sample a fraction of valid positions from the corpus.

    Returns:
        List of (batch_idx, batch_pos, seq_pos) tuples
    """
    logger.info(f"  Sampling {sample_fraction*100}% of positions...")

    all_positions = []
    num_batches = (len(data) + batch_size - 1) // batch_size

    # Collect all valid positions
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(data))
        batch_tokens = data[batch_start:batch_end]
        batch_dict = collate_batch(batch_tokens, device=torch.device("cpu"))  # Just for counting

        attention_mask = batch_dict["attention_mask"][:, :-1].bool()
        for b in range(attention_mask.shape[0]):
            for s in range(attention_mask.shape[1]):
                if attention_mask[b, s]:
                    all_positions.append((batch_idx, b, s))

    # Randomly sample
    num_sample = int(len(all_positions) * sample_fraction)
    sampled_indices = np.random.choice(len(all_positions), num_sample, replace=False)
    sampled_positions = [all_positions[i] for i in sampled_indices]

    logger.info(f"    Total positions: {len(all_positions):,}")
    logger.info(f"    Sampled positions: {len(sampled_positions):,}")

    return sampled_positions


def measure_ablation_effects(
    model,
    sae,
    layer: int,
    hook: str,
    feature_ids: List[int],
    sampled_positions: List[Tuple[int, int, int]],
    data: List[torch.Tensor],
    batch_size: int,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Ablate each feature at the sampled positions and measure effects.

    Returns:
        DataFrame with columns: feature_id, activation, co_activation_count,
                                relative_change, absolute_change
    """
    hook_name = f"blocks.{layer}.hook_{hook}"

    all_measurements = []

    logger.info(f"  Measuring ablation effects for {len(feature_ids)} features at {len(sampled_positions):,} positions...")

    # Group positions by batch
    positions_by_batch = defaultdict(list)
    for batch_idx, batch_pos, seq_pos in sampled_positions:
        positions_by_batch[batch_idx].append((batch_pos, seq_pos))

    for batch_idx in tqdm(sorted(positions_by_batch.keys()), desc=f"Layer {layer} - Ablating"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(data))
        batch_tokens = data[batch_start:batch_end]
        batch_dict = collate_batch(batch_tokens, device=model.cfg.device)

        # Get feature activations for this batch
        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch_dict["input_ids"],
                names_filter=[hook_name]
            )
            acts = cache[hook_name]
            sae_acts = sae.encode(acts)  # [batch, seq, d_sae]

        # Baseline: SAE reconstruction
        reconstruction_hook = create_reconstruction_hook(sae)
        with torch.no_grad():
            baseline_logits = model.run_with_hooks(
                batch_dict["input_ids"],
                fwd_hooks=[(hook_name, reconstruction_hook)]
            )
            baseline_probs = torch.softmax(baseline_logits[:, :-1, :], dim=-1)

        # Ablate each of the 60 random features
        for feature_id in feature_ids:
            ablation_hook = create_ablation_hook(sae, feature_id)

            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    batch_dict["input_ids"],
                    fwd_hooks=[(hook_name, ablation_hook)]
                )
                ablated_probs = torch.softmax(ablated_logits[:, :-1, :], dim=-1)

            # Measure at sampled positions in this batch
            for batch_pos, seq_pos in positions_by_batch[batch_idx]:
                # Get activation and co-activation count
                feature_act = sae_acts[batch_pos, seq_pos, feature_id].item()
                co_activation_count = (sae_acts[batch_pos, seq_pos] > 0).sum().item()

                # Get probability distributions
                baseline_prob_dist = baseline_probs[batch_pos, seq_pos]
                ablated_prob_dist = ablated_probs[batch_pos, seq_pos]

                # Compute both relative and absolute change
                absolute_change = (baseline_prob_dist - ablated_prob_dist)
                relative_change = absolute_change / (baseline_prob_dist + 1e-10)

                # Approach 1: Top-10 token (max relative change)
                top_10_rel = relative_change.topk(10).values.mean().item()
                top_10_abs = absolute_change.topk(10).values.mean().item()

                # Approach 2: All positive tokens (mean of all positive changes)
                positive_mask = relative_change > 0
                if positive_mask.any():
                    all_pos_rel = relative_change[positive_mask].mean().item()
                    all_pos_abs = absolute_change[positive_mask].mean().item()
                else:
                    all_pos_rel = 0.0
                    all_pos_abs = 0.0

                all_measurements.append({
                    'feature_id': feature_id,
                    'activation': feature_act,
                    'co_activation_count': co_activation_count,
                    'relative_change_top10': top_10_rel,
                    'absolute_change_top10': top_10_abs,
                    'relative_change_allpos': all_pos_rel,
                    'absolute_change_allpos': all_pos_abs
                })

    return pd.DataFrame(all_measurements)


def create_bar_plot(
    results_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    logger: logging.Logger
):
    """Create bar plot with confidence intervals across layers.

    Args:
        results_df: DataFrame with results for all layers
        metric_col: Column name for the metric to plot
        ylabel: Y-axis label
        title: Plot title
        output_path: Path to save the plot
        logger: Logger instance
    """
    layers = []
    means = []
    cis = []

    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]
        values = layer_df[metric_col].values

        # Calculate mean and 95% CI
        mean = values.mean()
        # Use scipy.stats to calculate confidence interval
        ci = stats.t.interval(
            confidence=0.95,
            df=len(values)-1,
            loc=mean,
            scale=stats.sem(values)
        )
        ci_error = mean - ci[0]  # Error bar size (symmetric)

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

    logger.info(f"  Saved: {output_path.name}")


def generate_all_plots(
    results_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
):
    """Generate all 28 bar plots with confidence intervals.

    Args:
        results_df: DataFrame with all results
        output_dir: Directory to save plots
        logger: Logger instance
    """
    logger.info("\n" + "="*80)
    logger.info("GENERATING PLOTS")
    logger.info("="*80)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # TOP-10 TOKEN APPROACH
    logger.info("\nTop-10 Token Approach plots:")

    # Overall (all 60 features)
    create_bar_plot(
        results_df, 'relative_change_top10',
        'Mean Relative Change', 'Overall: Relative Change (Top-10 Tokens)',
        plots_dir / 'top10_overall_relative.png', logger
    )
    create_bar_plot(
        results_df, 'absolute_change_top10',
        'Mean Absolute Change', 'Overall: Absolute Change (Top-10 Tokens)',
        plots_dir / 'top10_overall_absolute.png', logger
    )

    # Top-K by Metric Value
    for k in [5, 10, 20]:
        # Select top-K features per layer by metric
        top_k_df = pd.concat([
            results_df[results_df['layer'] == layer].nlargest(k, 'relative_change_top10')
            for layer in range(1, 12)
        ])

        create_bar_plot(
            top_k_df, 'relative_change_top10',
            'Mean Relative Change', f'Top-{k} by Metric: Relative Change (Top-10 Tokens)',
            plots_dir / f'top10_top{k}_bymetric_relative.png', logger
        )
        create_bar_plot(
            top_k_df, 'absolute_change_top10',
            'Mean Absolute Change', f'Top-{k} by Metric: Absolute Change (Top-10 Tokens)',
            plots_dir / f'top10_top{k}_bymetric_absolute.png', logger
        )

    # Top-K by Activation Value
    for k in [5, 10, 20]:
        # Select top-K features per layer by activation
        top_k_df = pd.concat([
            results_df[results_df['layer'] == layer].nlargest(k, 'activation')
            for layer in range(1, 12)
        ])

        create_bar_plot(
            top_k_df, 'relative_change_top10',
            'Mean Relative Change', f'Top-{k} by Activation: Relative Change (Top-10 Tokens)',
            plots_dir / f'top10_top{k}_byactivation_relative.png', logger
        )
        create_bar_plot(
            top_k_df, 'absolute_change_top10',
            'Mean Absolute Change', f'Top-{k} by Activation: Absolute Change (Top-10 Tokens)',
            plots_dir / f'top10_top{k}_byactivation_absolute.png', logger
        )

    # ALL POSITIVE TOKENS APPROACH
    logger.info("\nAll Positive Tokens Approach plots:")

    # Overall (all 60 features)
    create_bar_plot(
        results_df, 'relative_change_allpos',
        'Mean Relative Change', 'Overall: Relative Change (All Positive Tokens)',
        plots_dir / 'allpos_overall_relative.png', logger
    )
    create_bar_plot(
        results_df, 'absolute_change_allpos',
        'Mean Absolute Change', 'Overall: Absolute Change (All Positive Tokens)',
        plots_dir / 'allpos_overall_absolute.png', logger
    )

    # Top-K by Metric Value
    for k in [5, 10, 20]:
        # Select top-K features per layer by metric
        top_k_df = pd.concat([
            results_df[results_df['layer'] == layer].nlargest(k, 'relative_change_allpos')
            for layer in range(1, 12)
        ])

        create_bar_plot(
            top_k_df, 'relative_change_allpos',
            'Mean Relative Change', f'Top-{k} by Metric: Relative Change (All Positive Tokens)',
            plots_dir / f'allpos_top{k}_bymetric_relative.png', logger
        )
        create_bar_plot(
            top_k_df, 'absolute_change_allpos',
            'Mean Absolute Change', f'Top-{k} by Metric: Absolute Change (All Positive Tokens)',
            plots_dir / f'allpos_top{k}_bymetric_absolute.png', logger
        )

    # Top-K by Activation Value
    for k in [5, 10, 20]:
        # Select top-K features per layer by activation
        top_k_df = pd.concat([
            results_df[results_df['layer'] == layer].nlargest(k, 'activation')
            for layer in range(1, 12)
        ])

        create_bar_plot(
            top_k_df, 'relative_change_allpos',
            'Mean Relative Change', f'Top-{k} by Activation: Relative Change (All Positive Tokens)',
            plots_dir / f'allpos_top{k}_byactivation_relative.png', logger
        )
        create_bar_plot(
            top_k_df, 'absolute_change_allpos',
            'Mean Absolute Change', f'Top-{k} by Activation: Absolute Change (All Positive Tokens)',
            plots_dir / f'allpos_top{k}_byactivation_absolute.png', logger
        )

    logger.info(f"\nGenerated 28 plots in: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="Random feature sampling ablation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num_features", type=int, default=200,
                       help="Number of random features to sample per layer")
    parser.add_argument("--position_fraction", type=float, default=0.25,
                       help="Fraction of positions to sample (default: 0.25 = 25%)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup output
    output_dir = Path(config["output_dir"]) / "random_sampling"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("="*80)
    logger.info(f"RANDOM FEATURE SAMPLING ABLATION")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Features per layer: {args.num_features}")
    logger.info(f"Position sampling: {args.position_fraction*100}%")
    logger.info(f"Output directory: {output_dir}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Load model and SAEs
    logger.info("Loading model and SAEs...")
    loader = ModelLoader(
        model_name=config["model_name"],
        sae_release=config["sae_release"],
        hook=config["hook"],
        layers=list(range(1, 12)),  # Layers 1-11
        device=device,
        logger=logger
    )

    model = loader.load_model()
    saes = loader.load_saes()

    # Load corpus
    logger.info("Loading corpus...")
    corpus_loader = CorpusLoader(
        corpus_name=config["corpus_name"],
        max_passages=config.get("test_passages", config["max_passages"]),
        max_len=config["max_len"],
        tokenizer=model.tokenizer,
        logger=logger
    )
    data, _ = corpus_loader.load_and_tokenize()
    logger.info(f"Loaded {len(data)} passages")

    # Sample positions once (same positions for all layers)
    sampled_positions = sample_positions(
        data, config["batch_size"], args.position_fraction, logger
    )

    # Process each layer
    all_results = []

    for layer in range(1, 12):
        logger.info(f"\n{'='*80}")
        logger.info(f"LAYER {layer}")
        logger.info(f"{'='*80}")

        sae = saes[layer]
        d_sae = sae.cfg.d_sae

        # Randomly sample features for this layer
        random_feature_ids = np.random.choice(d_sae, size=args.num_features, replace=False).tolist()
        logger.info(f"  Randomly selected {args.num_features} features: {random_feature_ids[:10]}... (showing first 10)")

        # Measure ablation effects
        layer_results = measure_ablation_effects(
            model, sae, layer, config["hook"], random_feature_ids,
            sampled_positions, data, config["batch_size"], logger
        )

        layer_results['layer'] = layer
        all_results.append(layer_results)

        logger.info(f"  Completed {len(layer_results)} measurements for layer {layer}")

    # Combine all results
    results_df = pd.concat(all_results, ignore_index=True)

    # Save raw results
    results_path = output_dir / "random_sampling_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nRaw results saved to: {results_path}")

    # Analysis and reporting
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS")
    logger.info("="*80)

    # Per-layer detailed analysis
    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]

        logger.info(f"\n{'='*80}")
        logger.info(f"LAYER {layer}")
        logger.info(f"{'='*80}")

        logger.info(f"\nOverall Statistics (all {args.num_features} random features):")
        logger.info(f"  Total measurements: {len(layer_df):,}")

        logger.info(f"\n  TOP-10 TOKEN APPROACH:")
        logger.info(f"    Mean relative change: {layer_df['relative_change_top10'].mean():.4f} ({layer_df['relative_change_top10'].mean()*100:.1f}%)")
        logger.info(f"    Mean absolute change: {layer_df['absolute_change_top10'].mean():.6f}")
        logger.info(f"    Median relative change: {layer_df['relative_change_top10'].median():.4f}")

        logger.info(f"\n  ALL POSITIVE TOKENS APPROACH:")
        logger.info(f"    Mean relative change: {layer_df['relative_change_allpos'].mean():.4f} ({layer_df['relative_change_allpos'].mean()*100:.1f}%)")
        logger.info(f"    Mean absolute change: {layer_df['absolute_change_allpos'].mean():.6f}")
        logger.info(f"    Median relative change: {layer_df['relative_change_allpos'].median():.4f}")

    # Cross-layer analysis
    logger.info("\n" + "="*80)
    logger.info("CROSS-LAYER ANALYSIS")
    logger.info("="*80)

    logger.info("\n" + "="*80)
    logger.info("TOP-10 TOKEN APPROACH")
    logger.info("="*80)

    logger.info(f"\nTop-K by Metric Value (Relative Change):")
    logger.info(f"\n  Top-5 features per layer:")
    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]
        top_k = layer_df.nlargest(5, 'relative_change_top10')
        logger.info(
            f"    Layer {layer}: mean_rel={top_k['relative_change_top10'].mean():.4f} ({top_k['relative_change_top10'].mean()*100:.1f}%), "
            f"mean_abs={top_k['absolute_change_top10'].mean():.6f}"
        )

    logger.info(f"\n  Top-10 features per layer:")
    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]
        top_k = layer_df.nlargest(10, 'relative_change_top10')
        logger.info(
            f"    Layer {layer}: mean_rel={top_k['relative_change_top10'].mean():.4f} ({top_k['relative_change_top10'].mean()*100:.1f}%), "
            f"mean_abs={top_k['absolute_change_top10'].mean():.6f}"
        )

    logger.info(f"\n  Top-20 features per layer:")
    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]
        top_k = layer_df.nlargest(20, 'relative_change_top10')
        logger.info(
            f"    Layer {layer}: mean_rel={top_k['relative_change_top10'].mean():.4f} ({top_k['relative_change_top10'].mean()*100:.1f}%), "
            f"mean_abs={top_k['absolute_change_top10'].mean():.6f}"
        )

    logger.info(f"\nTop-K by Activation Value:")
    logger.info(f"\n  Top-5 features per layer (by activation):")
    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]
        top_k = layer_df.nlargest(5, 'activation')
        logger.info(
            f"    Layer {layer}: mean_rel={top_k['relative_change_top10'].mean():.4f} ({top_k['relative_change_top10'].mean()*100:.1f}%), "
            f"mean_abs={top_k['absolute_change_top10'].mean():.6f}, mean_activation={top_k['activation'].mean():.2f}"
        )

    logger.info(f"\n  Top-10 features per layer (by activation):")
    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]
        top_k = layer_df.nlargest(10, 'activation')
        logger.info(
            f"    Layer {layer}: mean_rel={top_k['relative_change_top10'].mean():.4f} ({top_k['relative_change_top10'].mean()*100:.1f}%), "
            f"mean_abs={top_k['absolute_change_top10'].mean():.6f}, mean_activation={top_k['activation'].mean():.2f}"
        )

    logger.info(f"\n  Top-20 features per layer (by activation):")
    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]
        top_k = layer_df.nlargest(20, 'activation')
        logger.info(
            f"    Layer {layer}: mean_rel={top_k['relative_change_top10'].mean():.4f} ({top_k['relative_change_top10'].mean()*100:.1f}%), "
            f"mean_abs={top_k['absolute_change_top10'].mean():.6f}, mean_activation={top_k['activation'].mean():.2f}"
        )

    logger.info("\n" + "="*80)
    logger.info("ALL POSITIVE TOKENS APPROACH")
    logger.info("="*80)

    logger.info(f"\nTop-K by Metric Value (Relative Change):")
    logger.info(f"\n  Top-5 features per layer:")
    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]
        top_k = layer_df.nlargest(5, 'relative_change_allpos')
        logger.info(
            f"    Layer {layer}: mean_rel={top_k['relative_change_allpos'].mean():.4f} ({top_k['relative_change_allpos'].mean()*100:.1f}%), "
            f"mean_abs={top_k['absolute_change_allpos'].mean():.6f}"
        )

    logger.info(f"\n  Top-10 features per layer:")
    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]
        top_k = layer_df.nlargest(10, 'relative_change_allpos')
        logger.info(
            f"    Layer {layer}: mean_rel={top_k['relative_change_allpos'].mean():.4f} ({top_k['relative_change_allpos'].mean()*100:.1f}%), "
            f"mean_abs={top_k['absolute_change_allpos'].mean():.6f}"
        )

    logger.info(f"\n  Top-20 features per layer:")
    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]
        top_k = layer_df.nlargest(20, 'relative_change_allpos')
        logger.info(
            f"    Layer {layer}: mean_rel={top_k['relative_change_allpos'].mean():.4f} ({top_k['relative_change_allpos'].mean()*100:.1f}%), "
            f"mean_abs={top_k['absolute_change_allpos'].mean():.6f}"
        )

    logger.info(f"\nTop-K by Activation Value:")
    logger.info(f"\n  Top-5 features per layer (by activation):")
    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]
        top_k = layer_df.nlargest(5, 'activation')
        logger.info(
            f"    Layer {layer}: mean_rel={top_k['relative_change_allpos'].mean():.4f} ({top_k['relative_change_allpos'].mean()*100:.1f}%), "
            f"mean_abs={top_k['absolute_change_allpos'].mean():.6f}, mean_activation={top_k['activation'].mean():.2f}"
        )

    logger.info(f"\n  Top-10 features per layer (by activation):")
    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]
        top_k = layer_df.nlargest(10, 'activation')
        logger.info(
            f"    Layer {layer}: mean_rel={top_k['relative_change_allpos'].mean():.4f} ({top_k['relative_change_allpos'].mean()*100:.1f}%), "
            f"mean_abs={top_k['absolute_change_allpos'].mean():.6f}, mean_activation={top_k['activation'].mean():.2f}"
        )

    logger.info(f"\n  Top-20 features per layer (by activation):")
    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]
        top_k = layer_df.nlargest(20, 'activation')
        logger.info(
            f"    Layer {layer}: mean_rel={top_k['relative_change_allpos'].mean():.4f} ({top_k['relative_change_allpos'].mean()*100:.1f}%), "
            f"mean_abs={top_k['absolute_change_allpos'].mean():.6f}, mean_activation={top_k['activation'].mean():.2f}"
        )

    # Generate all plots
    generate_all_plots(results_df, output_dir, logger)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()

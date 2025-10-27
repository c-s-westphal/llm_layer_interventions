#!/usr/bin/env python3
"""
Exhaustive Feature Ablation Analysis (1% Position Sampling)

For each sampled position:
- Ablates EVERY feature individually
- Measures relative probability change
- Tracks activation magnitude and co-activation count

Reports:
- Mean across all measurements
- Mean of top-20 by metric value
- Mean of top-k by activation value (k=5,10,20,50)
- Breakdown by activation bins
- Breakdown by co-activation count bins
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from src.data import CorpusLoader, collate_batch
from src.model import ModelLoader


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to both file and console."""
    logger = logging.getLogger("exhaustive_ablation")
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


def exhaustive_ablation_analysis(
    model,
    sae,
    layer: int,
    hook: str,
    sampled_positions: List[Tuple[int, int, int]],  # (batch_idx, batch_pos, seq_pos)
    data: List[torch.Tensor],
    batch_size: int,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Ablate every feature at every sampled position.

    Returns:
        DataFrame with columns: feature_id, activation, co_activation_count, relative_change
    """
    hook_name = f"blocks.{layer}.hook_{hook}"
    d_sae = sae.cfg.d_sae

    # Group positions by batch
    from collections import defaultdict
    positions_by_batch = defaultdict(list)
    for batch_idx, batch_pos, seq_pos in sampled_positions:
        positions_by_batch[batch_idx].append((batch_pos, seq_pos))

    all_measurements = []

    logger.info(f"  Processing {len(positions_by_batch)} batches with {len(sampled_positions)} positions...")

    for batch_idx, batch_positions in tqdm(list(positions_by_batch.items()), desc=f"Layer {layer}"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(data))
        batch_tokens = data[batch_start:batch_end]
        batch_dict = collate_batch(batch_tokens, device=model.cfg.device)

        # Get feature activations for co-activation count
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

        # For each feature, ablate and measure
        for feature_id in tqdm(range(d_sae), desc=f"Batch {batch_idx} - Features", leave=False):
            ablation_hook = create_ablation_hook(sae, feature_id)

            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    batch_dict["input_ids"],
                    fwd_hooks=[(hook_name, ablation_hook)]
                )
                ablated_probs = torch.softmax(ablated_logits[:, :-1, :], dim=-1)

            # Measure at each sampled position in this batch
            for batch_pos, seq_pos in batch_positions:
                # Get activation and co-activation count
                feature_act = sae_acts[batch_pos, seq_pos, feature_id].item()
                co_activation_count = (sae_acts[batch_pos, seq_pos] > 0).sum().item()

                # Get probability distributions
                baseline_prob_dist = baseline_probs[batch_pos, seq_pos]
                ablated_prob_dist = ablated_probs[batch_pos, seq_pos]

                # Compute relative change for top promoted token
                relative_change = (baseline_prob_dist - ablated_prob_dist) / (baseline_prob_dist + 1e-10)
                top_change = relative_change.max().item()

                all_measurements.append({
                    'feature_id': feature_id,
                    'activation': feature_act,
                    'co_activation_count': co_activation_count,
                    'relative_change': top_change
                })

    return pd.DataFrame(all_measurements)


def main():
    parser = argparse.ArgumentParser(description="Exhaustive ablation with 1% sampling")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--sample_rate", type=float, default=0.01, help="Fraction of positions to sample")
    parser.add_argument("--layer", type=int, default=5, help="Which layer to analyze (default: 5)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup output
    output_dir = Path(config["output_dir"]) / "exhaustive_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("="*80)
    logger.info(f"EXHAUSTIVE ABLATION ANALYSIS (Layer {args.layer}, {args.sample_rate*100}% sampling)")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Sample rate: {args.sample_rate}")
    logger.info(f"Output directory: {output_dir}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Load model and SAE for target layer
    logger.info(f"Loading model and SAE for layer {args.layer}...")
    loader = ModelLoader(
        model_name=config["model_name"],
        sae_release=config["sae_release"],
        hook=config["hook"],
        layers=[args.layer],
        device=device,
        logger=logger
    )

    model = loader.load_model()
    saes = loader.load_saes()
    sae = saes[args.layer]

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

    # Sample positions
    logger.info(f"\nSampling {args.sample_rate*100}% of positions...")
    all_positions = []
    batch_size = config["batch_size"]
    num_batches = (len(data) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(data))
        batch_tokens = data[batch_start:batch_end]
        batch_dict = collate_batch(batch_tokens, device=model.cfg.device)

        attention_mask = batch_dict["attention_mask"][:, :-1]
        for b in range(attention_mask.shape[0]):
            for s in range(attention_mask.shape[1]):
                if attention_mask[b, s]:
                    all_positions.append((batch_idx, b, s))

    # Random sample
    num_sample = int(len(all_positions) * args.sample_rate)
    sampled_indices = np.random.choice(len(all_positions), num_sample, replace=False)
    sampled_positions = [all_positions[i] for i in sampled_indices]

    logger.info(f"Total positions: {len(all_positions)}")
    logger.info(f"Sampled positions: {len(sampled_positions)}")
    logger.info(f"Features to ablate: {sae.cfg.d_sae}")
    logger.info(f"Total measurements: {len(sampled_positions) * sae.cfg.d_sae:,}")

    # Run exhaustive ablation
    logger.info(f"\nRunning exhaustive ablation on layer {args.layer}...")
    results_df = exhaustive_ablation_analysis(
        model, sae, args.layer, config["hook"],
        sampled_positions, data, batch_size, logger
    )

    # Save raw results
    results_path = output_dir / f"layer_{args.layer}_exhaustive_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nRaw results saved to: {results_path}")

    # Analysis and reporting
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS")
    logger.info("="*80)

    logger.info(f"\nOverall Statistics:")
    logger.info(f"  Total measurements: {len(results_df):,}")
    logger.info(f"  Mean relative change (all): {results_df['relative_change'].mean():.4f} ({results_df['relative_change'].mean()*100:.1f}%)")
    logger.info(f"  Median relative change: {results_df['relative_change'].median():.4f}")
    logger.info(f"  Std relative change: {results_df['relative_change'].std():.4f}")

    logger.info(f"\nTop-K by Metric Value:")
    for k in [20, 50, 100]:
        top_k = results_df.nlargest(k, 'relative_change')
        logger.info(f"  Top-{k} mean: {top_k['relative_change'].mean():.4f} ({top_k['relative_change'].mean()*100:.1f}%)")

    logger.info(f"\nTop-K by Activation Value:")
    for k in [5, 10, 20, 50]:
        top_k = results_df.nlargest(k, 'activation')
        logger.info(f"  Top-{k} by activation: mean_metric={top_k['relative_change'].mean():.4f} ({top_k['relative_change'].mean()*100:.1f}%), mean_activation={top_k['activation'].mean():.2f}")

    logger.info(f"\nBy Activation Bins:")
    bins = [0, 1, 5, 10, 20, 50, 100, np.inf]
    results_df['activation_bin'] = pd.cut(results_df['activation'], bins=bins)
    for bin_label, group in results_df.groupby('activation_bin'):
        logger.info(f"  Activation {bin_label}: n={len(group):,}, mean_metric={group['relative_change'].mean():.4f}")

    logger.info(f"\nBy Co-activation Count Bins:")
    co_bins = [0, 50, 100, 200, 500, 1000, np.inf]
    results_df['co_activation_bin'] = pd.cut(results_df['co_activation_count'], bins=co_bins)
    for bin_label, group in results_df.groupby('co_activation_bin'):
        logger.info(f"  Co-activation {bin_label}: n={len(group):,}, mean_metric={group['relative_change'].mean():.4f}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()

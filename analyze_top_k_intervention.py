#!/usr/bin/env python3
"""
Top-K Feature Ablation Analysis

Instead of pre-selecting features, this script:
1. At each position in the corpus, identifies the top-K strongest firing features
2. For each unique feature that appears in top-K somewhere:
   - Ablates it at all positions where it was in top-K
   - Measures the probability change for promoted tokens

This is more data-driven than pre-selecting features by composite score.
"""

import argparse
import logging
from collections import defaultdict
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
    logger = logging.getLogger("top_k_intervention")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(output_dir / "experiment.log")
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
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


def identify_top_k_features_per_position(
    model,
    sae,
    layer: int,
    hook: str,
    data: List[torch.Tensor],
    batch_size: int,
    top_k: int,
    logger: logging.Logger
) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    Identify top-K features at each position in the corpus.

    Returns:
        Dictionary mapping feature_id -> list of (batch_idx, batch_pos, seq_pos) tuples
        where that feature was in the top-K
    """
    logger.info(f"  Pass 1: Identifying top-{top_k} features at each position...")

    hook_name = f"blocks.{layer}.hook_{hook}"
    feature_positions = defaultdict(list)

    num_batches = (len(data) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"Layer {layer} - Finding top-{top_k}"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(data))
        batch_tokens = data[batch_start:batch_end]
        batch_dict = collate_batch(batch_tokens, device=model.cfg.device)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch_dict["input_ids"],
                names_filter=[hook_name]
            )
            acts = cache[hook_name]
            sae_acts = sae.encode(acts)  # [batch, seq, d_sae]

            # Get attention mask
            attention_mask = batch_dict["attention_mask"][:, :-1].bool()  # [batch, seq-1]
            sae_acts_valid = sae_acts[:, :-1, :]  # [batch, seq-1, d_sae]

            # For each valid position, get top-K features
            batch_s, seq_s, d_sae = sae_acts_valid.shape
            for b in range(batch_s):
                for s in range(seq_s):
                    if not attention_mask[b, s]:
                        continue

                    # Get top-K features at this position
                    acts_at_pos = sae_acts_valid[b, s]  # [d_sae]
                    top_k_values, top_k_indices = torch.topk(acts_at_pos, top_k)

                    # Only include features with non-zero activation
                    for feat_id, feat_val in zip(top_k_indices.tolist(), top_k_values.tolist()):
                        if feat_val > 0:
                            feature_positions[feat_id].append((batch_idx, b, s))

    logger.info(f"    Found {len(feature_positions)} unique features in top-{top_k}")

    return dict(feature_positions)


def measure_feature_ablation_effect(
    model,
    sae,
    layer: int,
    feature_id: int,
    hook: str,
    positions: List[Tuple[int, int, int]],
    data: List[torch.Tensor],
    batch_size: int,
    logger: logging.Logger
) -> Dict:
    """
    Measure the effect of ablating a feature at specific positions.

    Args:
        positions: List of (batch_idx, batch_pos, seq_pos) where feature was in top-K

    Returns:
        Dictionary with statistics about probability changes
    """
    hook_name = f"blocks.{layer}.hook_{hook}"

    # Group positions by batch for efficient processing
    positions_by_batch = defaultdict(list)
    for batch_idx, batch_pos, seq_pos in positions:
        positions_by_batch[batch_idx].append((batch_pos, seq_pos))

    all_prob_changes = []

    for batch_idx, batch_positions in positions_by_batch.items():
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(data))
        batch_tokens = data[batch_start:batch_end]
        batch_dict = collate_batch(batch_tokens, device=model.cfg.device)

        # Baseline: SAE reconstruction
        reconstruction_hook = create_reconstruction_hook(sae)
        with torch.no_grad():
            baseline_logits = model.run_with_hooks(
                batch_dict["input_ids"],
                fwd_hooks=[(hook_name, reconstruction_hook)]
            )
            baseline_probs = torch.softmax(baseline_logits[:, :-1, :], dim=-1)

        # Ablated: Remove this feature
        ablation_hook = create_ablation_hook(sae, feature_id)
        with torch.no_grad():
            ablated_logits = model.run_with_hooks(
                batch_dict["input_ids"],
                fwd_hooks=[(hook_name, ablation_hook)]
            )
            ablated_probs = torch.softmax(ablated_logits[:, :-1, :], dim=-1)

        # Extract probability changes at the specific positions
        for batch_pos, seq_pos in batch_positions:
            baseline_prob_dist = baseline_probs[batch_pos, seq_pos]  # [vocab]
            ablated_prob_dist = ablated_probs[batch_pos, seq_pos]    # [vocab]

            # Compute RELATIVE probability change: (baseline - ablated) / baseline
            # Add epsilon to avoid division by zero
            relative_change = (baseline_prob_dist - ablated_prob_dist) / (baseline_prob_dist + 1e-10)
            # Positive = feature promoted this token (ablation reduces its probability)

            # Store top promoted token changes (by relative reduction)
            top_promoted = relative_change.topk(10)
            all_prob_changes.extend(top_promoted.values.cpu().numpy())

    # Compute statistics
    prob_changes_array = np.array(all_prob_changes)

    stats = {
        'feature_id': feature_id,
        'layer': layer,
        'num_positions': len(positions),
        'mean_relative_change': float(prob_changes_array.mean()),
        'median_relative_change': float(np.median(prob_changes_array)),
        'std_relative_change': float(prob_changes_array.std()),
        'max_relative_change': float(prob_changes_array.max()),
        'min_relative_change': float(prob_changes_array.min()),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Top-K feature ablation analysis")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--top_k", type=int, default=20, help="Top K features per position")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["layers"] = list(range(1, 12))
    config["test_passages"] = config.get("test_passages", config["max_passages"])

    # Setup output
    output_dir = Path(config["output_dir"]) / "top_k_intervention"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("="*80)
    logger.info(f"TOP-{args.top_k} FEATURE ABLATION ANALYSIS")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Top-K: {args.top_k}")
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
        layers=config["layers"],
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

    # Process each layer
    all_results = []

    for layer in config["layers"]:
        logger.info(f"\n{'='*80}")
        logger.info(f"LAYER {layer}")
        logger.info(f"{'='*80}")

        sae = saes[layer]

        # Pass 1: Identify top-K features at each position
        feature_positions = identify_top_k_features_per_position(
            model, sae, layer, config["hook"], data,
            config["batch_size"], args.top_k, logger
        )

        # Pass 2: Measure ablation effect for each feature
        logger.info(f"  Pass 2: Measuring ablation effects for {len(feature_positions)} features...")

        for feature_id, positions in tqdm(
            list(feature_positions.items())[:100],  # Limit to top 100 most frequent for speed
            desc=f"Layer {layer} - Measuring effects"
        ):
            stats = measure_feature_ablation_effect(
                model, sae, layer, feature_id, config["hook"],
                positions, data, config["batch_size"], logger
            )
            all_results.append(stats)

        logger.info(f"  Completed {len(all_results)} feature measurements for layer {layer}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = output_dir / "top_k_ablation_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to: {results_path}")

    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY (RELATIVE PROBABILITY CHANGES)")
    logger.info("="*80)
    logger.info(f"Total features analyzed: {len(results_df)}")
    logger.info(f"Overall mean relative change: {results_df['mean_relative_change'].mean():.4f} ({results_df['mean_relative_change'].mean()*100:.2f}%)")
    logger.info(f"Overall median relative change: {results_df['median_relative_change'].mean():.4f} ({results_df['median_relative_change'].mean()*100:.2f}%)")
    logger.info(f"\nInterpretation: A value of 0.50 means ablating the feature reduces")
    logger.info(f"promoted token probability by 50% (e.g., 10% -> 5%)")

    logger.info(f"\nPer-Layer Statistics:")
    for layer in sorted(results_df['layer'].unique()):
        layer_results = results_df[results_df['layer'] == layer]
        logger.info(
            f"  Layer {layer}: n={len(layer_results)}, "
            f"mean={layer_results['mean_relative_change'].mean():.4f} ({layer_results['mean_relative_change'].mean()*100:.1f}%), "
            f"median={layer_results['median_relative_change'].mean():.4f}, "
            f"max={layer_results['mean_relative_change'].max():.4f} ({layer_results['mean_relative_change'].max()*100:.1f}%)"
        )

    logger.info(f"\nTop 10 features by effect size:")
    top_features = results_df.nlargest(10, 'mean_relative_change')
    for _, row in top_features.iterrows():
        logger.info(
            f"  Layer {row['layer']}, Feature {row['feature_id']}: "
            f"mean_change={row['mean_relative_change']:.4f} ({row['mean_relative_change']*100:.1f}%), "
            f"positions={row['num_positions']}"
        )

    logger.info("\nDone!")


if __name__ == "__main__":
    main()

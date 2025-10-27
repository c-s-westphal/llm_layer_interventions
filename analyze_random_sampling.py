#!/usr/bin/env python3
"""
Random Feature Sampling Ablation Analysis

At each position in the corpus:
1. Randomly sample 60 features (different random 60 per position)
2. Ablate each of the 60 features
3. Measure both relative and absolute probability change
4. Track activation magnitude and co-activation count

Reports per layer:
- Mean across all 60 random samples at all positions
- Mean of top-K by metric value (relative change) for K=5,10,20
- Mean of top-K by activation value for K=5,10,20
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


def sample_random_features_per_position(
    model,
    sae,
    layer: int,
    hook: str,
    data: List[torch.Tensor],
    batch_size: int,
    num_features_to_sample: int,
    d_sae: int,
    logger: logging.Logger
) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    For each position, randomly sample num_features_to_sample features.

    Returns:
        Dictionary mapping feature_id -> list of (batch_idx, batch_pos, seq_pos) tuples
        where that feature was randomly sampled
    """
    logger.info(f"  Pass 1: Sampling {num_features_to_sample} random features at each position...")

    hook_name = f"blocks.{layer}.hook_{hook}"
    feature_positions = defaultdict(list)

    num_batches = (len(data) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"Layer {layer} - Sampling features"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(data))
        batch_tokens = data[batch_start:batch_end]
        batch_dict = collate_batch(batch_tokens, device=model.cfg.device)

        # Get attention mask to identify valid positions
        attention_mask = batch_dict["attention_mask"][:, :-1].bool()  # [batch, seq-1]

        batch_s, seq_s = attention_mask.shape
        for b in range(batch_s):
            for s in range(seq_s):
                if not attention_mask[b, s]:
                    continue

                # Randomly sample num_features_to_sample features
                sampled_features = np.random.choice(
                    d_sae,
                    size=num_features_to_sample,
                    replace=False
                )

                for feat_id in sampled_features:
                    feature_positions[int(feat_id)].append((batch_idx, b, s))

    logger.info(f"    Sampled across {len(feature_positions)} unique features")
    return dict(feature_positions)


def measure_ablation_effects(
    model,
    sae,
    layer: int,
    hook: str,
    feature_positions: Dict[int, List[Tuple[int, int, int]]],
    data: List[torch.Tensor],
    batch_size: int,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Ablate each feature at positions where it was sampled and measure effects.

    Returns:
        DataFrame with columns: feature_id, activation, co_activation_count,
                                relative_change, absolute_change
    """
    hook_name = f"blocks.{layer}.hook_{hook}"

    all_measurements = []

    logger.info(f"  Pass 2: Measuring ablation effects for {len(feature_positions)} features...")

    # Group by batch for efficient processing
    batch_to_features = defaultdict(set)
    for feature_id, positions in feature_positions.items():
        for batch_idx, _, _ in positions:
            batch_to_features[batch_idx].add(feature_id)

    for batch_idx in tqdm(sorted(batch_to_features.keys()), desc=f"Layer {layer} - Ablating"):
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

        # Ablate each feature that was sampled in this batch
        for feature_id in batch_to_features[batch_idx]:
            ablation_hook = create_ablation_hook(sae, feature_id)

            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    batch_dict["input_ids"],
                    fwd_hooks=[(hook_name, ablation_hook)]
                )
                ablated_probs = torch.softmax(ablated_logits[:, :-1, :], dim=-1)

            # Measure at positions where this feature was sampled
            for batch_idx_check, batch_pos, seq_pos in feature_positions[feature_id]:
                if batch_idx_check != batch_idx:
                    continue

                # Get activation and co-activation count
                feature_act = sae_acts[batch_pos, seq_pos, feature_id].item()
                co_activation_count = (sae_acts[batch_pos, seq_pos] > 0).sum().item()

                # Get probability distributions
                baseline_prob_dist = baseline_probs[batch_pos, seq_pos]
                ablated_prob_dist = ablated_probs[batch_pos, seq_pos]

                # Compute both relative and absolute change for top promoted token
                absolute_change = (baseline_prob_dist - ablated_prob_dist)
                relative_change = absolute_change / (baseline_prob_dist + 1e-10)

                top_abs = absolute_change.max().item()
                top_rel = relative_change.max().item()

                all_measurements.append({
                    'feature_id': feature_id,
                    'activation': feature_act,
                    'co_activation_count': co_activation_count,
                    'relative_change': top_rel,
                    'absolute_change': top_abs
                })

    return pd.DataFrame(all_measurements)


def main():
    parser = argparse.ArgumentParser(description="Random feature sampling ablation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num_sample", type=int, default=60,
                       help="Number of features to randomly sample per position")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup output
    output_dir = Path(config["output_dir"]) / "random_sampling"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("="*80)
    logger.info(f"RANDOM FEATURE SAMPLING ABLATION (Sample {args.num_sample} per position)")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Features per position: {args.num_sample}")
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

    # Process each layer
    all_results = []

    for layer in range(1, 12):
        logger.info(f"\n{'='*80}")
        logger.info(f"LAYER {layer}")
        logger.info(f"{'='*80}")

        sae = saes[layer]
        d_sae = sae.cfg.d_sae

        # Pass 1: Sample random features at each position
        feature_positions = sample_random_features_per_position(
            model, sae, layer, config["hook"], data,
            config["batch_size"], args.num_sample, d_sae, logger
        )

        # Pass 2: Measure ablation effects
        layer_results = measure_ablation_effects(
            model, sae, layer, config["hook"], feature_positions,
            data, config["batch_size"], logger
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

    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]

        logger.info(f"\n{'='*80}")
        logger.info(f"LAYER {layer}")
        logger.info(f"{'='*80}")

        logger.info(f"\nOverall Statistics (all {args.num_sample} random samples):")
        logger.info(f"  Total measurements: {len(layer_df):,}")
        logger.info(f"  Mean relative change: {layer_df['relative_change'].mean():.4f} ({layer_df['relative_change'].mean()*100:.1f}%)")
        logger.info(f"  Mean absolute change: {layer_df['absolute_change'].mean():.6f}")
        logger.info(f"  Median relative change: {layer_df['relative_change'].median():.4f}")
        logger.info(f"  Median absolute change: {layer_df['absolute_change'].median():.6f}")

        logger.info(f"\nTop-K by Metric Value (Relative Change):")
        for k in [5, 10, 20]:
            if len(layer_df) >= k:
                top_k = layer_df.nlargest(k, 'relative_change')
                logger.info(
                    f"  Top-{k}: mean_rel={top_k['relative_change'].mean():.4f} ({top_k['relative_change'].mean()*100:.1f}%), "
                    f"mean_abs={top_k['absolute_change'].mean():.6f}"
                )

        logger.info(f"\nTop-K by Activation Value:")
        for k in [5, 10, 20]:
            if len(layer_df) >= k:
                top_k = layer_df.nlargest(k, 'activation')
                logger.info(
                    f"  Top-{k}: mean_rel={top_k['relative_change'].mean():.4f} ({top_k['relative_change'].mean()*100:.1f}%), "
                    f"mean_abs={top_k['absolute_change'].mean():.6f}, "
                    f"mean_activation={top_k['activation'].mean():.2f}"
                )

    logger.info("\n" + "="*80)
    logger.info("CROSS-LAYER SUMMARY")
    logger.info("="*80)

    logger.info(f"\nMean Relative Change by Layer:")
    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]
        logger.info(
            f"  Layer {layer}: {layer_df['relative_change'].mean():.4f} ({layer_df['relative_change'].mean()*100:.1f}%)"
        )

    logger.info(f"\nMean Absolute Change by Layer:")
    for layer in range(1, 12):
        layer_df = results_df[results_df['layer'] == layer]
        logger.info(
            f"  Layer {layer}: {layer_df['absolute_change'].mean():.6f}"
        )

    logger.info("\nDone!")


if __name__ == "__main__":
    main()

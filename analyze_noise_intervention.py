"""Analyze SAE feature interventions using ablation.

This script ablates (zeros out) features in SAE latent space, measures the KLD,
and analyzes probability changes on important tokens.
"""

import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

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
    logger = logging.getLogger("ablation_intervention")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(output_dir / "ablation_intervention.log")
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


def find_top_tokens_by_probability(
    model,
    data: List[torch.Tensor],
    batch_size: int,
    top_k: int = 10,
    sample_size: int = 50,
    logger: logging.Logger = None
) -> List[int]:
    """Find top-K tokens by actual output probability.

    Args:
        model: HookedTransformer model
        data: List of token tensors
        batch_size: Batch size for processing
        top_k: Number of top tokens to return
        sample_size: Number of samples to average over
        logger: Logger instance

    Returns:
        List of token IDs
    """
    logger = logger or logging.getLogger("ablation_intervention")
    logger.info(f"Finding top-{top_k} tokens by actual probability...")

    sample_size = min(sample_size, len(data))
    all_clean_probs = []

    for sample_idx in range(0, sample_size, batch_size):
        batch_end = min(sample_idx + batch_size, sample_size)
        batch_tokens = data[sample_idx:batch_end]
        batch_dict = collate_batch(batch_tokens, device=model.cfg.device)

        with torch.no_grad():
            clean_logits = model(batch_dict["input_ids"])
            clean_probs = torch.softmax(clean_logits[:, :-1, :], dim=-1)

            # Average over valid positions
            mask = batch_dict["attention_mask"][:, 1:].bool()
            avg_probs = (clean_probs * mask.unsqueeze(-1)).sum(dim=(0, 1)) / mask.sum()
            all_clean_probs.append(avg_probs.cpu())

    # Average across all samples
    avg_clean_probs = torch.stack(all_clean_probs).mean(dim=0)
    top_tokens = torch.topk(avg_clean_probs, k=top_k).indices.tolist()

    # Log the tokens
    logger.info(f"Top-{top_k} tokens by probability:")
    for i, token_id in enumerate(top_tokens[:10]):
        token_str = model.tokenizer.decode([token_id])
        prob = avg_clean_probs[token_id].item()
        logger.info(f"  {i+1}. Token {token_id} ('{token_str}'): {prob:.6f}")

    return top_tokens


def get_mean_activation(
    model,
    sae,
    layer: int,
    feature_id: int,
    hook: str,
    data: List[torch.Tensor],
    batch_size: int,
    logger: logging.Logger = None
) -> Tuple[float, float]:
    """Find mean and std activation for a feature across calibration data.

    Args:
        model: HookedTransformer model
        sae: SAE instance
        layer: Layer index
        feature_id: Feature ID
        hook: Hook type
        data: List of token tensors
        batch_size: Batch size
        logger: Logger instance

    Returns:
        Tuple of (mean_activation, std_activation)
    """
    logger = logger or logging.getLogger("ablation_intervention")
    logger.info(f"Computing activation statistics for layer {layer}, feature {feature_id}...")

    all_activations = []
    hook_name = f"blocks.{layer}.hook_{hook}"

    num_batches = (len(data) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
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
            sae_acts = sae.encode(acts)
            feature_acts = sae_acts[:, :, feature_id]

            all_activations.append(feature_acts.flatten().cpu())

    # Concatenate and compute statistics
    all_acts = torch.cat(all_activations)
    mean_act = all_acts.mean().item()
    std_act = all_acts.std().item()

    logger.info(f"  Mean activation: {mean_act:.4f}, Std: {std_act:.4f}")
    return mean_act, std_act


def compute_kl_divergence(
    clean_probs: torch.Tensor,
    noisy_probs: torch.Tensor,
    mask: torch.Tensor = None
) -> float:
    """Compute KL divergence KL(P_clean || P_noisy).

    Args:
        clean_probs: Clean probabilities [batch, seq, vocab]
        noisy_probs: Noisy probabilities [batch, seq, vocab]
        mask: Attention mask [batch, seq]

    Returns:
        Average KL divergence
    """
    kl_per_position = (clean_probs * (
        torch.log(clean_probs + 1e-10) - torch.log(noisy_probs + 1e-10)
    )).sum(dim=-1)  # [batch, seq]

    if mask is not None:
        mask = mask.bool()
        kl_mean = (kl_per_position * mask).sum() / mask.sum()
    else:
        kl_mean = kl_per_position.mean()

    return kl_mean.item()


def create_ablation_hook(sae, feature_id: int):
    """Create a hook that ablates (zeros out) a specific feature.

    Args:
        sae: SAE instance
        feature_id: Feature ID to ablate

    Returns:
        Hook function
    """
    def hook_fn(activations, hook):
        # Encode to SAE latent space
        sae_acts = sae.encode(activations)  # [batch, seq, d_sae]

        # Ablate the feature (set to 0)
        sae_acts[:, :, feature_id] = 0.0

        # Decode back
        modified_acts = sae.decode(sae_acts)

        return modified_acts

    return hook_fn


def create_reconstruction_hook(sae):
    """Create a hook that just does SAE reconstruction (no noise).

    Args:
        sae: SAE instance

    Returns:
        Hook function
    """
    def hook_fn(activations, hook):
        # Encode to SAE latent space
        sae_acts = sae.encode(activations)  # [batch, seq, d_sae]

        # Decode back (no noise added)
        reconstructed_acts = sae.decode(sae_acts)

        return reconstructed_acts

    return hook_fn


def measure_kld_with_ablation(
    model,
    sae,
    layer: int,
    feature_id: int,
    hook: str,
    data: List[torch.Tensor],
    batch_size: int,
    logger: logging.Logger = None
) -> float:
    """Measure KLD when ablating a feature.

    Compares reconstructed baseline to ablated.

    Args:
        model: HookedTransformer model
        sae: SAE instance
        layer: Layer index
        feature_id: Feature ID
        hook: Hook type
        data: List of token tensors
        batch_size: Batch size
        logger: Logger instance

    Returns:
        KLD value
    """
    logger = logger or logging.getLogger("ablation_intervention")
    hook_name = f"blocks.{layer}.hook_{hook}"

    logger.info(f"  Measuring KLD from ablating feature {feature_id}...")

    # Measure KLD on first batch
    batch_tokens = data[:batch_size]
    batch_dict = collate_batch(batch_tokens, device=model.cfg.device)

    # Baseline: SAE reconstruction without ablation
    reconstruction_hook = create_reconstruction_hook(sae)

    with torch.no_grad():
        baseline_logits = model.run_with_hooks(
            batch_dict["input_ids"],
            fwd_hooks=[(hook_name, reconstruction_hook)]
        )
        baseline_probs = torch.softmax(baseline_logits[:, :-1, :], dim=-1)

    # Ablated: SAE reconstruction WITH feature ablated
    ablation_hook = create_ablation_hook(sae, feature_id)

    with torch.no_grad():
        ablated_logits = model.run_with_hooks(
            batch_dict["input_ids"],
            fwd_hooks=[(hook_name, ablation_hook)]
        )
        ablated_probs = torch.softmax(ablated_logits[:, :-1, :], dim=-1)

    # Compute KLD
    mask = batch_dict["attention_mask"][:, 1:]
    kld = compute_kl_divergence(baseline_probs, ablated_probs, mask)

    logger.info(f"  KLD from ablation: {kld:.4f}")
    return kld


def measure_probability_changes(
    model,
    sae,
    layer: int,
    feature_id: int,
    hook: str,
    data: List[torch.Tensor],
    top_tokens: List[int],
    batch_size: int,
    tokenizer,
    logger: logging.Logger = None
) -> Dict[str, float]:
    """Measure probability changes on top tokens when ablating a feature.

    Args:
        model: HookedTransformer model
        sae: SAE instance
        layer: Layer index
        feature_id: Feature ID
        hook: Hook type
        data: List of token tensors
        top_tokens: List of important token IDs
        batch_size: Batch size
        tokenizer: Tokenizer for logging
        logger: Logger instance

    Returns:
        Dictionary with statistics (mean, median, etc.)
    """
    logger = logger or logging.getLogger("ablation_intervention")
    logger.info(f"  Measuring probability changes on TARGET feature {feature_id} (ablation)...")

    # Log which tokens we're analyzing
    logger.info(f"  Analyzing probability changes for tokens:")
    for i, token_id in enumerate(top_tokens[:10]):
        token_str = tokenizer.decode([token_id])
        logger.info(f"    {i+1}. Token {token_id} ('{token_str}')")

    hook_name = f"blocks.{layer}.hook_{hook}"
    top_tokens_tensor = torch.tensor(top_tokens, device=model.cfg.device)

    all_relative_changes = []

    num_batches = (len(data) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(data))
        batch_tokens = data[batch_start:batch_end]
        batch_dict = collate_batch(batch_tokens, device=model.cfg.device)

        # Baseline: SAE reconstruction without ablation
        reconstruction_hook = create_reconstruction_hook(sae)

        with torch.no_grad():
            baseline_logits = model.run_with_hooks(
                batch_dict["input_ids"],
                fwd_hooks=[(hook_name, reconstruction_hook)]
            )
            baseline_probs = torch.softmax(baseline_logits[:, :-1, :], dim=-1)

        # Ablated: SAE reconstruction WITH feature ablated
        ablation_hook = create_ablation_hook(sae, feature_id)

        with torch.no_grad():
            ablated_logits = model.run_with_hooks(
                batch_dict["input_ids"],
                fwd_hooks=[(hook_name, ablation_hook)]
            )
            ablated_probs = torch.softmax(ablated_logits[:, :-1, :], dim=-1)

        # Extract probabilities for top tokens
        baseline_top_probs = baseline_probs[:, :, top_tokens_tensor].mean(dim=-1)  # [batch, seq]
        ablated_top_probs = ablated_probs[:, :, top_tokens_tensor].mean(dim=-1)  # [batch, seq]

        # Compute relative change: (baseline - ablated) / baseline
        mask = batch_dict["attention_mask"][:, 1:].bool()
        relative_change = (baseline_top_probs - ablated_top_probs) / (baseline_top_probs + 1e-10)

        # Collect valid positions
        valid_changes = relative_change[mask].cpu().numpy()
        all_relative_changes.append(valid_changes)

    # Concatenate all changes
    all_changes = np.concatenate(all_relative_changes)

    # Compute statistics
    stats = {
        "mean": float(np.mean(all_changes)),
        "median": float(np.median(all_changes)),
        "std": float(np.std(all_changes)),
        "min": float(np.min(all_changes)),
        "max": float(np.max(all_changes)),
        "q25": float(np.percentile(all_changes, 25)),
        "q75": float(np.percentile(all_changes, 75))
    }

    logger.info(f"  Top-10 token probability change statistics (target feature):")
    logger.info(f"    Mean: {stats['mean']:.4f}")
    logger.info(f"    Median: {stats['median']:.4f}")
    logger.info(f"    Std: {stats['std']:.4f}")
    logger.info(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    return stats


def analyze_per_feature_tokens(
    model,
    sae,
    layer: int,
    feature_id: int,
    hook: str,
    data: List[torch.Tensor],
    batch_size: int,
    tokenizer,
    top_k: int = 20,
    logger: logging.Logger = None
) -> Dict[str, List]:
    """Find top-K promoted and suppressed tokens for a specific feature.

    Args:
        model: HookedTransformer model
        sae: SAE instance
        layer: Layer index
        feature_id: Feature ID
        hook: Hook type
        data: List of token tensors
        batch_size: Batch size
        tokenizer: Tokenizer for decoding
        top_k: Number of top promoted/suppressed tokens to return
        logger: Logger instance

    Returns:
        Dictionary with promoted and suppressed token info
    """
    logger = logger or logging.getLogger("ablation_intervention")
    logger.info(f"  Finding top-{top_k} promoted/suppressed tokens for feature {feature_id}...")

    hook_name = f"blocks.{layer}.hook_{hook}"
    vocab_size = model.cfg.d_vocab

    # Accumulate probability changes per token across all positions
    token_changes_sum = torch.zeros(vocab_size, device=model.cfg.device)
    num_valid_positions = 0

    num_batches = (len(data) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(data))
        batch_tokens = data[batch_start:batch_end]
        batch_dict = collate_batch(batch_tokens, device=model.cfg.device)

        # Baseline: SAE reconstruction without ablation
        reconstruction_hook = create_reconstruction_hook(sae)

        with torch.no_grad():
            baseline_logits = model.run_with_hooks(
                batch_dict["input_ids"],
                fwd_hooks=[(hook_name, reconstruction_hook)]
            )
            baseline_probs = torch.softmax(baseline_logits[:, :-1, :], dim=-1)

        # Ablated: SAE reconstruction WITH feature ablated
        ablation_hook = create_ablation_hook(sae, feature_id)

        with torch.no_grad():
            ablated_logits = model.run_with_hooks(
                batch_dict["input_ids"],
                fwd_hooks=[(hook_name, ablation_hook)]
            )
            ablated_probs = torch.softmax(ablated_logits[:, :-1, :], dim=-1)

        # Compute relative change: (baseline - ablated) / baseline for ALL tokens
        # Shape: [batch, seq, vocab]
        mask = batch_dict["attention_mask"][:, 1:].unsqueeze(-1).bool()  # [batch, seq, 1]
        relative_change = (baseline_probs - ablated_probs) / (baseline_probs + 1e-10)

        # Mask out padding positions
        relative_change = relative_change * mask

        # Sum changes per token across batch and sequence
        token_changes_sum += relative_change.sum(dim=(0, 1))
        num_valid_positions += mask.sum().item()

    # Average change per token
    avg_token_changes = token_changes_sum / (num_valid_positions + 1e-10)

    # Find top-K promoted tokens (most positive change)
    top_promoted_values, top_promoted_indices = torch.topk(avg_token_changes, k=min(top_k, vocab_size))

    # Find top-K suppressed tokens (most negative change)
    top_suppressed_values, top_suppressed_indices = torch.topk(-avg_token_changes, k=min(top_k, vocab_size))
    top_suppressed_values = -top_suppressed_values  # Convert back to negative

    # Convert to lists and decode tokens
    promoted_tokens = []
    for idx, value in zip(top_promoted_indices.cpu().numpy(), top_promoted_values.cpu().numpy()):
        token_str = tokenizer.decode([int(idx)])
        promoted_tokens.append({
            "token_id": int(idx),
            "token_str": token_str,
            "avg_change": float(value)
        })

    suppressed_tokens = []
    for idx, value in zip(top_suppressed_indices.cpu().numpy(), top_suppressed_values.cpu().numpy()):
        token_str = tokenizer.decode([int(idx)])
        suppressed_tokens.append({
            "token_id": int(idx),
            "token_str": token_str,
            "avg_change": float(value)
        })

    # Log results
    logger.info(f"  Top-{min(5, top_k)} PROMOTED tokens (feature {feature_id}):")
    for i, token_info in enumerate(promoted_tokens[:5]):
        logger.info(f"    {i+1}. '{token_info['token_str']}' (ID {token_info['token_id']}): {token_info['avg_change']:.4f}")

    logger.info(f"  Top-{min(5, top_k)} SUPPRESSED tokens (feature {feature_id}):")
    for i, token_info in enumerate(suppressed_tokens[:5]):
        logger.info(f"    {i+1}. '{token_info['token_str']}' (ID {token_info['token_id']}): {token_info['avg_change']:.4f}")

    return {
        "promoted_tokens": promoted_tokens,
        "suppressed_tokens": suppressed_tokens
    }


def measure_probability_changes_random_control(
    model,
    sae,
    layer: int,
    feature_id: int,
    hook: str,
    data: List[torch.Tensor],
    top_tokens: List[int],
    batch_size: int,
    tokenizer,
    logger: logging.Logger = None
) -> Dict[str, float]:
    """Measure probability changes when ablating a RANDOM feature (control).

    Args:
        model: HookedTransformer model
        sae: SAE instance
        layer: Layer index
        feature_id: Original feature ID (will select a different random one)
        hook: Hook type
        data: List of token tensors
        top_tokens: Same top-10 tokens as target measurement
        batch_size: Batch size
        tokenizer: Tokenizer for logging
        logger: Logger instance

    Returns:
        Dictionary with statistics (mean, median, etc.)
    """
    logger = logger or logging.getLogger("ablation_intervention")

    # Select a random feature (different from the target)
    d_sae = sae.cfg.d_sae
    random_feature_id = np.random.randint(0, d_sae)
    while random_feature_id == feature_id:
        random_feature_id = np.random.randint(0, d_sae)

    logger.info(f"  Measuring probability changes on RANDOM feature {random_feature_id} vs TARGET feature {feature_id} (control)...")

    hook_name = f"blocks.{layer}.hook_{hook}"
    top_tokens_tensor = torch.tensor(top_tokens, device=model.cfg.device)

    all_relative_changes = []

    num_batches = (len(data) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(data))
        batch_tokens = data[batch_start:batch_end]
        batch_dict = collate_batch(batch_tokens, device=model.cfg.device)

        # Baseline: SAE reconstruction without ablation
        reconstruction_hook = create_reconstruction_hook(sae)

        with torch.no_grad():
            baseline_logits = model.run_with_hooks(
                batch_dict["input_ids"],
                fwd_hooks=[(hook_name, reconstruction_hook)]
            )
            baseline_probs = torch.softmax(baseline_logits[:, :-1, :], dim=-1)

        # Ablated: SAE reconstruction WITH RANDOM feature ablated
        ablation_hook = create_ablation_hook(sae, random_feature_id)

        with torch.no_grad():
            ablated_logits = model.run_with_hooks(
                batch_dict["input_ids"],
                fwd_hooks=[(hook_name, ablation_hook)]
            )
            ablated_probs = torch.softmax(ablated_logits[:, :-1, :], dim=-1)

        # Extract probabilities for SAME top tokens
        baseline_top_probs = baseline_probs[:, :, top_tokens_tensor].mean(dim=-1)  # [batch, seq]
        ablated_top_probs = ablated_probs[:, :, top_tokens_tensor].mean(dim=-1)  # [batch, seq]

        # Compute relative change: (baseline - ablated) / baseline
        mask = batch_dict["attention_mask"][:, 1:].bool()
        relative_change = (baseline_top_probs - ablated_top_probs) / (baseline_top_probs + 1e-10)

        # Collect valid positions
        valid_changes = relative_change[mask].cpu().numpy()
        all_relative_changes.append(valid_changes)

    # Concatenate all changes
    all_changes = np.concatenate(all_relative_changes)

    # Compute statistics
    stats = {
        "mean": float(np.mean(all_changes)),
        "median": float(np.median(all_changes)),
        "std": float(np.std(all_changes)),
        "min": float(np.min(all_changes)),
        "max": float(np.max(all_changes)),
        "q25": float(np.percentile(all_changes, 25)),
        "q75": float(np.percentile(all_changes, 75))
    }

    logger.info(f"  Top-10 token probability change statistics (random feature {random_feature_id} control):")
    logger.info(f"    Mean: {stats['mean']:.4f}")
    logger.info(f"    Median: {stats['median']:.4f}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Ablation-based feature intervention analysis")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--top_k_activation", type=int, default=5, help="Top K features by activation")
    parser.add_argument("--top_k_interpretability", type=int, default=5, help="Top K features by interpretability")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override config for this analysis
    config["layers"] = list(range(1, 12))  # Layers 1-11 (skip layer 0 embeddings)
    config["test_passages"] = config.get("test_passages", config["max_passages"])

    # Setup output directory
    output_dir = Path(config["output_dir"]) / "ablation_intervention"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("="*80)
    logger.info("ABLATION-BASED FEATURE INTERVENTION ANALYSIS")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
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

    # Load calibration data
    logger.info("Loading calibration corpus...")
    calibration_loader = CorpusLoader(
        corpus_name=config["corpus_name"],
        max_passages=config["calibration_passages"],
        max_len=config["max_len"],
        tokenizer=model.tokenizer,
        logger=logger
    )
    calibration_data, _ = calibration_loader.load_and_tokenize()
    logger.info(f"Loaded {len(calibration_data)} calibration passages")

    # Load test data
    logger.info("Loading test corpus...")
    test_loader = CorpusLoader(
        corpus_name=config["corpus_name"],
        max_passages=config.get("test_passages", config["max_passages"]),
        max_len=config["max_len"],
        tokenizer=model.tokenizer,
        logger=logger
    )
    test_data, _ = test_loader.load_and_tokenize()
    logger.info(f"Loaded {len(test_data)} test passages")

    # Find top-10 tokens by probability
    top_tokens = find_top_tokens_by_probability(
        model,
        calibration_data,
        batch_size=config["batch_size"],
        top_k=10,
        sample_size=50,
        logger=logger
    )

    # Load interpretability scores from CSV
    logger.info("Loading interpretability scores from neuronpedia_features.csv...")
    features_df = pd.read_csv("data/neuronpedia_features.csv")

    # Get top features by activation (we'll compute this from calibration data)
    logger.info("Finding top features by activation...")
    feature_activations = {}

    for layer in tqdm(config["layers"], desc="Computing activations"):
        sae = saes[layer]
        hook_name = f"blocks.{layer}.hook_{config['hook']}"

        max_acts = []

        # Process calibration data
        num_batches = (len(calibration_data) + config["batch_size"] - 1) // config["batch_size"]

        for batch_idx in range(num_batches):
            batch_start = batch_idx * config["batch_size"]
            batch_end = min(batch_start + config["batch_size"], len(calibration_data))
            batch_tokens = calibration_data[batch_start:batch_end]
            batch_dict = collate_batch(batch_tokens, device=model.cfg.device)

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    batch_dict["input_ids"],
                    names_filter=[hook_name]
                )
                acts = cache[hook_name]
                sae_acts = sae.encode(acts)

                # Get max activation per feature across this batch
                batch_max = sae_acts.reshape(-1, sae_acts.shape[-1]).max(dim=0).values
                max_acts.append(batch_max.cpu())

        # Overall max per feature
        overall_max = torch.stack(max_acts).max(dim=0).values

        for feature_id in range(overall_max.shape[0]):
            feature_activations[(layer, feature_id)] = overall_max[feature_id].item()

    # Select top K features by activation PER LAYER
    top_by_activation = []
    logger.info(f"\nTop {args.top_k_activation} features by activation per layer:")
    for layer in config["layers"]:
        # Get features for this layer
        layer_features = {(l, f): act for (l, f), act in feature_activations.items() if l == layer}
        # Sort and take top K
        top_k = sorted(layer_features.items(), key=lambda x: x[1], reverse=True)[:args.top_k_activation]
        top_by_activation.extend(top_k)

        logger.info(f"  Layer {layer}:")
        for (l, feature_id), activation in top_k:
            logger.info(f"    Feature {feature_id}: {activation:.4f}")

    # Select top K features by interpretability PER LAYER
    top_by_interpretability_list = []
    logger.info(f"\nTop {args.top_k_interpretability} features by interpretability per layer:")
    for layer in config["layers"]:
        # Get features for this layer from CSV
        layer_features = features_df[features_df["layer"] == layer]
        # Sort by confidence and take top K
        top_k = layer_features.nlargest(min(args.top_k_interpretability, len(layer_features)), "label_confidence")
        top_by_interpretability_list.append(top_k)

        logger.info(f"  Layer {layer}:")
        for _, row in top_k.iterrows():
            logger.info(
                f"    Feature {row['feature_id']}: "
                f"conf={row['label_confidence']:.2f}, label='{row['label']}'"
            )

    # Concatenate all layers
    if top_by_interpretability_list:
        top_by_interpretability = pd.concat(top_by_interpretability_list, ignore_index=True)
    else:
        top_by_interpretability = pd.DataFrame()

    # Process features
    results = []
    per_feature_token_results = []

    logger.info("\n" + "="*80)
    logger.info("PROCESSING FEATURES BY ACTIVATION")
    logger.info("="*80)

    for (layer, feature_id), activation in top_by_activation:
        logger.info(f"\n--- Layer {layer}, Feature {feature_id} (max activation={activation:.4f}) ---")

        # Measure KLD from ablation
        ablation_kld = measure_kld_with_ablation(
            model, saes[layer], layer, feature_id, config["hook"],
            calibration_data, config["batch_size"], logger=logger
        )

        # Measure probability changes on TARGET feature
        stats = measure_probability_changes(
            model, saes[layer], layer, feature_id, config["hook"],
            test_data, top_tokens,
            config["batch_size"], model.tokenizer, logger
        )

        # Analyze per-feature tokens
        per_feature_analysis = analyze_per_feature_tokens(
            model, saes[layer], layer, feature_id, config["hook"],
            test_data, config["batch_size"], model.tokenizer,
            top_k=20, logger=logger
        )

        # Store per-feature token results
        for token_info in per_feature_analysis["promoted_tokens"]:
            per_feature_token_results.append({
                "layer": layer,
                "feature_id": feature_id,
                "selection_method": "activation",
                "direction": "promoted",
                "token_id": token_info["token_id"],
                "token_str": token_info["token_str"],
                "avg_change": token_info["avg_change"]
            })

        for token_info in per_feature_analysis["suppressed_tokens"]:
            per_feature_token_results.append({
                "layer": layer,
                "feature_id": feature_id,
                "selection_method": "activation",
                "direction": "suppressed",
                "token_id": token_info["token_id"],
                "token_str": token_info["token_str"],
                "avg_change": token_info["avg_change"]
            })

        # Measure probability changes on RANDOM feature (control)
        control_stats = measure_probability_changes_random_control(
            model, saes[layer], layer, feature_id, config["hook"],
            test_data, top_tokens,
            config["batch_size"], model.tokenizer, logger
        )

        # Prefix control stats
        control_stats_prefixed = {f"control_{k}": v for k, v in control_stats.items()}

        results.append({
            "layer": layer,
            "feature_id": feature_id,
            "selection_method": "activation",
            "max_activation": activation,
            "ablation_kld": ablation_kld,
            **stats,
            **control_stats_prefixed
        })

    logger.info("\n" + "="*80)
    logger.info("PROCESSING FEATURES BY INTERPRETABILITY")
    logger.info("="*80)

    for _, row in top_by_interpretability.iterrows():
        layer = int(row["layer"])
        feature_id = int(row["feature_id"])
        label = row["label"]
        conf = row["label_confidence"]

        logger.info(f"\n--- Layer {layer}, Feature {feature_id} ('{label}', conf={conf:.2f}) ---")

        # Measure KLD from ablation
        ablation_kld = measure_kld_with_ablation(
            model, saes[layer], layer, feature_id, config["hook"],
            calibration_data, config["batch_size"], logger=logger
        )

        # Measure probability changes on TARGET feature
        stats = measure_probability_changes(
            model, saes[layer], layer, feature_id, config["hook"],
            test_data, top_tokens,
            config["batch_size"], model.tokenizer, logger
        )

        # Analyze per-feature tokens
        per_feature_analysis = analyze_per_feature_tokens(
            model, saes[layer], layer, feature_id, config["hook"],
            test_data, config["batch_size"], model.tokenizer,
            top_k=20, logger=logger
        )

        # Store per-feature token results with label
        for token_info in per_feature_analysis["promoted_tokens"]:
            per_feature_token_results.append({
                "layer": layer,
                "feature_id": feature_id,
                "selection_method": "interpretability",
                "label": label,
                "label_confidence": conf,
                "direction": "promoted",
                "token_id": token_info["token_id"],
                "token_str": token_info["token_str"],
                "avg_change": token_info["avg_change"]
            })

        for token_info in per_feature_analysis["suppressed_tokens"]:
            per_feature_token_results.append({
                "layer": layer,
                "feature_id": feature_id,
                "selection_method": "interpretability",
                "label": label,
                "label_confidence": conf,
                "direction": "suppressed",
                "token_id": token_info["token_id"],
                "token_str": token_info["token_str"],
                "avg_change": token_info["avg_change"]
            })

        # Measure probability changes on RANDOM feature (control)
        control_stats = measure_probability_changes_random_control(
            model, saes[layer], layer, feature_id, config["hook"],
            test_data, top_tokens,
            config["batch_size"], model.tokenizer, logger
        )

        # Prefix control stats
        control_stats_prefixed = {f"control_{k}": v for k, v in control_stats.items()}

        results.append({
            "layer": layer,
            "feature_id": feature_id,
            "selection_method": "interpretability",
            "label": label,
            "label_confidence": conf,
            "ablation_kld": ablation_kld,
            **stats,
            **control_stats_prefixed
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_path = output_dir / "ablation_intervention_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to: {results_path}")

    # Save per-feature token results
    per_feature_tokens_df = pd.DataFrame(per_feature_token_results)
    per_feature_tokens_path = output_dir / "per_feature_tokens.csv"
    per_feature_tokens_df.to_csv(per_feature_tokens_path, index=False)
    logger.info(f"Per-feature token analysis saved to: {per_feature_tokens_path}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)

    logger.info("\nBy Selection Method:")
    logger.info(f"  Activation (top {args.top_k_activation} per layer):")
    activation_results = results_df[results_df["selection_method"] == "activation"]
    logger.info(f"    Mean of means: {activation_results['mean'].mean():.4f}")
    logger.info(f"    Mean of medians: {activation_results['median'].mean():.4f}")
    logger.info(f"    Mean ablation KLD: {activation_results['ablation_kld'].mean():.4f}")

    logger.info(f"\n  Interpretability (top {args.top_k_interpretability} per layer):")
    interp_results = results_df[results_df["selection_method"] == "interpretability"]
    logger.info(f"    Mean of means: {interp_results['mean'].mean():.4f}")
    logger.info(f"    Mean of medians: {interp_results['median'].mean():.4f}")
    logger.info(f"    Mean ablation KLD: {interp_results['ablation_kld'].mean():.4f}")

    logger.info("\nBy Layer and Feature Type:")
    for layer in sorted(results_df["layer"].unique()):
        logger.info(f"\n  Layer {layer}:")

        # Activation-based features
        layer_activation = results_df[
            (results_df["layer"] == layer) &
            (results_df["selection_method"] == "activation")
        ]
        if len(layer_activation) > 0:
            logger.info(
                f"    Activation (n={len(layer_activation)}): "
                f"mean_mean={layer_activation['mean'].mean():.4f}, "
                f"mean_median={layer_activation['median'].mean():.4f}, "
                f"control_mean_mean={layer_activation['control_mean'].mean():.4f}, "
                f"control_mean_median={layer_activation['control_median'].mean():.4f}"
            )
        else:
            logger.info(f"    Activation (n=0): none")

        # Interpretability-based features
        layer_interp = results_df[
            (results_df["layer"] == layer) &
            (results_df["selection_method"] == "interpretability")
        ]
        if len(layer_interp) > 0:
            logger.info(
                f"    Interpretability (n={len(layer_interp)}): "
                f"mean_mean={layer_interp['mean'].mean():.4f}, "
                f"mean_median={layer_interp['median'].mean():.4f}, "
                f"control_mean_mean={layer_interp['control_mean'].mean():.4f}, "
                f"control_mean_median={layer_interp['control_median'].mean():.4f}"
            )
        else:
            logger.info(f"    Interpretability (n=0): none")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()

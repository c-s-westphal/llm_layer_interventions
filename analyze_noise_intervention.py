"""Analyze SAE feature interventions using noise-based KLD targeting.

Instead of ablating features entirely, this script adds calibrated noise to features
until reaching a target KLD threshold, then measures probability changes on important tokens.
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
    logger = logging.getLogger("noise_intervention")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(output_dir / "noise_intervention.log")
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
    logger = logger or logging.getLogger("noise_intervention")
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


def get_max_activation(
    model,
    sae,
    layer: int,
    feature_id: int,
    hook: str,
    data: List[torch.Tensor],
    batch_size: int,
    logger: logging.Logger = None
) -> float:
    """Find maximum activation for a feature across calibration data.

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
        Maximum activation value
    """
    logger = logger or logging.getLogger("noise_intervention")
    logger.info(f"Finding max activation for layer {layer}, feature {feature_id}...")

    max_act = 0.0
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

            batch_max = feature_acts.max().item()
            max_act = max(max_act, batch_max)

    logger.info(f"  Max activation: {max_act:.4f}")
    return max_act


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


def create_noise_hook(sae, feature_id: int, noise_level: float, max_activation: float):
    """Create a hook that adds uniform noise to a feature.

    Args:
        sae: SAE instance
        feature_id: Feature ID to add noise to
        noise_level: Noise level (0 to 1, scaled by max_activation)
        max_activation: Maximum activation value for scaling

    Returns:
        Hook function
    """
    def hook_fn(activations, hook):
        # Encode to SAE latent space
        sae_acts = sae.encode(activations)  # [batch, seq, d_sae]

        # Add uniform noise to feature
        noise = torch.rand_like(sae_acts[:, :, feature_id]) * noise_level * max_activation
        sae_acts[:, :, feature_id] = sae_acts[:, :, feature_id] + noise

        # Decode back
        modified_acts = sae.decode(sae_acts)

        return modified_acts

    return hook_fn


def calibrate_noise_for_kld(
    model,
    sae,
    layer: int,
    feature_id: int,
    hook: str,
    data: List[torch.Tensor],
    max_activation: float,
    target_kld: float,
    batch_size: int,
    max_iterations: int = 20,
    tolerance: float = 0.01,
    logger: logging.Logger = None
) -> Tuple[float, float]:
    """Find noise level that achieves target KLD.

    Uses binary search to find the noise level that produces the target KLD.

    Args:
        model: HookedTransformer model
        sae: SAE instance
        layer: Layer index
        feature_id: Feature ID
        hook: Hook type
        data: List of token tensors
        max_activation: Maximum activation for scaling
        target_kld: Target KLD value
        batch_size: Batch size
        max_iterations: Maximum search iterations
        tolerance: KLD tolerance
        logger: Logger instance

    Returns:
        Tuple of (noise_level, achieved_kld)
    """
    logger = logger or logging.getLogger("noise_intervention")
    hook_name = f"blocks.{layer}.hook_{hook}"

    # Binary search bounds
    noise_low = 0.0
    noise_high = 1.0

    logger.info(f"Calibrating noise for layer {layer}, feature {feature_id} (target KLD={target_kld})...")

    for iteration in range(max_iterations):
        noise_level = (noise_low + noise_high) / 2.0

        # Measure KLD at this noise level (use first batch only for speed)
        batch_tokens = data[:batch_size]
        batch_dict = collate_batch(batch_tokens, device=model.cfg.device)

        # Clean run
        with torch.no_grad():
            clean_logits = model(batch_dict["input_ids"])
            clean_probs = torch.softmax(clean_logits[:, :-1, :], dim=-1)

        # Noisy run
        noise_hook = create_noise_hook(sae, feature_id, noise_level, max_activation)

        with torch.no_grad():
            noisy_logits = model.run_with_hooks(
                batch_dict["input_ids"],
                fwd_hooks=[(hook_name, noise_hook)]
            )
            noisy_probs = torch.softmax(noisy_logits[:, :-1, :], dim=-1)

        # Compute KLD
        mask = batch_dict["attention_mask"][:, 1:]
        kld = compute_kl_divergence(clean_probs, noisy_probs, mask)

        logger.info(f"  Iteration {iteration+1}: noise_level={noise_level:.4f}, KLD={kld:.4f}")

        # Check convergence
        if abs(kld - target_kld) < tolerance:
            logger.info(f"  Converged! Final noise_level={noise_level:.4f}, KLD={kld:.4f}")
            return noise_level, kld

        # Update search bounds
        if kld < target_kld:
            noise_low = noise_level
        else:
            noise_high = noise_level

    # Max iterations reached
    logger.warning(f"  Max iterations reached. Best noise_level={noise_level:.4f}, KLD={kld:.4f}")
    return noise_level, kld


def measure_probability_changes(
    model,
    sae,
    layer: int,
    feature_id: int,
    hook: str,
    data: List[torch.Tensor],
    top_tokens: List[int],
    noise_level: float,
    max_activation: float,
    batch_size: int,
    logger: logging.Logger = None
) -> Dict[str, float]:
    """Measure probability changes on top tokens at given noise level.

    Args:
        model: HookedTransformer model
        sae: SAE instance
        layer: Layer index
        feature_id: Feature ID
        hook: Hook type
        data: List of token tensors
        top_tokens: List of important token IDs
        noise_level: Calibrated noise level
        max_activation: Maximum activation
        batch_size: Batch size
        logger: Logger instance

    Returns:
        Dictionary with statistics (mean, median, etc.)
    """
    logger = logger or logging.getLogger("noise_intervention")
    hook_name = f"blocks.{layer}.hook_{hook}"
    top_tokens_tensor = torch.tensor(top_tokens, device=model.cfg.device)

    all_relative_changes = []

    num_batches = (len(data) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(data))
        batch_tokens = data[batch_start:batch_end]
        batch_dict = collate_batch(batch_tokens, device=model.cfg.device)

        # Clean run
        with torch.no_grad():
            clean_logits = model(batch_dict["input_ids"])
            clean_probs = torch.softmax(clean_logits[:, :-1, :], dim=-1)

        # Noisy run
        noise_hook = create_noise_hook(sae, feature_id, noise_level, max_activation)

        with torch.no_grad():
            noisy_logits = model.run_with_hooks(
                batch_dict["input_ids"],
                fwd_hooks=[(hook_name, noise_hook)]
            )
            noisy_probs = torch.softmax(noisy_logits[:, :-1, :], dim=-1)

        # Extract probabilities for top tokens
        clean_top_probs = clean_probs[:, :, top_tokens_tensor].mean(dim=-1)  # [batch, seq]
        noisy_top_probs = noisy_probs[:, :, top_tokens_tensor].mean(dim=-1)  # [batch, seq]

        # Compute relative change: (clean - noisy) / clean
        mask = batch_dict["attention_mask"][:, 1:].bool()
        relative_change = (clean_top_probs - noisy_top_probs) / (clean_top_probs + 1e-10)

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

    logger.info(f"  Probability change statistics:")
    logger.info(f"    Mean: {stats['mean']:.4f}")
    logger.info(f"    Median: {stats['median']:.4f}")
    logger.info(f"    Std: {stats['std']:.4f}")
    logger.info(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Noise-based feature intervention with KLD targeting")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--target_kld", type=float, default=0.1, help="Target KLD threshold")
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
    output_dir = Path(config["output_dir"]) / "noise_intervention"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("="*80)
    logger.info("NOISE-BASED INTERVENTION WITH KLD TARGETING")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Target KLD: {args.target_kld}")
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

    # Select top K features by activation
    top_by_activation = sorted(
        feature_activations.items(),
        key=lambda x: x[1],
        reverse=True
    )[:args.top_k_activation]

    logger.info(f"\nTop {args.top_k_activation} features by activation:")
    for (layer, feature_id), activation in top_by_activation:
        logger.info(f"  Layer {layer}, Feature {feature_id}: {activation:.4f}")

    # Select top K features by interpretability (label_confidence)
    top_by_interpretability = features_df.nlargest(args.top_k_interpretability, "label_confidence")

    logger.info(f"\nTop {args.top_k_interpretability} features by interpretability:")
    for _, row in top_by_interpretability.iterrows():
        logger.info(
            f"  Layer {row['layer']}, Feature {row['feature_id']}: "
            f"conf={row['label_confidence']:.2f}, label='{row['label']}'"
        )

    # Process features
    results = []

    logger.info("\n" + "="*80)
    logger.info("PROCESSING FEATURES BY ACTIVATION")
    logger.info("="*80)

    for (layer, feature_id), activation in top_by_activation:
        logger.info(f"\n--- Layer {layer}, Feature {feature_id} (activation={activation:.4f}) ---")

        # Get max activation
        max_act = get_max_activation(
            model, saes[layer], layer, feature_id, config["hook"],
            calibration_data, config["batch_size"], logger
        )

        # Calibrate noise
        noise_level, achieved_kld = calibrate_noise_for_kld(
            model, saes[layer], layer, feature_id, config["hook"],
            calibration_data, max_act, args.target_kld,
            config["batch_size"], logger=logger
        )

        # Measure probability changes
        stats = measure_probability_changes(
            model, saes[layer], layer, feature_id, config["hook"],
            test_data, top_tokens, noise_level, max_act,
            config["batch_size"], logger
        )

        results.append({
            "layer": layer,
            "feature_id": feature_id,
            "selection_method": "activation",
            "max_activation": max_act,
            "noise_level": noise_level,
            "achieved_kld": achieved_kld,
            **stats
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

        # Get max activation
        max_act = get_max_activation(
            model, saes[layer], layer, feature_id, config["hook"],
            calibration_data, config["batch_size"], logger
        )

        # Calibrate noise
        noise_level, achieved_kld = calibrate_noise_for_kld(
            model, saes[layer], layer, feature_id, config["hook"],
            calibration_data, max_act, args.target_kld,
            config["batch_size"], logger=logger
        )

        # Measure probability changes
        stats = measure_probability_changes(
            model, saes[layer], layer, feature_id, config["hook"],
            test_data, top_tokens, noise_level, max_act,
            config["batch_size"], logger
        )

        results.append({
            "layer": layer,
            "feature_id": feature_id,
            "selection_method": "interpretability",
            "label": label,
            "label_confidence": conf,
            "max_activation": max_act,
            "noise_level": noise_level,
            "achieved_kld": achieved_kld,
            **stats
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_path = output_dir / "noise_intervention_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to: {results_path}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)

    logger.info("\nBy Activation:")
    activation_results = results_df[results_df["selection_method"] == "activation"]
    logger.info(f"  Mean median probability change: {activation_results['median'].mean():.4f}")
    logger.info(f"  Mean noise level: {activation_results['noise_level'].mean():.4f}")

    logger.info("\nBy Interpretability:")
    interp_results = results_df[results_df["selection_method"] == "interpretability"]
    logger.info(f"  Mean median probability change: {interp_results['median'].mean():.4f}")
    logger.info(f"  Mean noise level: {interp_results['noise_level'].mean():.4f}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()

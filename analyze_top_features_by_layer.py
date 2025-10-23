"""Analyze the most active features per layer and measure their ablation effects.

This script:
1. Finds the top 5 most active features per layer (on calibration corpus)
2. Ablates each feature (alpha=0) and measures d_loss on test corpus
3. Plots layer vs mean d_loss with 95% CI error bars
"""

import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.data import load_corpus
from src.intervene import FeatureIntervention
from src.model import ModelLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("top_features_analysis")

# Load config
with open("configs/default.yaml", "r") as f:
    config = yaml.safe_load(f)

# Override config for this analysis
config["layers"] = list(range(12))  # All layers
config["num_calibration_passages"] = 200  # Enough to get good statistics
config["num_test_passages"] = 500  # Test on reasonable corpus size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ============================================================================
# Step 1: Load model and SAEs for all layers
# ============================================================================
logger.info("Step 1: Loading model and SAEs for all 12 layers...")

model_loader = ModelLoader(
    model_name=config["model_name"],
    sae_release=config["sae_release"],
    hook=config["hook"],
    layers=config["layers"],
    device=device,
    logger=logger
)

model = model_loader.load_model()
saes = model_loader.load_saes()

logger.info(f"Loaded model and {len(saes)} SAEs")

# ============================================================================
# Step 2: Load calibration corpus and find top 5 features per layer
# ============================================================================
logger.info("Step 2: Finding top 5 most active features per layer...")

calibration_data = load_corpus(
    dataset_name=config["corpus"]["name"],
    split=config["corpus"]["split"],
    num_passages=config["num_calibration_passages"],
    min_length=config["corpus"]["min_length"],
    max_length=config["corpus"]["max_length"],
    model=model,
    logger=logger
)

top_features_per_layer = {}  # {layer: [(feat_id, mean_activation), ...]}

batch_size = 16

for layer in tqdm(config["layers"], desc="Finding top features"):
    # Collect activations for this layer
    feature_activations = {}  # {feature_id: [activations]}

    num_batches = (len(calibration_data) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(calibration_data))
        batch_tokens = calibration_data[batch_start:batch_end]

        # Pad and collate
        from src.data import collate_batch
        batch_dict = collate_batch(batch_tokens, device=device)

        # Get activations at this layer
        with torch.no_grad():
            hook_name = f"blocks.{layer}.hook_{config['hook']}"
            _, cache = model.run_with_cache(
                batch_dict["input_ids"],
                names_filter=[hook_name]
            )

            acts = cache[hook_name]  # [batch, seq, d_model]

            # Encode with SAE
            sae = saes[layer]
            sae_acts = sae.encode(acts)  # [batch, seq, d_sae]

            # Get mean activation per feature across this batch
            # Average over batch and sequence dimensions
            mean_acts_per_feature = sae_acts.mean(dim=(0, 1)).cpu().numpy()  # [d_sae]

            # Store for this batch
            for feat_id in range(len(mean_acts_per_feature)):
                if feat_id not in feature_activations:
                    feature_activations[feat_id] = []
                feature_activations[feat_id].append(mean_acts_per_feature[feat_id])

    # Compute overall mean activation per feature
    feature_mean_activations = {
        feat_id: np.mean(acts_list)
        for feat_id, acts_list in feature_activations.items()
    }

    # Get top 5
    sorted_features = sorted(
        feature_mean_activations.items(),
        key=lambda x: x[1],
        reverse=True
    )
    top_5 = sorted_features[:5]
    top_features_per_layer[layer] = top_5

    logger.info(f"Layer {layer} top 5 features:")
    for rank, (feat_id, mean_act) in enumerate(top_5, 1):
        logger.info(f"  #{rank}: Feature {feat_id} (mean activation: {mean_act:.6f})")

# ============================================================================
# Step 3: Load test corpus
# ============================================================================
logger.info("Step 3: Loading test corpus...")

test_data = load_corpus(
    dataset_name=config["corpus"]["name"],
    split="test",
    num_passages=config["num_test_passages"],
    min_length=config["corpus"]["min_length"],
    max_length=config["corpus"]["max_length"],
    model=model,
    logger=logger
)

# ============================================================================
# Step 4: Ablate top features and measure d_loss per layer
# ============================================================================
logger.info("Step 4: Ablating top features and measuring effects...")

results_per_layer = {}  # {layer: [d_loss_1, d_loss_2, ..., d_loss_5]}

for layer in tqdm(config["layers"], desc="Ablating features"):
    top_features = top_features_per_layer[layer]
    d_losses = []

    for feat_id, mean_act in top_features:
        logger.info(f"Layer {layer}, Feature {feat_id}: Running ablation...")

        # Create intervention manager
        intervention_manager = FeatureIntervention(
            model=model,
            saes={layer: saes[layer]},
            hook=config["hook"],
            live_percentile=90,
            logger=logger
        )

        # No threshold needed since we always intervene
        intervention_manager.thresholds[(layer, feat_id)] = 0.0

        # Process test data in batches
        clean_losses = []
        ablation_losses = []

        num_batches = (len(test_data) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(test_data))
            batch_tokens = test_data[batch_start:batch_end]

            batch_dict = collate_batch(batch_tokens, device=device)

            with torch.no_grad():
                # Clean run
                clean_logits = model(batch_dict["input_ids"])
                clean_loss = torch.nn.functional.cross_entropy(
                    clean_logits[:, :-1, :].reshape(-1, clean_logits.shape[-1]),
                    batch_dict["input_ids"][:, 1:].reshape(-1),
                    reduction='none'
                )
                # Mask padding
                mask = batch_dict["attention_mask"][:, 1:].reshape(-1).bool()
                clean_loss = clean_loss[mask].mean().item()
                clean_losses.append(clean_loss)

                # Ablation run
                hook_fn = intervention_manager.create_intervention_hook(
                    layer, feat_id, alpha=0.0
                )
                hook_name = f"blocks.{layer}.hook_{config['hook']}"

                with model.hooks([(hook_name, hook_fn)]):
                    ablation_logits = model(batch_dict["input_ids"])

                ablation_loss = torch.nn.functional.cross_entropy(
                    ablation_logits[:, :-1, :].reshape(-1, ablation_logits.shape[-1]),
                    batch_dict["input_ids"][:, 1:].reshape(-1),
                    reduction='none'
                )
                ablation_loss = ablation_loss[mask].mean().item()
                ablation_losses.append(ablation_loss)

        # Compute average d_loss for this feature
        avg_clean = np.mean(clean_losses)
        avg_ablation = np.mean(ablation_losses)
        d_loss = avg_ablation - avg_clean

        logger.info(
            f"  Layer {layer}, Feature {feat_id}: "
            f"clean={avg_clean:.4f}, ablation={avg_ablation:.4f}, "
            f"d_loss={d_loss:.4f}"
        )

        d_losses.append(d_loss)

    results_per_layer[layer] = d_losses

# ============================================================================
# Step 5: Compute statistics and plot
# ============================================================================
logger.info("Step 5: Computing statistics and generating plot...")

layers = []
mean_d_losses = []
ci_lows = []
ci_highs = []

for layer in config["layers"]:
    d_losses = results_per_layer[layer]

    mean_d_loss = np.mean(d_losses)
    std_d_loss = np.std(d_losses, ddof=1)
    n = len(d_losses)

    # 95% CI: mean ± 1.96 * (std / sqrt(n))
    ci = 1.96 * (std_d_loss / np.sqrt(n))

    layers.append(layer)
    mean_d_losses.append(mean_d_loss)
    ci_lows.append(mean_d_loss - ci)
    ci_highs.append(mean_d_loss + ci)

    logger.info(
        f"Layer {layer}: mean d_loss = {mean_d_loss:.4f}, "
        f"95% CI = [{mean_d_loss - ci:.4f}, {mean_d_loss + ci:.4f}]"
    )

# Create plot
plt.figure(figsize=(10, 6))
plt.errorbar(
    layers,
    mean_d_losses,
    yerr=[
        np.array(mean_d_losses) - np.array(ci_lows),
        np.array(ci_highs) - np.array(mean_d_losses)
    ],
    fmt='o-',
    capsize=5,
    capthick=2,
    markersize=8,
    linewidth=2,
    label='Mean Δ Loss (top 5 features)'
)

plt.xlabel('Layer', fontsize=14)
plt.ylabel('Mean Δ Loss (ablation effect)', fontsize=14)
plt.title('Effect of Ablating Top 5 Most Active Features per Layer', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xticks(layers)
plt.tight_layout()

# Save plot
output_dir = Path("outputs/plots")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "top_features_by_layer.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
logger.info(f"Saved plot to {output_path}")

# Save data
import csv
csv_path = Path("outputs/csv") / "top_features_by_layer.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['layer', 'mean_d_loss', 'ci_low', 'ci_high', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'])

    for layer in config["layers"]:
        top_features = [str(feat_id) for feat_id, _ in top_features_per_layer[layer]]
        idx = layers.index(layer)
        writer.writerow([
            layer,
            mean_d_losses[idx],
            ci_lows[idx],
            ci_highs[idx],
            *top_features
        ])

logger.info(f"Saved data to {csv_path}")
logger.info("Analysis complete!")

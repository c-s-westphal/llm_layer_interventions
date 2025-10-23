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

from src.data import CorpusLoader, collate_batch
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
config["layers"] = list(range(1, 12))  # Layers 1-11 (skip layer 0 embeddings)
config["calibration_passages"] = 200  # Enough to get good statistics
config["test_passages"] = 500  # Test on reasonable corpus size

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
logger.info("Step 2: Loading calibration corpus...")

corpus_loader = CorpusLoader(
    corpus_name=config["corpus_name"],
    max_passages=config["calibration_passages"],
    max_len=config["max_len"],
    tokenizer=model.tokenizer,
    logger=logger
)

calibration_data, calibration_texts = corpus_loader.load_and_tokenize()
logger.info(f"Loaded {len(calibration_data)} calibration passages")

logger.info("Finding top 5 most active features per layer...")

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

test_corpus_loader = CorpusLoader(
    corpus_name=config["corpus_name"],
    max_passages=config["test_passages"],
    max_len=config["max_len"],
    tokenizer=model.tokenizer,
    logger=logger
)

test_data, test_texts = test_corpus_loader.load_and_tokenize()
logger.info(f"Loaded {len(test_data)} test passages")

# ============================================================================
# Step 4: Ablate top features and measure effects
# ============================================================================
logger.info("Step 4: Ablating top features and measuring effects...")

results_per_layer = {}  # {layer: {'d_loss': [...], 'kl': [...], 'top10_decrease': [...]}}
snippets_per_layer = {}  # {layer: {feat_id: snippet_info}}

for layer in tqdm(config["layers"], desc="Ablating features"):
    top_features = top_features_per_layer[layer]
    d_losses = []
    kl_divs = []
    top10_decreases = []
    snippets_per_layer[layer] = {}

    for feat_id, mean_act in top_features:
        logger.info(f"Layer {layer}, Feature {feat_id}: Running ablation...")

        sae = saes[layer]

        # First pass: Get clean output probabilities to find top-10 tokens
        logger.info("  First pass: Finding top-10 tokens by clean probability...")

        all_clean_probs = []  # Collect probabilities across corpus

        # Quick pass through small sample to find top tokens
        sample_size = min(50, len(test_data))
        for sample_idx in range(0, sample_size, batch_size):
            batch_end = min(sample_idx + batch_size, sample_size)
            batch_tokens = test_data[sample_idx:batch_end]
            batch_dict = collate_batch(batch_tokens, device=device)

            with torch.no_grad():
                clean_logits = model(batch_dict["input_ids"])
                clean_probs = torch.softmax(clean_logits[:, :-1, :], dim=-1)  # [batch, seq-1, vocab]

                # Average over batch and sequence
                mask = batch_dict["attention_mask"][:, 1:].bool()
                avg_probs = (clean_probs * mask.unsqueeze(-1)).sum(dim=(0, 1)) / mask.sum()  # [vocab]
                all_clean_probs.append(avg_probs.cpu())

        # Average across sample batches
        avg_clean_probs = torch.stack(all_clean_probs).mean(dim=0)  # [vocab_size]

        # Get top-10 tokens by probability
        top_10_tokens = torch.topk(avg_clean_probs, k=10).indices.tolist()
        top_10_probs = torch.topk(avg_clean_probs, k=10).values.tolist()

        logger.info(f"  Top-10 tokens by clean probability: {[repr(model.tokenizer.decode([t])) for t in top_10_tokens]}")
        logger.info(f"  Their clean probabilities: {[f'{p:.6f}' for p in top_10_probs]}")

        # Create intervention manager
        intervention_manager = FeatureIntervention(
            model=model,
            saes={layer: sae},
            hook=config["hook"],
            live_percentile=90,
            logger=logger
        )

        # No threshold needed since we always intervene
        intervention_manager.thresholds[(layer, feat_id)] = 0.0

        # Second pass: Measure ablation effects on full corpus
        logger.info("  Second pass: Measuring ablation effects...")

        # Track metrics across corpus
        clean_losses = []
        ablation_losses = []
        kls = []
        top10_prob_decreases = []

        # Track max decrease for snippet
        max_decrease = 0.0
        max_decrease_info = None

        num_batches = (len(test_data) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(test_data))
            batch_tokens = test_data[batch_start:batch_end]

            batch_dict = collate_batch(batch_tokens, device=device)

            with torch.no_grad():
                # Clean run
                clean_logits = model(batch_dict["input_ids"])
                clean_probs = torch.softmax(clean_logits[:, :-1, :], dim=-1)  # [batch, seq-1, vocab]

                clean_loss = torch.nn.functional.cross_entropy(
                    clean_logits[:, :-1, :].reshape(-1, clean_logits.shape[-1]),
                    batch_dict["input_ids"][:, 1:].reshape(-1),
                    reduction='none'
                )
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

                ablation_probs = torch.softmax(ablation_logits[:, :-1, :], dim=-1)  # [batch, seq-1, vocab]

                ablation_loss = torch.nn.functional.cross_entropy(
                    ablation_logits[:, :-1, :].reshape(-1, ablation_logits.shape[-1]),
                    batch_dict["input_ids"][:, 1:].reshape(-1),
                    reduction='none'
                )
                ablation_loss = ablation_loss[mask].mean().item()
                ablation_losses.append(ablation_loss)

                # Calculate KL divergence for each position
                # KL(P_clean || P_ablated) = sum(P_clean * log(P_clean / P_ablated))
                mask_2d = batch_dict["attention_mask"][:, 1:].bool()  # [batch, seq-1]

                kl_per_position = (clean_probs * (torch.log(clean_probs + 1e-10) - torch.log(ablation_probs + 1e-10))).sum(dim=-1)  # [batch, seq-1]
                kl_masked = kl_per_position[mask_2d]
                kls.extend(kl_masked.cpu().tolist())

                # Calculate top-10 token probability decrease
                top_10_tensor = torch.tensor(top_10_tokens, device=device)

                clean_top10_probs = clean_probs[:, :, top_10_tensor].mean(dim=-1)  # [batch, seq-1]
                ablated_top10_probs = ablation_probs[:, :, top_10_tensor].mean(dim=-1)  # [batch, seq-1]

                relative_decrease = (clean_top10_probs - ablated_top10_probs) / (clean_top10_probs + 1e-10)  # [batch, seq-1]
                relative_decrease_masked = relative_decrease[mask_2d]
                top10_prob_decreases.extend(relative_decrease_masked.cpu().tolist())

                # Debug: Print first batch stats
                if batch_idx == 0:
                    b, p = 0, 0
                    if mask_2d[b, p]:
                        logger.info(f"    [DEBUG first position]:")
                        logger.info(f"      Clean top-10 avg prob: {clean_top10_probs[b, p].item():.6f}")
                        logger.info(f"      Ablated top-10 avg prob: {ablated_top10_probs[b, p].item():.6f}")
                        logger.info(f"      Relative change: {relative_decrease[b, p].item():.4f}")
                        # Show individual token probs
                        for i, tok_idx in enumerate(top_10_tokens[:3]):  # Just first 3
                            tok_str = model.tokenizer.decode([tok_idx])
                            clean_p = clean_probs[b, p, tok_idx].item()
                            ablated_p = ablation_probs[b, p, tok_idx].item()
                            change = (clean_p - ablated_p) / (clean_p + 1e-10)
                            logger.info(f"        {repr(tok_str)}: {clean_p:.6f} → {ablated_p:.6f} (change: {change:+.4f})")

                # Find max decrease in this batch for snippet
                for b_idx in range(len(batch_tokens)):
                    seq_len = mask_2d[b_idx].sum().item()
                    for pos_idx in range(seq_len):
                        decrease = relative_decrease[b_idx, pos_idx].item()
                        if decrease > max_decrease:
                            max_decrease = decrease
                            passage_idx = batch_start + b_idx
                            max_decrease_info = {
                                'passage_idx': passage_idx,
                                'text': test_texts[passage_idx],
                                'tokens': batch_tokens[b_idx],
                                'position': pos_idx,
                                'decrease': decrease,
                                'clean_prob': clean_top10_probs[b_idx, pos_idx].item(),
                                'ablated_prob': ablated_top10_probs[b_idx, pos_idx].item()
                            }

        # Compute averages
        avg_d_loss = np.mean(ablation_losses) - np.mean(clean_losses)
        avg_kl = np.mean(kls)
        avg_top10_decrease = np.mean(top10_prob_decreases)

        logger.info(
            f"  Layer {layer}, Feature {feat_id}: "
            f"d_loss={avg_d_loss:.4f}, KL={avg_kl:.4f}, top10_decrease={avg_top10_decrease:.4f}"
        )

        d_losses.append(avg_d_loss)
        kl_divs.append(avg_kl)
        top10_decreases.append(avg_top10_decrease)
        snippets_per_layer[layer][feat_id] = {
            'top_10_tokens': top_10_tokens,
            'top_10_token_strings': [model.tokenizer.decode([t]) for t in top_10_tokens],
            'max_decrease_info': max_decrease_info
        }

    results_per_layer[layer] = {
        'd_loss': d_losses,
        'kl': kl_divs,
        'top10_decrease': top10_decreases
    }

# ============================================================================
# Step 5: Compute statistics and generate plots
# ============================================================================
logger.info("Step 5: Computing statistics and generating plots...")

# Collect statistics for each metric
layers = []
mean_d_losses = []
mean_kls = []
mean_top10_decreases = []

d_loss_cis = []
kl_cis = []
top10_cis = []

for layer in config["layers"]:
    results = results_per_layer[layer]

    # D-loss statistics
    d_losses = results['d_loss']
    mean_d_loss = np.mean(d_losses)
    std_d_loss = np.std(d_losses, ddof=1)
    n = len(d_losses)
    ci_d_loss = 1.96 * (std_d_loss / np.sqrt(n))

    # KL statistics
    kls = results['kl']
    mean_kl = np.mean(kls)
    std_kl = np.std(kls, ddof=1)
    ci_kl = 1.96 * (std_kl / np.sqrt(n))

    # Top-10 decrease statistics
    top10s = results['top10_decrease']
    mean_top10 = np.mean(top10s)
    std_top10 = np.std(top10s, ddof=1)
    ci_top10 = 1.96 * (std_top10 / np.sqrt(n))

    layers.append(layer)
    mean_d_losses.append(mean_d_loss)
    mean_kls.append(mean_kl)
    mean_top10_decreases.append(mean_top10)

    d_loss_cis.append((mean_d_loss - ci_d_loss, mean_d_loss + ci_d_loss))
    kl_cis.append((mean_kl - ci_kl, mean_kl + ci_kl))
    top10_cis.append((mean_top10 - ci_top10, mean_top10 + ci_top10))

    logger.info(
        f"Layer {layer}: d_loss={mean_d_loss:.4f}, KL={mean_kl:.4f}, top10_decrease={mean_top10:.4f}"
    )

# Create 3-panel plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: D-Loss
ax = axes[0]
d_loss_errs = [[mean_d_losses[i] - d_loss_cis[i][0] for i in range(len(layers))],
               [d_loss_cis[i][1] - mean_d_losses[i] for i in range(len(layers))]]
ax.errorbar(layers, mean_d_losses, yerr=d_loss_errs, fmt='o-', capsize=5, capthick=2, markersize=8, linewidth=2)
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Mean Δ Loss', fontsize=12)
ax.set_title('Ablation Effect: Δ Loss', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xticks(layers)

# Plot 2: KL Divergence
ax = axes[1]
kl_errs = [[mean_kls[i] - kl_cis[i][0] for i in range(len(layers))],
           [kl_cis[i][1] - mean_kls[i] for i in range(len(layers))]]
ax.errorbar(layers, mean_kls, yerr=kl_errs, fmt='o-', capsize=5, capthick=2, markersize=8, linewidth=2, color='orange')
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Mean KL Divergence', fontsize=12)
ax.set_title('KL(P_clean || P_ablated)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xticks(layers)

# Plot 3: Top-10 Token Probability Decrease
ax = axes[2]
top10_errs = [[mean_top10_decreases[i] - top10_cis[i][0] for i in range(len(layers))],
              [top10_cis[i][1] - mean_top10_decreases[i] for i in range(len(layers))]]
ax.errorbar(layers, mean_top10_decreases, yerr=top10_errs, fmt='o-', capsize=5, capthick=2, markersize=8, linewidth=2, color='green')
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Relative Probability Decrease', fontsize=12)
ax.set_title('Top-10 Token Prob Decrease', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xticks(layers)

plt.tight_layout()

# Save plot
output_dir = Path("outputs/plots")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "top_features_metrics_by_layer.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
logger.info(f"Saved plot to {output_path}")
plt.close()

# Save CSV data
import csv
csv_path = Path("outputs/csv") / "top_features_metrics_by_layer.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'layer',
        'mean_d_loss', 'd_loss_ci_low', 'd_loss_ci_high',
        'mean_kl', 'kl_ci_low', 'kl_ci_high',
        'mean_top10_decrease', 'top10_ci_low', 'top10_ci_high',
        'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'
    ])

    for layer in config["layers"]:
        top_features = [str(feat_id) for feat_id, _ in top_features_per_layer[layer]]
        idx = layers.index(layer)
        writer.writerow([
            layer,
            mean_d_losses[idx], d_loss_cis[idx][0], d_loss_cis[idx][1],
            mean_kls[idx], kl_cis[idx][0], kl_cis[idx][1],
            mean_top10_decreases[idx], top10_cis[idx][0], top10_cis[idx][1],
            *top_features
        ])

logger.info(f"Saved data to {csv_path}")

# ============================================================================
# Step 6: Save snippets
# ============================================================================
logger.info("Step 6: Saving snippets...")

snippets_dir = Path("outputs/snippets")
snippets_dir.mkdir(parents=True, exist_ok=True)

for layer in config["layers"]:
    layer_dir = snippets_dir / f"layer_{layer}"
    layer_dir.mkdir(exist_ok=True)

    for feat_id, snippet_data in snippets_per_layer[layer].items():
        max_info = snippet_data['max_decrease_info']

        if max_info is None:
            continue

        # Create snippet file
        snippet_path = layer_dir / f"feature_{feat_id}_max_decrease.txt"

        with open(snippet_path, 'w') as f:
            f.write(f"Layer {layer}, Feature {feat_id}\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Top-10 Output Tokens:\n")
            for i, token_str in enumerate(snippet_data['top_10_token_strings'], 1):
                f.write(f"  {i}. {repr(token_str)}\n")
            f.write("\n")

            f.write(f"Maximum Probability Decrease: {max_info['decrease']:.4f}\n")
            f.write(f"  Clean prob (top-10 avg): {max_info['clean_prob']:.6f}\n")
            f.write(f"  Ablated prob (top-10 avg): {max_info['ablated_prob']:.6f}\n")
            f.write(f"  Relative decrease: {max_info['decrease']:.2%}\n\n")

            f.write(f"Context (position {max_info['position']} in passage):\n")
            f.write("-" * 80 + "\n")

            # Decode tokens with context window
            tokens = max_info['tokens'].cpu().tolist()
            pos = max_info['position']

            # Context: 10 tokens before and after
            context_start = max(0, pos - 10)
            context_end = min(len(tokens), pos + 11)

            context_tokens = tokens[context_start:context_end]
            context_text = model.tokenizer.decode(context_tokens)

            # Highlight the position
            before_text = model.tokenizer.decode(tokens[context_start:pos+1])
            target_token = model.tokenizer.decode([tokens[pos+1]]) if pos+1 < len(tokens) else ""
            after_text = model.tokenizer.decode(tokens[pos+2:context_end]) if pos+2 < len(tokens) else ""

            f.write(f"{before_text}[{target_token}]{after_text}\n")
            f.write("-" * 80 + "\n\n")

            f.write("Full passage:\n")
            f.write(max_info['text'][:500])  # First 500 chars
            if len(max_info['text']) > 500:
                f.write("...")
            f.write("\n")

logger.info(f"Saved snippets to {snippets_dir}")
logger.info("Analysis complete!")

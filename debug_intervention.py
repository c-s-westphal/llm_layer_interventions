"""Debug script to test if interventions are working correctly.

This script tests:
1. Does alpha=1 produce identical outputs to clean (no intervention)?
2. Does alpha=0 actually zero out features?
3. Are activations being modified correctly?
"""

import os
# Fix RunPod environment issue
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

print("=" * 80)
print("INTERVENTION DEBUG SCRIPT")
print("=" * 80)

# Load model
print("\n1. Loading GPT-2 Small...")
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
print(f"   Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

# Load SAEs for layer 0 and 1
print("\n2. Loading SAEs for layers 0 and 1...")
saes = {}
for layer_idx in [0, 1]:
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{layer_idx}.hook_resid_pre",
        device="cuda"
    )
    saes[layer_idx] = sae
    print(f"   Layer {layer_idx} SAE loaded: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

# Test input
test_text = "The quick brown fox jumps over the lazy dog"
tokens = model.to_tokens(test_text)
print(f"\n3. Test input: '{test_text}'")
print(f"   Tokens shape: {tokens.shape}")

# Find active features for each layer
print(f"\n4. Finding active features on this text...")

with torch.no_grad():
    hook_names = [f"blocks.{layer_idx}.hook_resid_pre" for layer_idx in [0, 1]]
    _, cache = model.run_with_cache(tokens, names_filter=hook_names)

active_features = {}
for layer_idx in [0, 1]:
    hook_name = f"blocks.{layer_idx}.hook_resid_pre"
    acts = cache[hook_name]  # [1, seq, d_model]
    sae_acts = saes[layer_idx].encode(acts)  # [1, seq, d_sae]

    # Get max activation across all positions for each feature
    max_acts = sae_acts[0].max(dim=0).values  # [d_sae]

    # Find top 10 most active features
    top_k = 10
    top_values, top_indices = torch.topk(max_acts, k=top_k)

    active_features[layer_idx] = []
    print(f"\n   Layer {layer_idx} - Top {top_k} active features:")
    for i, (feat_idx, feat_val) in enumerate(zip(top_indices.tolist(), top_values.tolist())):
        print(f"      #{i+1}: Feature {feat_idx:5d} with max activation {feat_val:.4f}")
        active_features[layer_idx].append((feat_idx, feat_val))

# Select most active feature from layer 0 for detailed testing
layer = 0
feature_idx = active_features[layer][0][0]
sae = saes[layer]
print(f"\n5. Selected feature {feature_idx} from layer {layer} for detailed testing (activation: {active_features[layer][0][1]:.4f})")

# ============================================================================
# TEST 1: Clean run (no intervention)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: Clean run (no intervention)")
print("=" * 80)

with torch.no_grad():
    clean_logits = model(tokens)
    clean_loss = torch.nn.functional.cross_entropy(
        clean_logits[0, :-1, :],
        tokens[0, 1:],
        reduction='mean'
    )

print(f"Clean loss: {clean_loss.item():.6f}")
print(f"Clean logits shape: {clean_logits.shape}")
print(f"Sample clean logits (first 5): {clean_logits[0, -1, :5].tolist()}")

# ============================================================================
# TEST 2: Run with alpha=1 (should show only SAE reconstruction error)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Intervention with alpha=1.0 (SAE reconstruction baseline)")
print("=" * 80)

def intervention_hook_alpha1(activation, hook):
    """Intervention with alpha=1 (no change)."""
    print(f"   Hook called! Activation shape: {activation.shape}")

    # Encode with SAE (ALL positions)
    sae_acts = sae.encode(activation)  # [batch, seq, d_sae]
    print(f"   SAE encoding shape: {sae_acts.shape}")

    # Check feature values across all positions
    feature_vals = sae_acts[:, :, feature_idx]  # [batch, seq]
    print(f"   Feature {feature_idx} max value across sequence: {feature_vals.max().item():.6f}")
    print(f"   Feature {feature_idx} at last position: {feature_vals[0, -1].item():.6f}")

    # Apply alpha=1 (no change)
    sae_acts[:, :, feature_idx] *= 1.0

    # Decode back (ALL positions)
    modified_act = sae.decode(sae_acts)

    # Check reconstruction error
    recon_error = (activation - modified_act).abs().mean()
    print(f"   SAE reconstruction error: {recon_error.item():.6f}")

    return modified_act

hook_name = f"blocks.{layer}.hook_resid_pre"

with torch.no_grad():
    with model.hooks([(hook_name, intervention_hook_alpha1)]):
        alpha1_logits = model(tokens)

    alpha1_loss = torch.nn.functional.cross_entropy(
        alpha1_logits[0, :-1, :],
        tokens[0, 1:],
        reduction='mean'
    )

print(f"Alpha=1 loss: {alpha1_loss.item():.6f}")
print(f"Sample alpha=1 logits (first 5): {alpha1_logits[0, -1, :5].tolist()}")

# Compare
delta_loss_alpha1 = alpha1_loss.item() - clean_loss.item()
logits_diff = (clean_logits - alpha1_logits).abs().max().item()

print(f"\n*** COMPARISON: Clean vs Alpha=1 ***")
print(f"Delta loss: {delta_loss_alpha1:.6f} (SAE reconstruction baseline)")
print(f"Max logits difference: {logits_diff:.6f}")

# Should match SAE baseline (~0.077)
if abs(delta_loss_alpha1) < 0.05 or abs(delta_loss_alpha1) > 0.15:
    print("⚠️  WARNING: Alpha=1 loss unexpected (should be ~0.08 for SAE reconstruction)")
else:
    print("✅ Alpha=1 shows expected SAE reconstruction error")

# ============================================================================
# TEST 3: Run with alpha=0 (ablation - should change a lot)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Intervention with alpha=0.0 (ablation)")
print("=" * 80)

def intervention_hook_alpha0(activation, hook):
    """Intervention with alpha=0 (ablation)."""
    print(f"   Hook called! Activation shape: {activation.shape}")

    # Encode with SAE (ALL positions)
    sae_acts = sae.encode(activation)  # [batch, seq, d_sae]

    # Check feature values across all positions
    feature_vals = sae_acts[:, :, feature_idx]  # [batch, seq]
    print(f"   Feature {feature_idx} max value across sequence: {feature_vals.max().item():.6f}")
    print(f"   Feature {feature_idx} mean value (non-zero): {feature_vals[feature_vals > 0].mean().item() if (feature_vals > 0).any() else 0:.6f}")

    # Apply alpha=0 (ablate ALL positions)
    sae_acts[:, :, feature_idx] *= 0.0
    print(f"   Feature {feature_idx} after ablation: {sae_acts[:, :, feature_idx].max().item():.6f}")

    # Decode back (ALL positions)
    modified_act = sae.decode(sae_acts)

    return modified_act

with torch.no_grad():
    with model.hooks([(hook_name, intervention_hook_alpha0)]):
        alpha0_logits = model(tokens)

    alpha0_loss = torch.nn.functional.cross_entropy(
        alpha0_logits[0, :-1, :],
        tokens[0, 1:],
        reduction='mean'
    )

print(f"Alpha=0 loss: {alpha0_loss.item():.6f}")
print(f"Sample alpha=0 logits (first 5): {alpha0_logits[0, -1, :5].tolist()}")

# Compare
delta_loss_alpha0 = alpha0_loss.item() - clean_loss.item()
logits_diff_alpha0 = (clean_logits - alpha0_logits).abs().max().item()

print(f"\n*** COMPARISON: Clean vs Alpha=0 ***")
print(f"Delta loss: {delta_loss_alpha0:.6f}")
print(f"Max logits difference: {logits_diff_alpha0:.6f}")

# Should show more than just SAE reconstruction (actual ablation effect)
delta_ablation_effect = abs(delta_loss_alpha0 - delta_loss_alpha1)
print(f"Ablation effect (vs alpha=1 baseline): {delta_ablation_effect:.6f}")

if delta_ablation_effect < 0.01:
    print("⚠️  WARNING: Alpha=0 same as alpha=1 (no ablation effect beyond reconstruction)")
else:
    print("✅ Alpha=0 ablation has measurable effect")

# ============================================================================
# TEST 4: Check pipeline implementation with alpha=1 (baseline reconstruction)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: Checking pipeline implementation (alpha=1)")
print("=" * 80)

# Simulate what the pipeline does
from src.intervene import FeatureIntervention
from src.metrics import InterventionMetrics

print("Creating FeatureIntervention manager...")
intervention_manager = FeatureIntervention(
    model=model,
    saes={layer: sae},
    hook="resid_pre",
    live_percentile=90,
)

# Mock calibration
intervention_manager.thresholds[(layer, feature_idx)] = 0.001

print("Creating intervention hook via pipeline (alpha=1.0)...")
hook_fn = intervention_manager.create_intervention_hook(layer, feature_idx, alpha=1.0)

with torch.no_grad():
    with model.hooks([(hook_name, hook_fn)]):
        pipeline_alpha1_logits = model(tokens)

    pipeline_alpha1_loss = torch.nn.functional.cross_entropy(
        pipeline_alpha1_logits[0, :-1, :],
        tokens[0, 1:],
        reduction='mean'
    )

delta_pipeline_alpha1 = pipeline_alpha1_loss.item() - clean_loss.item()
logits_diff_pipeline = (clean_logits - pipeline_alpha1_logits).abs().max().item()

print(f"\nPipeline alpha=1 loss: {pipeline_alpha1_loss.item():.6f}")
print(f"Delta from clean: {delta_pipeline_alpha1:.6f} (SAE reconstruction baseline)")
print(f"Max logits difference: {logits_diff_pipeline:.6f}")

# Should match the direct alpha=1 test (both apply SAE reconstruction)
if abs(delta_pipeline_alpha1 - delta_loss_alpha1) > 0.001:
    print("⚠️  ISSUE: Pipeline alpha=1 doesn't match direct alpha=1")
else:
    print("✅ Pipeline matches direct implementation")

# ============================================================================
# TEST 5: Check if pipeline ablation (alpha=0) works on ACTIVE feature
# ============================================================================
print("\n" + "=" * 80)
print("TEST 5: Checking pipeline ablation (alpha=0) on ACTIVE feature")
print("=" * 80)

# Use an ACTIVE feature (second most active to avoid same as primary test feature)
test_feature_idx = active_features[layer][1][0] if len(active_features[layer]) > 1 else active_features[layer][0][0]
test_feature_activation = active_features[layer][1][1] if len(active_features[layer]) > 1 else active_features[layer][0][1]

print(f"Testing with feature {test_feature_idx} (activation: {test_feature_activation:.4f})...")

# Mock calibration
intervention_manager.thresholds[(layer, test_feature_idx)] = 0.001

print("Creating intervention hook via pipeline (alpha=0.0)...")
hook_fn_alpha0 = intervention_manager.create_intervention_hook(layer, test_feature_idx, alpha=0.0)

with torch.no_grad():
    with model.hooks([(hook_name, hook_fn_alpha0)]):
        pipeline_alpha0_logits = model(tokens)

    pipeline_alpha0_loss = torch.nn.functional.cross_entropy(
        pipeline_alpha0_logits[0, :-1, :],
        tokens[0, 1:],
        reduction='mean'
    )

delta_pipeline_alpha0 = pipeline_alpha0_loss.item() - clean_loss.item()
logits_diff_alpha0_pipeline = (clean_logits - pipeline_alpha0_logits).abs().max().item()

print(f"\nPipeline alpha=0 loss: {pipeline_alpha0_loss.item():.6f}")
print(f"Delta from clean: {delta_pipeline_alpha0:.6f}")
print(f"Max logits difference: {logits_diff_alpha0_pipeline:.6f}")

# Should differ from alpha=1 (ablation effect beyond reconstruction)
delta_vs_alpha1 = abs(delta_pipeline_alpha0 - delta_pipeline_alpha1)
print(f"Delta vs alpha=1: {delta_vs_alpha1:.6f}")

if delta_vs_alpha1 < 0.001:
    print("⚠️  WARNING: Alpha=0 same as alpha=1 (no ablation effect)")
else:
    print("✅ Alpha=0 differs from alpha=1 (ablation works)")

# ============================================================================
# TEST 6: Verify SAE reconstruction baseline is consistent
# ============================================================================
print("\n" + "=" * 80)
print("TEST 6: Measure SAE reconstruction error baseline")
print("=" * 80)

# Test with a "no-op" intervention (multiply all features by 1.0)
def sae_passthrough_hook(activations, hook):
    """Hook that applies SAE reconstruction without modifying features."""
    sae_acts = sae.encode(activations)
    # Don't modify any features - just reconstruct
    reconstructed = sae.decode(sae_acts)
    return reconstructed

with torch.no_grad():
    with model.hooks([(hook_name, sae_passthrough_hook)]):
        sae_baseline_logits = model(tokens)

    sae_baseline_loss = torch.nn.functional.cross_entropy(
        sae_baseline_logits[0, :-1, :],
        tokens[0, 1:],
        reduction='mean'
    )

delta_sae_baseline = sae_baseline_loss.item() - clean_loss.item()

print(f"SAE reconstruction baseline loss: {sae_baseline_loss.item():.6f}")
print(f"Delta from clean: {delta_sae_baseline:.6f}")
print(f"This is the minimum 'noise floor' for all interventions")

# This should match alpha=1 intervention
if abs(delta_sae_baseline - delta_pipeline_alpha1) < 0.001:
    print("✅ Alpha=1 matches pure SAE reconstruction (as expected)")
else:
    print("⚠️  WARNING: Alpha=1 differs from pure reconstruction")

# ============================================================================
# TEST 7: Compare interventions across layers 0 and 1
# ============================================================================
print("\n" + "=" * 80)
print("TEST 7: Compare interventions across layers 0 and 1")
print("=" * 80)

for test_layer in [0, 1]:
    print(f"\n--- Testing Layer {test_layer} ---")

    # Get most active feature for this layer
    test_feat = active_features[test_layer][0][0]
    test_feat_act = active_features[test_layer][0][1]

    print(f"Feature {test_feat} (activation: {test_feat_act:.4f})")

    # Create intervention manager for this layer
    test_intervention_manager = FeatureIntervention(
        model=model,
        saes={test_layer: saes[test_layer]},
        hook="resid_pre",
        live_percentile=90,
    )
    test_intervention_manager.thresholds[(test_layer, test_feat)] = 0.001

    # Test alpha=0 and alpha=2
    test_hook_name = f"blocks.{test_layer}.hook_resid_pre"

    results = {}
    for alpha_val in [0.0, 1.0, 2.0]:
        hook_fn_test = test_intervention_manager.create_intervention_hook(test_layer, test_feat, alpha=alpha_val)

        with torch.no_grad():
            with model.hooks([(test_hook_name, hook_fn_test)]):
                test_logits = model(tokens)

            test_loss = torch.nn.functional.cross_entropy(
                test_logits[0, :-1, :],
                tokens[0, 1:],
                reduction='mean'
            )

        results[alpha_val] = test_loss.item()
        print(f"  Alpha={alpha_val}: loss={test_loss.item():.6f} (Δ={test_loss.item() - clean_loss.item():+.6f})")

    # Check if alpha makes a difference
    delta_0_vs_1 = abs(results[0.0] - results[1.0])
    delta_2_vs_1 = abs(results[2.0] - results[1.0])

    print(f"  |Alpha=0 - Alpha=1|: {delta_0_vs_1:.6f}")
    print(f"  |Alpha=2 - Alpha=1|: {delta_2_vs_1:.6f}")

    if delta_0_vs_1 > 0.001 or delta_2_vs_1 > 0.001:
        print(f"  ✅ Layer {test_layer}: Interventions have measurable effects")
    else:
        print(f"  ⚠️  Layer {test_layer}: No intervention effects detected")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nLoss values:")
print(f"  Clean:                  {clean_loss.item():.6f}")
print(f"  SAE reconstruction:     {sae_baseline_loss.item():.6f}  (Δ = {delta_sae_baseline:+.6f}) [baseline]")
print(f"  Alpha=1 (direct):       {alpha1_loss.item():.6f}  (Δ = {delta_loss_alpha1:+.6f})")
print(f"  Alpha=1 (pipeline):     {pipeline_alpha1_loss.item():.6f}  (Δ = {delta_pipeline_alpha1:+.6f})")
print(f"  Alpha=0 (direct):       {alpha0_loss.item():.6f}  (Δ = {delta_loss_alpha0:+.6f})")
print(f"  Alpha=0 (pipeline):     {pipeline_alpha0_loss.item():.6f}  (Δ = {delta_pipeline_alpha0:+.6f})")

print(f"\n✅ = Pass, ⚠️ = Fail")
print(f"  Alpha=1 matches SAE baseline:          {'✅' if abs(delta_loss_alpha1 - delta_sae_baseline) < 0.001 else '⚠️ '}")
print(f"  Pipeline matches direct (alpha=1):     {'✅' if abs(delta_pipeline_alpha1 - delta_loss_alpha1) < 0.001 else '⚠️ '}")
print(f"  Alpha=0 has ablation effect:           {'✅' if abs(delta_loss_alpha0 - delta_loss_alpha1) > 0.01 else '⚠️ '}")
print(f"  Pipeline alpha=0 differs from alpha=1: {'✅' if abs(delta_pipeline_alpha0 - delta_pipeline_alpha1) > 0.001 else '⚠️ '}")

print(f"\n🔍 Key Diagnostic Info:")
print(f"  SAE reconstruction error: {delta_sae_baseline:.6f} (noise floor for all interventions)")
print(f"  Primary test feature (layer {layer}): {feature_idx} with activation {active_features[layer][0][1]:.4f}")
print(f"  Secondary test feature (layer {layer}): {test_feature_idx} with activation {test_feature_activation:.4f}")

print(f"\n📊 Top Active Features by Layer:")
for test_layer in [0, 1]:
    print(f"  Layer {test_layer}: Feature {active_features[test_layer][0][0]} (act={active_features[test_layer][0][1]:.4f}), "
          f"Feature {active_features[test_layer][1][0]} (act={active_features[test_layer][1][1]:.4f}), "
          f"Feature {active_features[test_layer][2][0]} (act={active_features[test_layer][2][1]:.4f})")

print("\n" + "=" * 80)

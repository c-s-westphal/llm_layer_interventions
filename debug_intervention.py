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

# Load one SAE for testing
print("\n2. Loading SAE for layer 0...")
sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.0.hook_resid_pre",
    device="cuda"
)
print(f"   SAE loaded: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

# Test input
test_text = "The quick brown fox jumps over the lazy dog"
tokens = model.to_tokens(test_text)
print(f"\n3. Test input: '{test_text}'")
print(f"   Tokens shape: {tokens.shape}")

# Pick a feature to test
layer = 0
feature_idx = 12453  # From your CSV

print(f"\n4. Testing feature {feature_idx} at layer {layer}")

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
# TEST 2: Run with alpha=1 (should be identical to clean)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Intervention with alpha=1.0 (should match clean)")
print("=" * 80)

def intervention_hook_alpha1(activation, hook):
    """Intervention with alpha=1 (no change)."""
    print(f"   Hook called! Activation shape: {activation.shape}")

    # Original activation
    orig_act = activation[0, -1].clone()

    # Encode with SAE
    sae_acts = sae.encode(activation[0, -1])
    print(f"   SAE encoding shape: {sae_acts.shape}")
    print(f"   Feature {feature_idx} original value: {sae_acts[feature_idx].item():.6f}")

    # Apply alpha=1 (no change)
    sae_acts[feature_idx] *= 1.0
    print(f"   Feature {feature_idx} after alpha=1: {sae_acts[feature_idx].item():.6f}")

    # Decode back
    modified_act = sae.decode(sae_acts)

    # Check reconstruction error
    recon_error = (orig_act - modified_act).abs().mean()
    print(f"   SAE reconstruction error: {recon_error.item():.6f}")

    # Replace activation
    activation[0, -1] = modified_act
    return activation

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
print(f"Delta loss: {delta_loss_alpha1:.6f} (should be small due to SAE reconstruction)")
print(f"Max logits difference: {logits_diff:.6f} (should be small)")

if abs(delta_loss_alpha1) > 0.1:
    print("⚠️  WARNING: Alpha=1 has large effect! Intervention may be broken.")
else:
    print("✅ Alpha=1 shows small reconstruction error (expected)")

# ============================================================================
# TEST 3: Run with alpha=0 (ablation - should change a lot)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Intervention with alpha=0.0 (ablation)")
print("=" * 80)

def intervention_hook_alpha0(activation, hook):
    """Intervention with alpha=0 (ablation)."""
    print(f"   Hook called! Activation shape: {activation.shape}")

    # Encode with SAE
    sae_acts = sae.encode(activation[0, -1])
    print(f"   Feature {feature_idx} original value: {sae_acts[feature_idx].item():.6f}")

    # Apply alpha=0 (ablate)
    sae_acts[feature_idx] *= 0.0
    print(f"   Feature {feature_idx} after alpha=0: {sae_acts[feature_idx].item():.6f}")

    # Decode back
    modified_act = sae.decode(sae_acts)

    # Replace activation
    activation[0, -1] = modified_act
    return activation

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

# Note: Even if feature isn't active (value=0), ablating it still goes through
# SAE reconstruction, so we expect SOME change (at minimum, reconstruction error)
if abs(delta_loss_alpha0) < 0.01:
    print("⚠️  WARNING: Alpha=0 has minimal effect (possible issue or inactive feature)")
else:
    print("✅ Alpha=0 changes output")

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
# TEST 5: Check if pipeline ablation (alpha=0) works
# ============================================================================
print("\n" + "=" * 80)
print("TEST 5: Checking pipeline ablation (alpha=0)")
print("=" * 80)

# Try a different feature
test_feature_idx = 100
print(f"Testing with feature {test_feature_idx}...")

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
print(f"  Alpha=1 has small effect:              {'✅' if abs(delta_loss_alpha1) < 0.1 else '⚠️ '}")
print(f"  Pipeline matches direct (alpha=1):     {'✅' if abs(delta_pipeline_alpha1 - delta_loss_alpha1) < 0.001 else '⚠️ '}")
print(f"  Pipeline matches SAE baseline:         {'✅' if abs(delta_pipeline_alpha1 - delta_sae_baseline) < 0.001 else '⚠️ '}")
print(f"  Alpha=0 differs from alpha=1:          {'✅' if abs(delta_pipeline_alpha0 - delta_pipeline_alpha1) > 0.001 else '⚠️ '}")

print(f"\n🔍 Key Diagnostic Info:")
print(f"  SAE reconstruction error: {delta_sae_baseline:.6f} (noise floor for all interventions)")
print(f"  Feature {feature_idx} activation: {sae.encode(model.run_with_cache(tokens, names_filter=[hook_name])[1][hook_name])[0, -1, feature_idx].item():.6f}")
print(f"  Feature {test_feature_idx} activation: {sae.encode(model.run_with_cache(tokens, names_filter=[hook_name])[1][hook_name])[0, -1, test_feature_idx].item():.6f}")

print("\n" + "=" * 80)

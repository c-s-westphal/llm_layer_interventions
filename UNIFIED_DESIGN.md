# Unified Discovery + Measurement Design

## Problem
Currently doing redundant work:
1. Causal discovery: Run baseline + ablated on calibration data
2. Measurement: Run baseline + ablated on test data
→ **2x passes through data, computing the same thing twice!**

## Solution: Single Pass

```python
def discover_and_measure_promoted_tokens(
    model, sae, layer, feature_id, hook,
    data, batch_size, activation_threshold,
    top_k=15, min_effect=0.0001
):
    """
    Single pass: discover top-K promoted tokens AND measure their effects.

    Process:
    1. Iterate through ALL data once
    2. At each position where feature > threshold:
       - Compute baseline probs (with feature)
       - Compute ablated probs (without feature)
       - Track change for EVERY token in vocabulary
    3. After all data processed:
       - Find tokens with mean_effect > min_effect (promoted)
       - Select top-K by effect size
       - Compute statistics (mean, median, std) on those top-K

    Returns:
        {
            'promoted_tokens': List[int],  # top-K token IDs
            'token_effects': Dict[int, float],  # all token effects
            'measurement_stats': {
                'mean': float,
                'median': float,
                'std': float,
                'min': float,
                'max': float,
                'num_active_positions': int
            },
            'control_stats': {...}  # from random feature
        }
    """
```

## Implementation Details

### Efficient Token Effect Accumulation

```python
# Track per-token effects across all positions
token_effect_sum = torch.zeros(vocab_size, device='cpu')
token_effect_squared_sum = torch.zeros(vocab_size, device='cpu')
token_count = 0

for batch in data:
    # Get activations
    feature_acts = get_feature_activations(...)
    active_mask = feature_acts > threshold

    if active_mask.sum() == 0:
        continue

    # Baseline vs ablated (single forward pass each)
    baseline_probs = run_with_reconstruction(...)  # [batch, seq, vocab]
    ablated_probs = run_with_ablation(...)         # [batch, seq, vocab]

    # Compute change at active positions
    change = baseline_probs[active_mask] - ablated_probs[active_mask]  # [n_active, vocab]

    # Accumulate statistics
    token_effect_sum += change.sum(dim=0).cpu()
    token_effect_squared_sum += (change ** 2).sum(dim=0).cpu()
    token_count += active_mask.sum().item()

# Compute mean and std for each token
mean_effects = token_effect_sum / token_count
std_effects = torch.sqrt(
    token_effect_squared_sum / token_count - mean_effects ** 2
)

# Filter for promoted (positive effect > threshold)
promoted_mask = mean_effects > min_effect
promoted_token_ids = torch.where(promoted_mask)[0]
promoted_effects = mean_effects[promoted_mask]

# Select top-K
top_k_indices = promoted_effects.topk(min(top_k, len(promoted_effects))).indices
top_k_token_ids = promoted_token_ids[top_k_indices].tolist()
```

### Combined Measurement

```python
# Now compute detailed stats on the top-K promoted tokens
# We already have their effects, just need to compute statistics
top_k_effects = mean_effects[top_k_token_ids]

stats = {
    'mean': top_k_effects.mean().item(),
    'median': top_k_effects.median().item(),
    'std': top_k_effects.std().item(),
    'min': top_k_effects.min().item(),
    'max': top_k_effects.max().item(),
    'num_active_positions': token_count
}
```

## Benefits

1. **2x faster**: Single pass instead of two separate passes
2. **Causal discovery**: Directly finds tokens the feature promotes
3. **Automatic filtering**: Only measures on promoted tokens (effect > 0.0001)
4. **Full vocabulary analysis**: Still get all token effects for deeper analysis

## Integration

Replace these two functions:
- `discover_feature_tokens_empirical()` (correlational, wrong)
- `measure_probability_changes()` (redundant pass)

With single function:
- `discover_and_measure_promoted_tokens()` (causal, efficient)

## Firing Rate Optimization

Currently computing firing rates for ALL features (270k):
```python
for feature_id in range(sae.cfg.d_sae):  # 24,576 features × 11 layers!
    compute_firing_rate(...)
```

Should only compute for labeled features (55):
```python
# Read CSV first
labeled_features = pd.read_csv("data/neuronpedia_features.csv")

# Only compute for labeled features
for layer in layers:
    layer_labeled = labeled_features[labeled_features['layer'] == layer]
    for _, row in layer_labeled.iterrows():
        feature_id = row['feature_id']
        compute_firing_rate(layer, feature_id, ...)
```

**Time savings**: 270k → 55 features = **5000x faster!**
- Old: 5.5 hours
- New: ~4 seconds

## Expected Output

```
--- Layer 8, Feature 10987 ('evidentials', conf=0.66) ---
  Discovering promoted tokens causally (threshold: effect > 0.0001)...

  Found 23 promoted tokens from 3870 active positions
  Top-5 PROMOTED tokens (causal discovery):
    1. ' but' (ID 475): effect=+0.0021
    2. ' and' (ID 290): effect=+0.0016
    3. ' though' (ID 996): effect=+0.0015
    4. ' however' (ID 2158): effect=+0.0015
    5. ' although' (ID 1936): effect=+0.0013

  Measurement statistics on 23 promoted tokens:
    Mean: 0.0056 (POSITIVE = feature promotes these)
    Median: 0.0037
    Std: 0.0190
    Range: [0.0001, 0.0210]

  Control: Mean=0.0001, Median=0.0000
```

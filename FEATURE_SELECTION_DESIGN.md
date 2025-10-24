# Feature Selection Design: Interpretable + Frequent Features

## Problem
Current interpretable feature selection ignores firing frequency, leading to:
- Features that fire rarely (13-40 positions out of 213,600)
- Unstable probability change estimates
- Extreme negative values (-37k) due to small sample sizes

## Current Strategy
```
Interpretable: top-K by label_confidence (ignoring activation)
Activation: top-K by max_activation (ignoring interpretability)
```

## Proposed Strategy: Composite Score

### Firing Rate Metric
First, compute a "firing rate" for each feature:
```python
firing_rate = (num_positions_above_P65) / (total_positions)
```

This measures: "What percentage of positions does this feature activate on?"

### Composite Selection Score
```python
# Option 1: Multiplicative with log scaling
score = label_confidence * log10(1 + 100 * firing_rate)

# Option 2: Weighted harmonic mean
score = 2 / (1/label_confidence + 1/(firing_rate_percentile/100))

# Option 3: Threshold + rank (RECOMMENDED)
# - Filter: confidence >= 0.7 AND firing_rate >= 0.001 (0.1%)
# - Rank by: confidence * sqrt(firing_rate_percentile)
```

### Recommendation: Option 3
```python
# Step 1: Filter for minimum quality
valid_features = features[
    (features.label_confidence >= 0.7) &
    (features.firing_rate >= 0.001)  # At least 0.1% of positions
]

# Step 2: Compute firing rate percentile within layer
valid_features['firing_rate_pct'] = valid_features.groupby('layer')['firing_rate'].rank(pct=True)

# Step 3: Composite score
valid_features['composite_score'] = (
    valid_features['label_confidence'] *
    np.sqrt(valid_features['firing_rate_pct'])
)

# Step 4: Select top-K per layer
top_k_per_layer = valid_features.groupby('layer').apply(
    lambda g: g.nlargest(K, 'composite_score')
)
```

### Why This Works
- **Confidence term**: Ensures we get interpretable features (0.7+ threshold)
- **Firing rate threshold**: Eliminates ultra-rare features (ensures ≥200 positions with 200k total)
- **Sqrt of percentile**: Balances the two - doesn't overweight firing rate
- **Example scores**:
  - conf=0.95, firing_rate=1% (P90) → score ≈ 0.90
  - conf=0.85, firing_rate=5% (P99) → score ≈ 0.85
  - conf=0.95, firing_rate=0.01% (P10) → score ≈ 0.30

## Diagnostic Logging for Extreme Changes

When `|mean_prob_change| > 10`, print detailed diagnostics:

```python
if abs(mean_prob_change) > 10:
    logger.warning(f"EXTREME PROBABILITY CHANGE DETECTED: {mean_prob_change:.2f}")
    logger.warning(f"Feature: {label}, Layer: {layer}, Positions: {num_positions}")
    logger.warning(f"")
    logger.warning(f"Top-5 affected tokens (showing baseline → intervention):")

    for i, token_id in enumerate(top_k_tokens[:5]):
        baseline_prob = baseline_probs[i]
        intervention_prob = intervention_probs[i]
        abs_change = intervention_prob - baseline_prob
        rel_change = (abs_change / baseline_prob) if baseline_prob > 0 else float('inf')

        logger.warning(
            f"  {i+1}. '{tokenizer.decode([token_id])}' (ID {token_id}): "
            f"baseline={baseline_prob:.6f} → interv={intervention_prob:.6f} "
            f"(abs Δ={abs_change:+.6f}, rel Δ={rel_change:+.2f})"
        )

    logger.warning(f"")
    logger.warning(f"Interpretation:")
    if mean_prob_change < 0:
        logger.warning(f"  → Feature PROMOTES these tokens (ablating it decreased their probability)")
        logger.warning(f"  → Extreme value due to: {'very small baseline probs' if baseline_prob < 0.001 else 'strong feature effect'}")
    else:
        logger.warning(f"  → Feature SUPPRESSES these tokens (ablating it increased their probability)")
    logger.warning(f"")
```

## Implementation Plan
1. Add firing rate computation in feature selection loop
2. Replace line 1071 with composite score selection
3. Add diagnostic logging in measurement loop (around line 634)
4. Update config to add `min_firing_rate` parameter

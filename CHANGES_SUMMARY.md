# Summary of Changes: Composite Feature Selection + Diagnostic Logging

## Motivation

Analysis of the previous results revealed:
1. **Interpretable features fired rarely** (13-500 positions vs 500-13k for activation-selected)
2. **Extreme negative probability changes** (-37k) caused by small sample sizes
3. **Strong correlation**: Features with >500 positions → 81% positive, <50 positions → 58% positive but extreme outliers

## Changes Made

### 1. Composite Feature Selection (lines 1125-1230)

**Previous Approach:**
- Selected top-K interpretable features by `label_confidence` only
- Ignored firing frequency entirely
- Result: Many features with <50 active positions

**New Approach:**
```python
# Step 1: Compute firing rates for all features
firing_rate = (positions_above_P65) / (total_positions)

# Step 2: Filter for minimum thresholds
valid_features = features[
    (confidence >= 0.65) &          # Ensure interpretability
    (firing_rate >= 0.0005)          # Ensure ≥0.05% firing rate
]

# Step 3: Composite score
firing_rate_pct = rank(firing_rate, within_layer)
composite_score = confidence * sqrt(firing_rate_pct)

# Step 4: Select top-K by composite score
```

**Benefits:**
- ✅ Balances interpretability AND frequency
- ✅ Ensures minimum 100+ active positions (0.05% of 200k = 100)
- ✅ sqrt() prevents firing rate from dominating confidence
- ✅ Within-layer percentile ensures fair comparison

**Example Scores:**
- Feature A: conf=0.95, fire=1.0% (P90) → score = 0.95 × √0.90 ≈ 0.90
- Feature B: conf=0.85, fire=5.0% (P99) → score = 0.85 × √0.99 ≈ 0.85
- Feature C: conf=0.95, fire=0.01% (P10) → score = 0.95 × √0.10 ≈ 0.30

### 2. Diagnostic Logging for Extreme Changes (lines 634-708)

**When:** Triggered when `|mean_probability_change| > 1.0` (100% relative change)

**What it shows:**
```
⚠️  EXTREME PROBABILITY CHANGE DETECTED
Feature 6789: mean relative change = -37.25
Active positions: 13, Total: 213600

Example from position [2, 45]:
  Feature activation: 4.5231
  Top-5 token probabilities (baseline → ablated):
    'passive' (ID 14523): 0.000123 → 0.000003 (Δ=-0.000120, rel=-37.25)
    'was' (ID 373): 0.000089 → 0.000002 (Δ=-0.000087, rel=-32.14)
    ...

Interpretation:
  → Feature PROMOTES these tokens (ablating decreased their probability)
  → Extreme value likely due to: SMALL SAMPLE SIZE (13 positions)
```

**Why this helps:**
1. Shows **actual probabilities** (not just relative changes)
2. Reveals when baseline probs are tiny (0.0001) → small absolute changes = huge relative changes
3. Flags **small sample sizes** as likely cause
4. Explains **direction**: negative = promoting, positive = suppressing

## Understanding Negative Changes

**Common confusion:** "Why is ablating it causing an increase?"

**Clarification:**
- **Negative change** = ablated_prob < baseline_prob
- This means: ablating the feature **DECREASED** the probability
- Therefore: the feature was **PROMOTING** those tokens

**Example:**
```
Baseline (feature active):    token 'passive' has P = 0.001
Ablated (feature removed):    token 'passive' has P = 0.0001
Relative change = (0.001 - 0.0001) / 0.001 = 0.9 = -90%  [NEGATIVE]
→ Feature promotes 'passive'
```

## Expected Outcomes

With the new composite selection:

1. **Higher firing rates** for interpretable features
   - Old: 13-40 positions (0.01-0.02%)
   - New: 100-2000 positions (0.05-1%)

2. **More stable estimates**
   - Reduced extreme outliers
   - Better sample sizes

3. **Still interpretable**
   - Minimum confidence 0.65
   - Balanced score rewards both quality and frequency

4. **Diagnostic output**
   - Clear warnings for extreme changes
   - Shows actual probabilities
   - Explains likely causes

## Testing

Run the analysis script to validate:
```bash
python analyze_noise_intervention.py \
  --config configs/default.yaml \
  --top_k_interpretability 5
```

Expected in logs:
- "Computing firing rates..." (new step)
- "Top 5 features by COMPOSITE SCORE..." (new selection)
- Features should show `fire=0.xx%` values ≥0.05%
- Extreme changes should show diagnostic warnings

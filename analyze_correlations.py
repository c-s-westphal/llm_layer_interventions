#!/usr/bin/env python3
"""
Analyze correlations between feature activation frequency and probability change metrics.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

def parse_nohup_log(log_path):
    """Extract per-feature metrics from nohup.out."""
    with open(log_path, 'r') as f:
        content = f.read()

    # Split by feature sections
    feature_sections = re.split(r'--- Layer (\d+), Feature (\d+)', content)

    results = []

    # Process in groups of 3: [text before, layer, feature_id, text with data]
    for i in range(1, len(feature_sections), 3):
        if i+2 >= len(feature_sections):
            break

        layer = int(feature_sections[i])
        feature_id = int(feature_sections[i+1])
        section_text = feature_sections[i+2]

        # Extract feature label if available
        label_match = re.search(r"\('([^']+)', conf=([\d.]+)\)", section_text[:200])
        if label_match:
            label = label_match.group(1)
            confidence = float(label_match.group(2))
            is_interpretable = True
        else:
            # Check for "max activation"
            max_act_match = re.search(r'\(max activation=([\d.]+)\)', section_text[:200])
            if max_act_match:
                label = f"activation_selected_{feature_id}"
                confidence = 0.0
                is_interpretable = False
            else:
                continue

        # Extract number of active positions
        positions_match = re.search(
            r'Positions with activation > ([\d.]+): (\d+) \(([\d.]+)%\)',
            section_text
        )
        if not positions_match:
            continue

        threshold = float(positions_match.group(1))
        num_positions = int(positions_match.group(2))
        pct_positions = float(positions_match.group(3))

        # Extract probability change statistics
        stats_match = re.search(
            r'Probability change statistics on (\d+) active positions:\s+' +
            r'[^\n]*Mean: ([-\d.]+)\s+' +
            r'[^\n]*Median: ([-\d.]+)\s+' +
            r'[^\n]*Std: ([\d.]+)\s+' +
            r'[^\n]*Range: \[([-\d.]+), ([-\d.]+)\]',
            section_text,
            re.MULTILINE
        )

        if not stats_match:
            continue

        mean_prob_change = float(stats_match.group(2))
        median_prob_change = float(stats_match.group(3))
        std_prob_change = float(stats_match.group(4))
        min_prob_change = float(stats_match.group(5))
        max_prob_change = float(stats_match.group(6))

        # Extract control statistics
        control_match = re.search(
            r'Control: Mean=([-\d.]+), Median=([-\d.]+)',
            section_text
        )
        if control_match:
            control_mean = float(control_match.group(1))
            control_median = float(control_match.group(2))
        else:
            control_mean = np.nan
            control_median = np.nan

        results.append({
            'layer': layer,
            'feature_id': feature_id,
            'label': label,
            'confidence': confidence,
            'is_interpretable': is_interpretable,
            'threshold': threshold,
            'num_positions': num_positions,
            'pct_positions': pct_positions,
            'mean_prob_change': mean_prob_change,
            'median_prob_change': median_prob_change,
            'std_prob_change': std_prob_change,
            'min_prob_change': min_prob_change,
            'max_prob_change': max_prob_change,
            'control_mean': control_mean,
            'control_median': control_median,
        })

    return pd.DataFrame(results)


def analyze_correlations(df):
    """Analyze correlations between positions and probability changes."""

    print("="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    print()

    # Overall statistics
    print("Overall Statistics:")
    print(f"  Total features analyzed: {len(df)}")
    print(f"  Interpretable features: {df['is_interpretable'].sum()}")
    print(f"  Activation-selected features: {(~df['is_interpretable']).sum()}")
    print()

    # Distribution of signs
    print("Sign Distribution of Mean Probability Changes:")
    positive = (df['mean_prob_change'] > 0).sum()
    negative = (df['mean_prob_change'] < 0).sum()
    zero = (df['mean_prob_change'] == 0).sum()
    print(f"  Positive: {positive} ({100*positive/len(df):.1f}%)")
    print(f"  Negative: {negative} ({100*negative/len(df):.1f}%)")
    print(f"  Zero: {zero} ({100*zero/len(df):.1f}%)")
    print()

    # Correlation: num_positions vs mean_prob_change
    corr_pos_mean = df['num_positions'].corr(df['mean_prob_change'])
    print(f"Correlation (num_positions vs mean_prob_change): {corr_pos_mean:.4f}")

    # Correlation: num_positions vs abs(mean_prob_change)
    df['abs_mean_prob_change'] = df['mean_prob_change'].abs()
    corr_pos_abs = df['num_positions'].corr(df['abs_mean_prob_change'])
    print(f"Correlation (num_positions vs |mean_prob_change|): {corr_pos_abs:.4f}")
    print()

    # Split by number of positions
    print("Mean Probability Change by Position Bins:")
    df['position_bin'] = pd.cut(df['num_positions'],
                                  bins=[0, 20, 50, 100, 500, 10000],
                                  labels=['1-20', '21-50', '51-100', '101-500', '500+'])

    for bin_name, group in df.groupby('position_bin', observed=True):
        mean_val = group['mean_prob_change'].mean()
        n = len(group)
        n_positive = (group['mean_prob_change'] > 0).sum()
        n_negative = (group['mean_prob_change'] < 0).sum()
        print(f"  {bin_name:10s}: mean={mean_val:8.4f}, n={n:3d}, pos={n_positive:3d}, neg={n_negative:3d}")
    print()

    # Compare interpretable vs activation-selected
    print("Interpretable vs Activation-Selected Features:")
    for is_interp, group in df.groupby('is_interpretable'):
        label = "Interpretable" if is_interp else "Activation"
        mean_pos = group['num_positions'].mean()
        mean_change = group['mean_prob_change'].mean()
        n_positive = (group['mean_prob_change'] > 0).sum()
        n_negative = (group['mean_prob_change'] < 0).sum()
        print(f"  {label:15s}: avg_positions={mean_pos:7.1f}, avg_change={mean_change:8.4f}, pos={n_positive:3d}, neg={n_negative:3d}")
    print()

    # Look at extreme negative values
    print("Features with Extreme Negative Mean Changes (< -1.0):")
    extreme_neg = df[df['mean_prob_change'] < -1.0].sort_values('mean_prob_change')
    if len(extreme_neg) > 0:
        for _, row in extreme_neg.head(10).iterrows():
            print(f"  Layer {row['layer']:2d}, Feature {row['feature_id']:5d} ({row['label']:30s}): "
                  f"mean={row['mean_prob_change']:8.2f}, positions={row['num_positions']:4d}")
    else:
        print("  (None)")
    print()

    # Look at features with very few positions and negative values
    print("Features with <50 Positions AND Negative Mean:")
    few_pos_neg = df[(df['num_positions'] < 50) & (df['mean_prob_change'] < 0)]
    print(f"  Count: {len(few_pos_neg)}")
    print(f"  Mean positions: {few_pos_neg['num_positions'].mean():.1f}")
    print(f"  Mean prob change: {few_pos_neg['mean_prob_change'].mean():.4f}")
    print()

    print("Features with >100 Positions AND Negative Mean:")
    many_pos_neg = df[(df['num_positions'] > 100) & (df['mean_prob_change'] < 0)]
    print(f"  Count: {len(many_pos_neg)}")
    print(f"  Mean positions: {many_pos_neg['num_positions'].mean():.1f}")
    print(f"  Mean prob change: {many_pos_neg['mean_prob_change'].mean():.4f}")
    print()

    # Save detailed results
    output_path = Path(__file__).parent / "outputs" / "correlation_analysis.csv"
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Detailed results saved to: {output_path}")
    print()

    return df


if __name__ == "__main__":
    log_path = Path(__file__).parent / "nohup.out"

    print("Parsing log file...")
    df = parse_nohup_log(log_path)
    print(f"Extracted {len(df)} features")
    print()

    analyze_correlations(df)

# GPT-2 Small SAE Interventions

A reproducible pipeline for performing feature interventions on GPT-2 Small using Sparse Autoencoders (SAEs) from the res-jb release.

## Overview

This project:
- Loads GPT-2 Small via TransformerLens and res-jb SAEs via SAE-Lens at `resid_post` hooks for all 12 layers
- Selects 5 interpretable features per layer from Neuronpedia
- Performs interventions by scaling feature activations by α (default 2.0)
- Measures effects on model output using cross-entropy loss, KL divergence, and logit deltas
- Generates CSV reports, plots, and a markdown summary

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Prepare feature selection**: Edit `data/neuronpedia_features_template.csv` with actual features from Neuronpedia (https://www.neuronpedia.org/gpt2-small/res-jb), then save as `data/neuronpedia_features.csv`

2. **Run the experiment**:
```bash
python -m src.run --config configs/default.yaml
```

Or with CLI arguments:
```bash
python -m src.run \
  --model gpt2-small \
  --sae-release gpt2-small-res-jb \
  --hook resid_post \
  --layers 0-11 \
  --alpha 2.0 \
  --corpus wikitext2 \
  --max_passages 200 \
  --max_len 256 \
  --feature_source csv \
  --seed 1234
```

3. **View results**: Check `outputs/report.md` for a summary, `outputs/csv/` for detailed metrics, and `outputs/plots/` for visualizations.

## Project Structure

```
project/
├── src/
│   ├── data.py          # Load WikiText-2 corpus
│   ├── features.py      # Feature selection from Neuronpedia or CSV
│   ├── model.py         # Load GPT-2 and SAEs
│   ├── intervene.py     # Intervention logic (alpha scaling)
│   ├── metrics.py       # Loss, KL, logit-delta calculations
│   ├── run.py           # Main experiment loop
│   ├── plots.py         # Plotting utilities
│   ├── report.py        # Generate markdown report
│   └── utils.py         # Logging, seeding, batching
├── outputs/             # Generated results
│   ├── csv/            # Metrics CSVs
│   ├── plots/          # Visualization PNGs
│   ├── snippets/       # Example text snippets
│   └── report.md       # Summary report
├── configs/
│   └── default.yaml    # Default configuration
├── data/
│   └── neuronpedia_features.csv  # Selected features
└── requirements.txt
```

## Configuration

Edit `configs/default.yaml` or pass CLI arguments. Key options:

- `alpha`: Scaling factor for interventions (default: 2.0)
- `alpha_sweep`: List of α values to test (default: [0, 0.5, 1, 2, 4])
- `live_percentile`: Threshold for "live" features (default: 90)
- `max_passages`: Number of text passages (default: 200)
- `batch_size`: Batch size (default: 16, auto-reduces on OOM)

## GPU Requirements

- Minimum: 16GB VRAM (RTX 4080)
- Recommended: 24GB VRAM (RTX 3090/4090, A5000)
- Optimal: 40GB+ VRAM (A100)

## Outputs

- `per_feature_metrics.csv`: Detailed metrics per (layer, feature, alpha)
- `per_layer_summary.csv`: Aggregated metrics by layer
- `layer_vs_dloss_alpha2.png`: Layer-wise effect visualization
- `report.md`: Experiment summary with charts and examples

## Citation

SAEs from: Bloom, J. (2024). Open Source Sparse Autoencoders for all Residual Stream Layers of GPT2-Small.

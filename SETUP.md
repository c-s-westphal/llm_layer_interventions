# Setup Guide

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Prepare feature selection**:

You need to provide a CSV file with features to analyze. The CSV should have these columns:
- `layer`: Layer index (0-11)
- `feature_id`: Feature ID from the SAE
- `label`: Human-readable feature description
- `label_confidence`: Confidence score for the label

**Option A: Use the template** (for testing):
```bash
cp data/neuronpedia_features_template.csv data/neuronpedia_features.csv
```

**Option B: Get real features from Neuronpedia**:

Visit https://www.neuronpedia.org/gpt2-small/res-jb and browse features by layer. For each layer 0-11, select the top 5 features by label confidence and add them to `data/neuronpedia_features.csv`.

Example format:
```csv
layer,feature_id,label,label_confidence
0,12453,articles about physics,0.89
0,8721,possessive pronouns,0.85
0,4332,numeric expressions,0.82
...
```

3. **Run the experiment**:

With default settings:
```bash
python -m src.run
```

With custom config:
```bash
python -m src.run --config configs/custom.yaml
```

With CLI arguments:
```bash
python -m src.run --alpha 3.0 --max-passages 100 --batch-size 8
```

4. **View results**:

Check `outputs/report.md` for a summary of results, including:
- Summary statistics
- Layer-wise trends
- Visualizations
- Example snippets

## Configuration Options

Edit `configs/default.yaml` or pass CLI arguments:

### Model Settings
- `model_name`: Model to use (default: "gpt2-small")
- `sae_release`: SAE release (default: "gpt2-small-res-jb")
- `hook`: Hook point (default: "resid_post")
- `layers`: Layers to analyze (default: "0-11")

### Intervention Settings
- `alpha`: Scaling factor (default: 2.0)
- `alpha_sweep`: List of alpha values to test (default: [0, 0.5, 1, 2, 4])
- `live_percentile`: Threshold percentile (default: 90)

### Corpus Settings
- `corpus_name`: Dataset name (default: "wikitext2")
- `max_passages`: Number of passages (default: 200)
- `max_len`: Max token length (default: 256)
- `calibration_passages`: Passages for calibration (default: 50)

### Computation Settings
- `device`: Device to use (default: "auto")
- `batch_size`: Batch size (default: 16)
- `seed`: Random seed (default: 1234)

## GPU Requirements

- **Minimum**: 16GB VRAM (e.g., RTX 4080)
- **Recommended**: 24GB VRAM (e.g., RTX 3090/4090, A5000)
- **Optimal**: 40GB+ VRAM (e.g., A100)

The pipeline includes automatic OOM handling and will reduce batch size if needed.

## Troubleshooting

### Out of Memory

If you get OOM errors:
1. Reduce `batch_size` (try 8, 4, or even 2)
2. Reduce `max_passages`
3. Reduce `max_len`

### Missing Features

If features CSV is not found:
```bash
# Check if file exists
ls data/neuronpedia_features.csv

# If not, copy template
cp data/neuronpedia_features_template.csv data/neuronpedia_features.csv
```

### Import Errors

If you get import errors:
```bash
# Make sure you're in the project root
pwd

# Run with python -m
python -m src.run
```

## Development

To modify the pipeline:

1. Edit relevant module in `src/`
2. Re-run experiment
3. Check logs in `outputs/experiment.log`

## Citation

If you use this pipeline, please cite:

- SAEs: Bloom, J. (2024). Open Source Sparse Autoencoders for all Residual Stream Layers of GPT2-Small.
- TransformerLens: Nanda, N. et al. TransformerLens.
- SAE-Lens: Bloom, J. et al. SAE-Lens.

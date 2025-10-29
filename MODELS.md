# Available Models for SAE Ablation Analysis

This document lists models with pretrained SAEs that can be analyzed using this codebase.

## Models Ready to Run

### 1. GPT-2 Small (Current)
- **Config**: `configs/default.yaml`
- **Model**: `gpt2-small`
- **SAE Release**: `gpt2-small-res-jb`
- **Layers**: 0-11 (12 layers)
- **Size**: 117M parameters
- **Status**: ✅ Tested and working

**Run with:**
```bash
python analyze_random_sampling.py --config configs/default.yaml
```

---

### 2. Gemma-Scope-2b (Recommended)
- **Config**: `configs/gemma_scope_2b.yaml`
- **Model**: `gemma-2b`
- **SAE Release**: `gemma-scope-2b-pt-res`
- **Layers**: 0-25 (26 layers)
- **Size**: 2B parameters
- **Hook**: `resid_pre` (residual stream)
- **Status**: ⚠️ Not yet tested

**Why recommended**: Most comprehensive layer coverage (26 layers), modern architecture

**Run with:**
```bash
python analyze_random_sampling.py --config configs/gemma_scope_2b.yaml --num_features 200 --position_fraction 0.25
```

**Expected runtime**: ~8-10 hours (more layers than GPT-2)

---

### 3. Gemma-2b (Joseph Bloom's Release)
- **Config**: `configs/gemma_2b.yaml`
- **Model**: `gemma-2b`
- **SAE Release**: `gemma-2b-res-jb`
- **Layers**: 0, 6, 10, 12, 17 (5 layers only)
- **Size**: 2B parameters
- **Status**: ⚠️ Not yet tested

**Limitation**: Only 5 layers available, not suitable for full cross-layer analysis

**Run with:**
```bash
python analyze_random_sampling.py --config configs/gemma_2b.yaml
```

---

### 4. Llama-3.1-8B (Limited)
- **Config**: `configs/llama_3.1_8b.yaml`
- **Model**: `meta-llama/Llama-3.1-8B`
- **SAE Release**: `temporal-sae-llama-3.1-8b`
- **Layers**: 15, 26 (only 2 layers)
- **Size**: 8B parameters
- **Status**: ⚠️ Not yet tested

**Limitation**: Only 2 layers available, requires Llama access

**Run with:**
```bash
python analyze_random_sampling.py --config configs/llama_3.1_8b.yaml
```

---

## Additional Models with SAEs (May require setup)

### Gemma-2b-IT (Instruction-Tuned)
- Release: `gemma-2b-it-res-jb`
- Similar to Gemma-2b but instruction-tuned

### Gemma-Scope-27b
- Release: `gemma-scope-27b-pt-res`
- Layers: 10, 22, 34 (3 layers)
- Size: 27B parameters
- ⚠️ Very large, may require multi-GPU

### DeepSeek-R1-Distill-Llama-8B
- Release: `deepseek-r1-distill-llama-8b-qresearch`
- Layer: 19 only (1 layer)
- Not suitable for cross-layer analysis

---

## How to Add a New Model

1. Create a config file in `configs/`:
```yaml
model_name: "your-model-name"
sae_release: "sae-release-name"
hook: "resid_pre"  # or "resid_post"
layers: "0-N"  # or list like [0, 5, 10]
batch_size: 16  # adjust based on model size
output_dir: "outputs_modelname"
```

2. Run the analysis:
```bash
python analyze_random_sampling.py --config configs/your_config.yaml
```

3. Check SAE availability at:
   - https://github.com/jbloomAus/SAELens/blob/main/sae_lens/pretrained_saes.yaml
   - https://huggingface.co/models?library=saelens

---

## Recommended Next Steps

1. **Gemma-Scope-2b**: Best for comprehensive cross-layer analysis (26 layers)
2. Compare GPT-2 Small vs Gemma-2b results
3. Test if architectural differences affect feature ablation patterns

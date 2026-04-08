# Router Intervention Experiments - Usage Guide

This guide explains how to use the `run_project_out` script for router intervention experiments on Mixtral-style MoE models.

## What does the script do?

The `run_project_out` script:
1. **Loads a Mixtral model** and evaluates baseline loss on text data
2. **Computes SVD-based interventions** by projecting out singular vectors from router weights
3. **Compares multiple variations**: SVD, orthogonal, random, zero, shuffle
4. **Tests multiple k values**: How many singular vectors to project out (e.g., k=1,2,4,8,16)
5. **Saves results** with loss and delta for each variant

## Prerequisites

Before running, ensure you have:

### 1. SVD Cache

Generate SVD vectors for router weights:

```bash
python -m src.experiments.router_interventions.compute_svd \
  --model_id mistralai/Mixtral-8x7B-v0.1 \
  --output_dir ./svd_cache \
  --layers 0-31
```

This creates files like:
- `svd_cache/mistralai_Mixtral_8x7B_v0.1_layer0_expert0.pkl`
- ... through expert7, for each layer

### 2. GPU Memory

- **Mixtral-8x7B** requires ~92GB in bfloat16
- Use `--target-layer-only-gpu` for single GPU with <92GB (recommended)
- This intelligently fills GPU with layers while guaranteeing target layer is on GPU

## Basic Usage

### Single k value

```bash
python -m src.experiments.router_interventions.run_project_out \
  --svd_dir ./svd_cache \
  --layer_idx 15 \
  --top-k 4 \
  --output_file results_layer15_k4.json
```

### Multiple k values

```bash
python -m src.experiments.router_interventions.run_project_out \
  --svd_dir ./svd_cache \
  --layer_idx 15 \
  --top-k "1,2,4,8,16,32,64,128" \
  --output_file results_layer15_multi_k.json
```

### With optimized GPU usage

```bash
python -m src.experiments.router_interventions.run_project_out \
  --svd_dir ./svd_cache \
  --layer_idx 15 \
  --top-k "1,2,4,8,16" \
  --target-layer-only-gpu \
  --num_samples 50 \
  --output_file results_layer15.json
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--svd_dir` | *required* | Path to SVD cache directory |
| `--layer_idx` | 0 | Layer index to intervene on (0-31 for Mixtral) |
| `--top-k` | "1" | Comma-separated k values (e.g., "1,2,4,8") |
| `--output_file` | `results_project_out.json` | Where to save results |
| `--num_samples` | 100 | Number of text samples to evaluate on |
| `--seq_len` | 512 | Sequence length for each sample |
| `--batch_size` | 4 | Batch size (reduce if OOM) |
| `--variations` | all | Comma-separated: `svd,orthogonal,random,zero,shuffle` |
| `--target-layer-only-gpu` | off | **Recommended**: Intelligently fill GPU with layers, guarantee target layer on GPU |
| `--dataset` | `wikitext` | Dataset: `wikitext` or `text` (with `--text-file`) |

## Understanding the Output

### Console Output

```
2026-02-07 15:50:00,123 - INFO - Loading tokenizer took 2.50 seconds
2026-02-07 15:50:15,456 - INFO - Loading model took 45.30 seconds
2026-02-07 15:50:20,789 - INFO - Baseline loss: 2.1945

2026-02-07 15:50:25,123 - INFO - Computing baseline: ZERO (all router weights = 0)
2026-02-07 15:50:30,456 - INFO -   → loss=6.5234, delta=+4.3289

2026-02-07 15:50:32,789 - INFO - Computing baseline: SHUFFLE (permute router rows)
2026-02-07 15:50:38,123 - INFO -   → loss=2.5123, delta=+0.3178

=== Running interventions with top_k=1 ===
2026-02-07 15:50:40,456 - INFO - Computing: SVD (k=1)
2026-02-07 15:50:45,789 - INFO -   → loss=2.1998, delta=+0.0053

=== Running interventions with top_k=4 ===
2026-02-07 15:50:50,123 - INFO - Computing: SVD (k=4)
2026-02-07 15:50:55,456 - INFO -   → loss=2.2045, delta=+0.0100

============================================================
TIMING SUMMARY
  Loading model: 45.30s (55.2%)
  Baseline evaluation: 5.20s (6.3%)
  Zero intervention: 4.80s (5.9%)
  Shuffle intervention: 5.30s (6.5%)
  Intervention svd (k=1): 5.20s (6.3%)
  ...
  TOTAL: 82.10s
============================================================
```

### JSON Output

```json
{
  "config": {
    "model_id": "mistralai/Mixtral-8x7B-v0.1",
    "layer_idx": 15,
    "top_k": [1, 2, 4, 8],
    ...
  },
  "baseline_loss": 2.1945,
  "by_k": {
    "1": {
      "svd": {"loss": 2.1998, "delta": 0.0053},
      "orthogonal": {"loss": 2.1950, "delta": 0.0005},
      "random": {"loss": 2.1947, "delta": 0.0002},
      "zero": {"loss": 6.5234, "delta": 4.3289},
      "shuffle": {"loss": 2.5123, "delta": 0.3178}
    },
    "2": { ... },
    ...
  }
}
```

**Interpreting results:**
- `delta > 0`: Intervention **increases** loss (hurts performance) → vector is important
- `delta ≈ 0`: Intervention has minimal effect → vector is less critical
- `delta < 0`: Intervention **decreases** loss (improves performance) → rare, interesting case

## VS Code Launch Configuration

The `.vscode/launch.json` file includes two pre-configured setups:

1. **Router: Project out** - Quick test with layer 0, k=1
2. **Router: Project out (multiple k)** - Full experiment with layer 15, k=1,4,16,64,128

To use:
1. Open VS Code in the repo root
2. Press F5 or go to Run and Debug
3. Select one of the "Router: Project out" configurations
4. Edit `args` in `launch.json` to customize (especially `--svd_dir`)

## Memory Management Strategies

### Strategy 1: Intelligent GPU Filling (Recommended)

```bash
python -m src.experiments.router_interventions.run_project_out \
  --target-layer-only-gpu \
  --svd_dir ./svd_cache \
  --layer_idx 15
```

**What it does:**
- Estimates how many layers fit on your GPU
- Loads embeddings, norm, lm_head, + target layer to GPU (priority)
- Fills remaining GPU space with layers 0, 1, 2, ... (sequential)
- Offloads remaining layers to CPU

**Example** (44GB GPU + Mixtral):
- ~12-15 layers on GPU (including layer 15)
- Layers 16-31 on CPU
- Uses ~40GB GPU memory

### Strategy 2: Full Auto (Multi-GPU or Large GPU)

```bash
python -m src.experiments.router_interventions.run_project_out \
  --svd_dir ./svd_cache \
  --layer_idx 15
```

Uses HuggingFace's `device_map="auto"` - good for multi-GPU setups.

### Strategy 3: Force Single Device (>92GB GPU required)

```bash
python -m src.experiments.router_interventions.run_project_out \
  --use-single-device \
  --svd_dir ./svd_cache \
  --layer_idx 15
```

**Warning**: Requires ~92GB VRAM for Mixtral-8x7B. Will OOM on smaller GPUs.

## Common Workflows

### Quick Test (5 minutes)

```bash
python -m src.experiments.router_interventions.run_project_out \
  --svd_dir ./svd_cache \
  --layer_idx 0 \
  --top-k 1 \
  --num_samples 10 \
  --target-layer-only-gpu
```

### Full Experiment (30-60 minutes)

```bash
python -m src.experiments.router_interventions.run_project_out \
  --svd_dir ./svd_cache \
  --layer_idx 15 \
  --top-k "1,2,4,8,16,32,64,128" \
  --num_samples 100 \
  --target-layer-only-gpu \
  --output_file results_layer15_full.json
```

### Sweep Across Layers

```bash
for layer in 0 5 10 15 20 25 30; do
  python -m src.experiments.router_interventions.run_project_out \
    --svd_dir ./svd_cache \
    --layer_idx $layer \
    --top-k "1,4,16" \
    --target-layer-only-gpu \
    --output_file results_layer${layer}.json
done
```

## Troubleshooting

### "CUDA out of memory"

1. **Add `--target-layer-only-gpu`** (recommended)
2. Reduce `--batch-size` (try 2 or 1)
3. Reduce `--seq-len` (try 256 or 128)
4. Reduce `--num_samples` (try 50)

### "No SVD vectors loaded"

- Check `--svd_dir` path exists
- Verify files like `mistralai_Mixtral_8x7B_v0.1_layer0_expert0.pkl` are present
- Ensure you ran `compute_svd` first

### "Gate weights are still on meta device"

- Use `--target-layer-only-gpu` (forces materialization)
- Or use `--use-single-device` if you have enough VRAM

### Slow performance

- Use `--target-layer-only-gpu` to load more layers on GPU
- Reduce `--num_samples` for faster iteration
- The script now includes detailed timing breakdowns to identify bottlenecks

## Advanced: Custom Text Dataset

```bash
python -m src.experiments.router_interventions.run_project_out \
  --svd_dir ./svd_cache \
  --layer_idx 15 \
  --dataset text \
  --text-file ./my_custom_texts.json \
  --top-k "1,4" \
  --target-layer-only-gpu
```

`my_custom_texts.json` format:
```json
[
  "First text sample here...",
  "Second text sample...",
  "Third sample..."
]
```

## Tips for Research

1. **Start small**: Test with `--num_samples 10` and `--top-k 1` first
2. **Compare layers**: Run experiments on early (0-5), middle (10-15), and late (25-31) layers
3. **Watch deltas**: Large positive delta means the SVD direction is important for routing
4. **Check timing**: The script reports timing breakdown - use it to optimize
5. **Save everything**: Each experiment creates a JSON with full config + results for reproducibility

## Getting Help

- Check `README.md` for prerequisites and setup
- Review `src/experiments/router_interventions/core/config.py` for all available options
- Use `python -m src.experiments.router_interventions.run_project_out --help`

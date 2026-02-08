# Quantization methods and how they affect runs

## What’s in the codebase

| Setup | Method | Typical weight memory (Mixtral ~47B params) | Used in |
|--------|--------|---------------------------------------------|--------|
| **Our runners (default)** | **bfloat16** (no quantization) | ~14 GB (2 bytes/param) | `project_out_runner`, `vector_intervention_runner` |
| **Gate-hook** | **8-bit** (BitsAndBytes LLM.int8()) | ~7 GB (1 byte/param) | gate-hook `forward.py` |
| **Optional** | **4-bit** (e.g. NF4 + double quant) | ~3.5–4 GB | Not in runners by default; you can add |

So the main **difference in quantization methods** is:

- **bfloat16**: Full 16-bit weights. No quantization; highest accuracy, most GPU memory.
- **8-bit**: Weights stored as int8, dequantized on the fly. About half the weight memory; accuracy usually very close to bfloat16.
- **4-bit**: Weights in 4-bit (e.g. NF4). Smallest memory, some accuracy loss (especially in sensitive parts like router logits).

---

## How each option affects runs

### 1. **Memory (GPU VRAM)**

- **bfloat16**: ~14 GB just for weights + activations. With long sequences or big batches, activation memory (and attention) dominates; quantization doesn’t reduce that.
- **8-bit**: Weights ~half → frees several GB. Lets you load the model on a smaller GPU or leave more room for **batch size × sequence length** (e.g. 2000×32).
- **4-bit**: Weights ~quarter → even more free VRAM. Helps most when you’re limited by **weight** size; once you’re activation-bound, gains are smaller.

So: **quantization mainly reduces weight memory**. If you still OOM at batch size 4 with 512-token sequences, the bottleneck is **attention/activations** (O(batch × seq_len²) and O(batch × seq_len × hidden)), not weights. That’s why we added **short-sequence titles mode** (e.g. `wikitext_titles`, seq_len=32, batch_size=2000) to match gate-hook.

### 2. **Speed**

- **bfloat16**: Fastest per forward if you have enough memory (no dequant).
- **8-bit**: Slightly slower per step due to dequantization; often similar or acceptable. Gate-hook runs 2000×32 with 8-bit.
- **4-bit**: Often **faster** in practice (less memory bandwidth, better cache use) despite more dequant work.

### 3. **Accuracy / behavior**

- **bfloat16**: Reference; no quantization error.
- **8-bit**: Usually negligible impact on loss and router behavior; gate-hook uses it for logging router logits and activations.
- **4-bit**: Can change router logits and expert choices a bit; loss may differ slightly. If you care about exact gate-hook–style router analysis, 8-bit is safer than 4-bit.

### 4. **Compatibility**

- **bfloat16**: Works everywhere; model is loaded then moved to CUDA if available.
- **8-bit**: Needs `bitsandbytes`. Model is loaded then moved to CUDA if available (same as gate-hook style).
- **4-bit**: Same lib; some ops may run in different code paths; keep an eye on NaNs or odd values in router logits if you use it.

---

## Summary table

| Method   | Weight memory | Activation memory | Speed (typical) | Accuracy vs bf16 |
|----------|----------------|--------------------|-----------------|--------------------|
| bfloat16 | 100%           | Same               | Baseline        | Reference          |
| 8-bit    | ~50%           | Same               | Slightly slower | Very close         |
| 4-bit    | ~25%           | Same               | Often faster    | Some loss          |

**Bottom line:**  
- Use **8-bit** to match gate-hook and roughly halve weight memory with minimal impact on results.  
- Use **4-bit** only if you need maximum memory savings and accept possible small changes in loss/router behavior.  
- To run **large batches** (e.g. 2000 titles), combine quantization with **short sequences** (e.g. `--dataset wikitext_titles --seq-len 32`); otherwise attention/activation memory will still dominate.

# Why gate-hook branch runs much faster than project-out experiments

## Summary

**gate-hook** runs **one forward pass per batch** and only **observes** router behavior.  
**Your project-out** runs **many forward passes** (one per intervention × per k) because you **modify weights and re-evaluate** each time. That’s why gate-hook is much faster.

**Why gate-hook can use batch 2000 and we can’t even use batch 4 (with quant):**  
Gate-hook uses **max_length=32** (Wikipedia *titles* — short strings). We use **seq_len=512** for LM loss. Activation memory scales roughly as **seq_len²** (attention). So 512² ≈ 256× more than 32² per sample. With quantization the model already uses ~44 GB; there’s almost no room for activations at 512. **Fix:** use **--seq-len 64** or **128** with quantization so batch_size 4 (or more) fits. We auto-cap seq_len to 128 when quantization is on unless you override with **--seq-len**.

---

## 1. What each branch does

### gate-hook (sagearc/GaleMoE, branch `gate-hook`)

- **Goal:** Record router logits and expert activations **during a normal forward**.
- **How:** Monkey-patch `MixtralSparseMoeBlock.forward` and expert MLP forwards so they run as usual but **write router_logits and activations to a cache**.
- **Forward:** Single call per batch:
  ```python
  output = model(input_ids=..., attention_mask=..., output_router_logits=True)
  ```
- **No weight changes.** The model runs once; they just capture what the gate/router does.
- **Batching:** Can use large batches (e.g. 2000 × 32 tokens in `forward.py`) because there’s only one forward and no extra memory for multiple weight sets.

So: **1 forward per batch**, no intervention, just logging.

### Your project-out (this repo)

- **Goal:** Measure **loss under interventions** (project-out SVD, orthogonal, random, zero, shuffle) for different `top_k`.
- **How:** For each intervention and each k:
  1. **Modify** the router/gate weights (e.g. project out directions, zero, shuffle).
  2. **Run a full forward** to compute loss with those weights.
  3. **Restore** original weights.
- **Forward:** Many calls per batch set:
  - 1× baseline
  - 1× zero (if in variations)
  - 1× shuffle (if in variations)
  - For each k: 1× per project-out variant (svd, orthogonal, random)
- So with e.g. 5 variants and 5 k values: **1 + 2 + (3 × 5) = 18 full forward passes** over the same data (and more if you add variants).

So: **N forwards per “experiment”**, where N = 1 baseline + 2 baselines (zero/shuffle) + (number of project-out variants × number of k values).

---

## 2. Why gate-hook is faster (by design)

| Aspect | gate-hook | Your project-out |
|--------|-----------|-------------------|
| **Forwards per “run”** | 1 per batch | Many (1 baseline + zero + shuffle + variants × k) |
| **Weight changes** | None | Yes: modify → evaluate → restore, repeated |
| **What is measured** | Router logits / activations | Loss under each intervention |
| **Batch size** | Can be large (e.g. 2000×32) | Often 1 with 8-bit to avoid OOM |
| **Core cost** | One forward | Forward × (1 + 2 + n_variants × n_k) |

So gate-hook is faster mainly because:

1. **Single forward per batch** – no repeated evaluation with different weights.
2. **No intervention loop** – they don’t re-run the model for each “variant” or k.
3. **Observation only** – patches only add logging; computation is the same as one normal forward.
4. **Larger batches** – no need for batch_size=1 to survive OOM under many forwards.

Your code is slower because it’s answering a different question: “What is the loss if we **change** the router weights in these ways?” That question inherently requires **one full forward per (intervention, k)**.

---

## 3. Technical details from gate-hook

- **Patched modules:**  
  `MixtralSparseMoeBlock.forward` and `MixtralBlockSparseTop2MLP.forward` in `src/forward/patched_blocks.py`.
- **Router logits:**  
  Their patched `MixtralSparseMoeBlock.forward` returns `(final_hidden_states, router_logits)`. They get router logits from that single forward (and can also use `output_router_logits=True` on the model).
- **No `RouterManager`:**  
  They never replace `gate.weight.data`; they only read/cache activations and logits during the standard forward.

So their “gate hook” is literally: run the normal forward once and record gate/router outputs. Your “project-out” is: change gate weights, run forward, repeat for each intervention and k.

---

## 4. Can you make project-out faster without changing the goal?

You can’t get “one forward” like gate-hook and still measure loss under many different weight interventions. But you can reduce work:

- **Fewer k values** – e.g. `--top-k "1,4,16"` instead of `"1,4,16,64,128"`.
- **Fewer variations** – e.g. drop `orthogonal` or `random` if not needed.
- **Fewer samples** – `--num_samples` (and possibly shorter `--seq-len`) to get quicker runs.
- **Larger batch when possible** – e.g. `--batch-size 2` if memory allows (reduces number of forward passes per evaluation).
- **Skip zero/shuffle** if you only care about project-out variants.

So: gate-hook is faster because it does a single forward and only observes; your code is slower because it correctly does many forwards to measure loss under each intervention. The difference is in the experiment design, not a bug in your implementation.

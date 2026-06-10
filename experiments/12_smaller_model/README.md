# 12_smaller_model — Observations

## Goal
A **capacity ablation of experiment 09**. 09 swept seeds 42–51 of the 06/07/08
trainings at the baseline shape `n_layers=4, n_heads=4`. 12 reruns the *identical*
sweep — same `random_frame` depth-3 dataset, same 30 epochs, same seeds — but with
a deliberately **smaller model: `n_layers=3, n_heads=2`**. Two questions:

1. **Capacity:** does the smaller model still reach ~1.0 test exact-match on
   `random_frame` depth-3 (where the L4/H4 models saturated)?
2. **Heatmap consistency:** are the cross-seed attention heatmaps as robust at
   L3/H2 as at L4/H4? Fewer heads also makes head-matching cheaper and cleaner
   (`2! = 2` pairings per layer vs `4! = 24`).

> **Scope note.** Like 09, this experiment covers **training only** — producing the
> checkpoints. Epochs stay at **30** (the 09 baseline) so heatmaps remain directly
> comparable. Grokking / held-out-depth is out of scope (that lives in exp 10).

## Setup
- **Datasets/architectures:** reuses the existing 06/07/08 scripts unchanged — one
  source of truth, **no local runner copies** (06/07/08 already expose `--n-layers`
  and `--n-heads`, so unlike exp 10 we don't need to fork them):
  | arch key | reused script | model |
  |---|---|---|
  | `vaswani` | `06_.../run.py` | Vaswani encoder-decoder, sinusoidal PE |
  | `vaswani_rope` | `07_.../run.py` | Vaswani encoder-decoder, RoPE on Q/K |
  | `gpt2_rope` | `08_.../run.py` | GPT2-style decoder-only, RoPE |
- **Dataset:** `kylelovesllms/hi_hf_frames_d3_random_100` (`random_frame`, depth 3).
- **Shape:** **`n_layers=3, n_heads=2`** (vs 09's 4/4). Width unchanged
  (`d_model=128, d_ff=512`), so this shrinks depth and head-count, not hidden size.
- **Seeds:** 42–51 (10). **Epochs:** 30. All other knobs (tokenizer, max-length,
  optimizer, eval) stay at each script's own defaults.
- **Checkpoints:** local only (`--no-push`), under
  `results/<arch>/L3_H2/seed_<S>/best/`. The `L<l>_H<h>` path/run-name segment keeps
  12's outputs and wandb runs from ever colliding with 09's (or with a different
  shape, if you sweep `--n-layers`/`--n-heads`).

`run.py` (copy `run_proposal.py` → `run.py`) is a thin launcher that shells out to
the 06/07/08 scripts with a different `--seed` each and `--n-layers 3 --n-heads 2`;
it never redefines model or training logic.

## How to run

**Default — packed (3 jobs, one per architecture, all 10 seeds share one GPU):**
```bash
sbatch slurm/run_gpu.sbatch experiments/12_smaller_model/run.py --arch vaswani
sbatch slurm/run_gpu.sbatch experiments/12_smaller_model/run.py --arch vaswani_rope
sbatch slurm/run_gpu.sbatch experiments/12_smaller_model/run.py --arch gpt2_rope
```

**Alternative — array (1 GPU per run, 30 tasks):**
```bash
sbatch --array=0-29 slurm/run_gpu.sbatch experiments/12_smaller_model/run.py
# task -> (arch, seed):  arch = ARCHS[task // 10], seed = 42 + task % 10
```

**Local smoke test (2 seeds, 1 epoch, no wandb / no push):**
```bash
uv run python experiments/12_smaller_model/run.py \
    --arch vaswani --seeds 42-43 --epochs 1 --max-parallel 2 --no-wandb
```

Per-run output goes to `logs/12_smaller_model_<arch>_L3_H2_seed_<S>.log`; wandb runs
are named `12_smaller_model_<arch>_L3_H2_seed_<S>` (disable with `--no-wandb`).

## Viewing all runs together in wandb
The launcher sets two env vars per subprocess (honored automatically by
`wandb.init` / the HF Trainer's wandb integration — **no edits to 06/07/08**):
- `WANDB_RUN_GROUP = 12_smaller_model_<arch>_L3_H2` — the 10 seeds of each arch
  share a **group**, so the UI shows a mean curve + min/max band across seeds.
- `WANDB_TAGS = 12_smaller_model,<arch>,L3_H2,seed_<S>` — filter all runs at once
  with the `12_smaller_model` tag, or compare against 09 by overlaying groups.

## Results
_(Sweep complete at **two** shrunk shapes — the planned **L3/H2** and an even smaller
**L2/H2** — each 3 archs × 10 seeds (42–51), 30 epochs, `random_frame` depth-3.
Cross-seed heatmaps were also computed, so the "deferred" step below is done.)_

**1. Capacity — encoder-decoders shrink gracefully; the decoder-only does not.**
Final held-out-frame **test exact-match** (mean over 10 seeds, min–max):

| arch | L3/H2 | L2/H2 |
|---|---|---|
| `vaswani` | **0.9998** (0.998–1.0) | **0.9994** (0.997–1.0) |
| `vaswani_rope` | **0.9989** (0.993–1.0) | **0.9998** (0.998–1.0) |
| `gpt2_rope` | 0.944 (0.718–1.0) | **0.703** (0.408–0.994) |

- Both **Vaswani encoder-decoders stay saturated (~1.0)** even at L2/H2 — the task
  does not need the L4/H4 capacity of 09; depth 2–3 and 2 heads suffice.
- **`gpt2_rope` (decoder-only) is capacity- and seed-fragile**: it drops to mean
  0.94 at L3/H2 and **0.70 at L2/H2**, with huge seed spread (one seed 0.41, another
  0.99). The single concatenated-stream causal-LM setup needs more depth/heads than
  the encoder-decoders to reach the same accuracy.

**2. Heatmap consistency — smaller ⇒ somewhat less seed-consistent.** Mean
order-invariant matched cosine vs the seed-42 reference (compare to 09's L4/H4):

| arch | L4/H4 (exp 09) | L3/H2 | L2/H2 |
|---|---|---|---|
| `vaswani` | 0.835 | 0.790 | 0.749 |
| `vaswani_rope` | 0.752 | 0.703 | 0.673 |
| `gpt2_rope` | 0.656 | 0.594 | 0.660 |

- For both Vaswani variants, **consistency declines monotonically as the model
  shrinks** (fewer heads/layers → fewer redundant equivalent solutions to average
  over, and each head carries more of the computation, so seeds diverge more).
- `gpt2_rope` is the exception (L2/H2 ≈ L4/H4), but its heatmaps are less meaningful
  there since accuracy itself is poor/seed-dependent (0.70 mean).

Numbers per (family, layer, seed) are in `results/<arch>/L<l>_H2/heatmap_consistency.csv`;
grids under `figures/<arch>_layer_<l>/heatmaps/depth<D>-ex<N>/`.

## Notes
-

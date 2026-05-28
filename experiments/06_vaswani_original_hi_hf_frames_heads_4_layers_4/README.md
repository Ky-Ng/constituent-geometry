# 06 — Vaswani (original) on frame-based HI → HF, `n_heads=4, n_layers=4`, random-frame split

First training run against the **frame-based** dataset produced by experiment
05. Uses the original Vaswani encoder-decoder (`modeling_vaswani.py`) at the
paper-ish width (`d_model=128, d_ff=512`) and the requested `n_heads=4,
n_layers=4` shape.

## Why

Experiments 02 – 04 used the legacy exhaustive enumeration (`hi_hf_dataset_depth_3`).
The new sub-categorized grammar makes that enumeration infeasible (~3.3B
sentences at depth 3 — see [experiment 05](../05_build_dataset_with_frames/README.md)),
so experiment 05 switched to per-frame sampling and frame-level splits. This
experiment is the first training run on that new data.

The split policy used here is **`random_frame`**: train and val/test share no
parse skeletons, but the depth distribution is matched across splits. So
val/test exact-match measures generalization to **unseen frames at the same
depths** — not recursion holdout (that's the `held_out_depth` variant, a
separate experiment).

## Model

`d_model=128, n_heads=4, n_layers=4, d_ff=512, dropout=0.1`. `head_dim = 128 / 4 = 32`.

Same `n_heads`/`n_layers` shape as experiment 04, but ~paper width — 04 had
shrunk width as a capacity probe and is not directly comparable on params.

## Data

Default dataset: `kylelovesllms/hi_hf_frames_d3_random_100` (HF Hub).
Override with `--dataset <repo-id>` to point at any other dataset built by
experiment 05's `random_frame` policy. The dataset is expected to expose
three splits: `train`, `validation`, `test`.

Tokenizer: `data/tokenizer` (built by `src/architecture/build_tokenizer.py`;
see top-level README if it doesn't exist locally yet).

## Tracking

- wandb project: `constituent-geometry`
- wandb run name: `06_vaswani_original_hi_hf_frames_heads_4_layers_4_random_depth_3`

## Setup

```bash
# Local smoke test (1 epoch, no wandb)
uv run python experiments/06_vaswani_original_hi_hf_frames_heads_4_layers_4_random/run.py \
    --epochs 1 --no-wandb

# Cluster (high-priority queue, single GPU, full 30-epoch run)
sbatch slurm/run_gpu.sbatch \
    experiments/06_vaswani_original_hi_hf_frames_heads_4_layers_4_random/run.py

# Point at a different frame dataset (e.g. depth-2 random)
sbatch slurm/run_gpu.sbatch \
    experiments/06_vaswani_original_hi_hf_frames_heads_4_layers_4_random/run.py \
    --dataset kylelovesllms/hi_hf_frames_d2_random_100 \
    --run-name 06_vaswani_original_hi_hf_frames_heads_4_layers_4_random_depth_2
```

## Results

_TODO: fill in after running (best val exact-match, test exact-match on
held-out frames, loss curve, params, wall-clock)._

## Observations

_TODO._

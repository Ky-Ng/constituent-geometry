# 04 — Depth-holdout Vaswani HI → HF (n_heads=4, n_layers=4)

Capacity control for **experiment 03**. Same depth-holdout task and split, same
small *width* (`d_model=64, d_ff=256`), but the *shape* is restored to
`n_heads=4, n_layers=4`.

## Why

Experiment 03 (`n_heads=1, n_layers=2`) underfit: in-dist val exact-match stayed
~0 and loss plateaued at ~0.74 after 30 epochs. Before shrinking further we test
the obvious suspect — expressivity. A single attention head can form only one
attention pattern per position, which may be too weak for a structural reorder.

Hypothesis test:
- **04 learns in-dist (exact-match → ~1) and generalizes to depth-2** ⇒ 03's
  bottleneck was head/layer count. Next: walk *down* (4 → 2 → 1 heads/layers) to
  find the smallest model that still solves the task — the interpretability target.
- **04 still won't learn in-dist** ⇒ not capacity; debug lr / schedule / data /
  label handling instead.

The task, split, methodology, and `--max-eval` behavior are identical to 03 — see
`experiments/03_holdout_depth_2_vaswani_hi_hf/README.md`.

## Model

`d_model=64, n_heads=4, n_layers=4, d_ff=256` → **467,904 trainable params**
(≈2× experiment 03; the increase is the 4 vs 2 layers — `n_heads` does not change
the count). `head_dim = 64 / 4 = 16`.

## Setup

```bash
# local (capped OOD eval, ~1 min on MPS)
uv run python experiments/04_holdout_depth_2_vaswani_hi_hf_heads_4_layers_4/run.py --epochs 30 --max-eval 2000
# cluster (full 98k OOD eval)
sbatch slurm/run_gpu.sbatch experiments/04_holdout_depth_2_vaswani_hi_hf_heads_4_layers_4/run.py --epochs 30
```

## Results

_TODO: fill in after running (in-dist val exact-match & loss curve, OOD depth-2
exact-match, and the verdict on the capacity hypothesis)._

## Observations

_TODO._

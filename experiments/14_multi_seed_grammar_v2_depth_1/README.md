# Experiment 14 — Multi-seed sweep on v2 grammar depth-1 dataset

## Goal

Replicate experiment 09 (multi-seed consistency check) on the new v2 grammar dataset
(`kylelovesllms/hi-hf-v2-frames-d1-k100`), which adds relative clauses and adjuncts
(from experiment 13) at depth 1. The primary question: do all three architectures still
saturate test exact-match, and are cross-seed heatmaps consistent?

## Setup

- **Dataset:** `kylelovesllms/hi-hf-v2-frames-d1-k100` (v2 grammar, depth 1, k=100)
- **Architectures:** vaswani, vaswani_rope, gpt2_rope (same as 06/07/08)
- **Seeds:** 42–51 (10 seeds, same as 09)
- **Epochs:** 30
- **Model shape:** 4 heads, 4 layers (unchanged from 09)

## How to run

Copy `run_proposal.py` to `run.py`, then:

```bash
# Packed mode (one job per arch, 10 seeds share one GPU)
sbatch slurm/run_gpu.sbatch experiments/14_multi_seed_grammar_v2_depth_1/run.py --arch vaswani
sbatch slurm/run_gpu.sbatch experiments/14_multi_seed_grammar_v2_depth_1/run.py --arch vaswani_rope
sbatch slurm/run_gpu.sbatch experiments/14_multi_seed_grammar_v2_depth_1/run.py --arch gpt2_rope

# Array mode (one GPU per (arch, seed) pair)
sbatch --array=0-29 slurm/run_gpu.sbatch experiments/14_multi_seed_grammar_v2_depth_1/run.py

# Local smoke test
uv run python experiments/14_multi_seed_grammar_v2_depth_1/run.py \
    --arch vaswani --seeds 42-43 --epochs 1 --max-parallel 2 --no-wandb
```

## wandb

- **Group** (per arch): `14_multi_seed_grammar_v2_depth_1_{arch}`
- **Tags** (per run): `14_multi_seed_grammar_v2_depth_1`, `{arch}`, `seed_{N}`

## Results

**Status: jobs running (30 epochs), numbers below are interim (~epoch 1)**

All three architectures saturate test exact-match within the **first epoch** on the depth-1
v2 grammar — much faster than experiment 09 (which needed the full 30 epochs on the depth-3
v1 dataset). Depth-1 appears substantially easier.

| Arch | Mean EM | Min EM | Max EM |
|---|---|---|---|
| vaswani | 0.9993 | 0.9957 | 1.0000 |
| vaswani_rope | 0.9998 | 0.9990 | 1.0000 |
| gpt2_rope | 0.9992 | 0.9970 | 1.0000 |

All seeds across all architectures are tightly clustered near 1.0 with no outliers,
suggesting depth-1 task difficulty is well within the capacity of all three models.

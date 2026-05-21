# 03 — Depth-holdout Vaswani HI → HF (shrunk: n_heads=1, n_layers=2)

Follow-up to experiment 02. Where 02 used a random split (train/val share
structure, so its ~0.003 loss can't distinguish algorithm from interpolation),
this experiment turns the eval into a **distribution shift along recursion depth**
and shrinks the model toward an interpretability-friendly size.

## Design

| set | depths | size | role |
|-----|--------|------|------|
| train | {0, 1} | ~5,875 | training |
| in-dist val | {0, 1} | ~653 | model selection / early stop (`metric_for_best_model="exact_match"`) |
| OOD test | {2} | 98,304 (cap with `--max-eval`) | final generalization number, run **once** at the end |

Depth 0 has no embedding; depth 1 shows the recursive CP step exactly once. If the
model reorders held-out **depth-2** sentences correctly, it must have learned to
*apply the rule recursively* — the result that would justify induction-head-style
attention analysis. The depth split needs no data regen: it reuses the `depth`
column already in `data/hi_hf_dataset_depth_3` (concatenate all splits, re-filter).

**Methodology:** we select the best checkpoint on the *in-distribution* val set and
report depth-2 **once**, never tuning on it.

## Model

`d_model=64, n_heads=1, n_layers=2, d_ff=256` → **234,432 trainable params**
(shared/tied embedding 960; encoder 99,968; decoder 133,504; `lm_head` tied → 0;
sinusoidal PE is a buffer, not a parameter). `n_heads` does not affect the count —
it changes head expressivity, not parameters.

## Setup

```bash
# local (cap the OOD eval so the no-KV-cache generation finishes in ~1 min on MPS)
uv run python experiments/03_holdout_depth_2_vaswani_hi_hf/run.py --epochs 30 --max-eval 2000
# cluster (full 98k OOD eval)
sbatch slurm/run_gpu.sbatch experiments/03_holdout_depth_2_vaswani_hi_hf/run.py --epochs 30
```

`--max-eval N` caps the depth-2 eval; the uncapped 98k set takes ~1h on MPS because
`.generate()` has no KV cache (recomputes the full prefix each step).

## Results (30 epochs, `--max-eval 2000`)

| metric | value |
|--------|-------|
| in-dist val exact-match | **~0** throughout |
| in-dist val loss | plateaued **~0.74** (barely moved over the last ~6 epochs) |
| OOD depth-2 exact-match | 0.0 |
| OOD depth-2 loss | ~2.04 |

## Observations

**The model underfit — it failed to learn even the *training* distribution.** A
solved version of this near-deterministic task should reach loss ~0.01–0.05; this
run plateaued at ~0.74 with exact-match ~0. So this is the opposite of the
memorization worry: not too easy, but not learned at all (in-dist).

**Leading suspect: too little expressivity.** With `n_heads=1` each layer can form
only one attention pattern per position, plausibly too weak for a structural
reorder that must track multiple relations at once. This motivated **experiment
04**, which keeps the same small width (`d_model=64, d_ff=256`) but restores
`n_heads=4, n_layers=4` to test whether the bottleneck was head/layer count. If 04
learns in-dist, walk *down* from there to find the smallest model that still solves
the task; if 04 still won't learn, the bug is elsewhere (lr/schedule/data).

### Confirmed non-issues
- The `[transformers] There were missing keys ...` warning for the embed/`lm_head`
  weights is **cosmetic**: only `model.shared.weight` is saved (the rest are tied),
  and after reload all four point to the same tensor with sensible values.

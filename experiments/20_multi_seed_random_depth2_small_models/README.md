# 20_multi_seed_random_depth2_small_models — Observations

## Goal
Run a **shrunk, single-epoch variant of experiment 18**. Exp 18 ran the
grokking-probe machinery (3 archs × 5 seeds, no early-stop, constant LR, wd=1.0)
on the v2 grammar's random-frame depth-2 split with the runners' *default*
4-layer / 4-head models over a 6-epoch horizon. 20 keeps the **same dataset,
archs, seeds, tokenizers, wd/LR and eval cadence**, but trains **smaller** models
for **one** epoch and asks how capacity (layers × heads) trades off on this
in-distribution split.

This is a quick capacity probe, **not** a grokking horizon: "grok" behavior (a
held-out transition long past convergence) is not expected in a single epoch. We
keep 18's no-early-stop / constant-LR / wd=1.0 config only so the comparison to 18
stays apples-to-apples.

## Capacity sweep
Four (layers, heads) configs per arch (d_model/d_ff/dropout unchanged at
128 / 512 / 0.1; heads of 1 and 2 both divide d_model=128):

| config | layers | heads |
|---|---|---|
| `L2H2` | 2 | 2 |
| `L2H1` | 2 | 1 |
| `L3H2` | 3 | 2 |
| `L3H1` | 3 | 1 |

Full sweep = **3 archs × 4 configs × 5 seeds = 60 runs.**

## Dataset (unchanged from exp 18)
`kylelovesllms/hi-hf-v2-frames-1.3M-d2-k2-random`

| split | rows | content |
|---|---|---|
| train | **2,229,638** | depth-{0,1,2} frames, random 80% |
| validation | 278,704 | depth-{0,1,2}, frame-disjoint 10% |
| test | 278,706 | depth-{0,1,2}, frame-disjoint 10% |

Random 80/10/10 over `frame_id`; val/test are unseen *frames* of the *same*
depths, so the gap is lexical/frame-level, **not** structural. At batch 64 the
train split is **~34,838 steps/epoch**, so one epoch ≈ 34,838 steps.

## Setup
- **Architectures:** drive the local runner copies under `runners/` —
  byte-identical to exp 18's (themselves 06/07/08 + `--weight-decay` /
  `--lr-scheduler` and the `--max-eval-samples` cap, **no early-stop**):
  | arch key | local runner | based on | model |
  |---|---|---|---|
  | `vaswani` | `runners/vaswani_run.py` | 06 | Vaswani enc-dec, sinusoidal PE |
  | `vaswani_rope` | `runners/vaswani_rope_run.py` | 07 | Vaswani enc-dec, RoPE on Q/K |
  | `gpt2_rope` | `runners/gpt2_rope_run.py` | 08 | GPT2-style decoder-only, RoPE |

  **No runner edits were needed** — all three already accept `--n-layers` /
  `--n-heads` (exp 18 left them at the 4/4 default) and already run a full-set
  `val_full` pass after training (see Evaluation). The launcher only threads the
  per-config layer/head counts through.
- **Tokenizers:** the **v2** tokenizers under `artifacts/` (copied from exp 18) —
  the **137-token** seq2seq tokenizer for `vaswani`/`vaswani_rope`, and the
  **138-token** `<sot>`-augmented one for `gpt2_rope`.
- **Seeds:** 42–46 (5). **Epochs:** default **1** (vs 18's 6).
- **Weight decay:** default **1.0** (matches 18). **LR schedule:**
  `constant_with_warmup` (matches 18).
- **Eval/checkpoint cadence** (step-based, unchanged from 18): `--eval-steps 1000`
  (~34 eval points over the single epoch), `--save-steps 10000` (~3
  checkpoints/run; must stay a multiple of `--eval-steps`).
- **Checkpoints:** local only (`--no-push`), under
  `results/<arch>/L<L>H<H>_ep<E>_wd<WD>/seed_<S>/`.

`run.py` (copy `run_proposal.py` → `run.py`) is a thin launcher that shells out to
the runners with a different `--seed` / `--n-layers` / `--n-heads` per job; it
never redefines model or training logic.

## Evaluation — split into two stages (training is fast; full eval is separate)
Unlike exp 18 (which ran a full-validation pass inline at the end of every run),
exp 20 keeps **all eval during training capped** so the 60 runs finish quickly,
and moves the expensive full-set pass into a **separate post-hoc script**.

- **During training (in the runners):** per-step generation-eval runs on a **fixed
  random 2,500-row subset** of val (and a 2,500-row `test` pass at the end), via
  `--max-eval-samples 2500` (shuffle seed 0). This is the *only* eval the training
  job does — the runners' old uncapped `val_full` block was **removed** (replaced
  by a pointer comment). Appears in wandb as `eval_exact_match`.
- **After training (separate script):** [`run_full_evaluate.py`](run_full_evaluate.py)
  (copy `run_full_evaluate_proposal.py` → `run_full_evaluate.py`) loads each run's
  saved `best/` checkpoint and scores it on the **ENTIRE 278,704-row validation
  set** (and optionally full test). It does **not** re-implement eval — it imports
  the runners' own `make_tokenize_fn` / `GenEvalTrainer` / `CausalLMCollator`, so
  the metric matches the training curve exactly. Its `--arch/--config/--seeds/
  --epochs/--weight-decay` mirror `run.py` and resolve the same `best/` dirs;
  results are written one JSON per run under `results/full_eval/`, aggregated to
  `results/full_eval/summary.csv` via `--summarize`.

## How to run

**Recommended — one packed job per (arch, config): 12 jobs, each packs its 5 seeds
on one GPU.** The models are tiny and underutilize an A6000:
```bash
for cfg in 2x2 2x1 3x2 3x1; do
  for arch in vaswani vaswani_rope gpt2_rope; do
    sbatch slurm/run_gpu.sbatch \
      experiments/20_multi_seed_random_depth2_small_models/run.py \
      --arch $arch --config $cfg
  done
done
```
(`--config LxH` = layers × heads; `--config all` packs every config in one job.)

**Alternative — array (1 GPU per run, 60 tasks).** Lower wall-clock per run but
uses 60 GPUs; prefer when the cluster is idle:
```bash
sbatch --array=0-59 slurm/run_gpu.sbatch \
    experiments/20_multi_seed_random_depth2_small_models/run.py
# task -> (arch, config, seed):
#   arch   = ARCHS[task // 20]
#   config = CONFIGS[(task % 20) // 5]
#   seed   = 42 + (task % 5)
```

**Local smoke test (1 arch, 1 config, 2 seeds, frequent eval/save, no wandb/push):**
```bash
uv run python experiments/20_multi_seed_random_depth2_small_models/run.py \
    --arch vaswani --config 2x2 --seeds 42-43 \
    --eval-steps 50 --save-steps 50 --max-parallel 2 --no-wandb
```

Per-run output goes to
`logs/20_multi_seed_random_depth2_small_models_<arch>_L<L>H<H>_ep<E>_wd<WD>_seed_<S>.log`.

## Full-validation evaluation (after training)
Run [`run_full_evaluate.py`](run_full_evaluate.py) once the runs above have
written their `best/` checkpoints. It is heavy (generation over all 278,704 val
rows per model), so prefer array mode for the full 60-run sweep:
```bash
# all 60 runs, 1 GPU each, full validation set:
sbatch --array=0-59 slurm/run_gpu.sbatch \
    experiments/20_multi_seed_random_depth2_small_models/run_full_evaluate.py
# then aggregate the per-run JSONs into results/full_eval/summary.csv (no GPU):
uv run python experiments/20_multi_seed_random_depth2_small_models/run_full_evaluate.py --summarize
```
In-process (one arch+config, all 5 seeds; tokenizes the full split once and reuses
it across seeds), or a quick capped sanity check:
```bash
uv run python experiments/20_multi_seed_random_depth2_small_models/run_full_evaluate.py \
    --arch vaswani --config 2x2                       # full val, 5 seeds
uv run python experiments/20_multi_seed_random_depth2_small_models/run_full_evaluate.py \
    --arch vaswani --config 2x2 --seeds 42 --max-eval-samples 500   # fast sanity
```
Add `--splits validation,test` to also score the full test split. Results: one
JSON per run at `results/full_eval/<arch>_L<L>H<H>_ep<E>_wd<WD>_seed_<S>.json`,
keys `validation_full_exact_match` / `test_full_exact_match`.

## Resuming from checkpoint
Each `checkpoint-<step>/` holds optimizer + scheduler + RNG + `trainer_state.json`.
The launcher's `--resume` makes each `(arch, config, seed)` auto-resume from the
latest `checkpoint-*` in **its own** results dir; a run with no checkpoint yet
starts fresh, so `--resume` is safe on the first submission. Best paired with the
preemptable partition:
```bash
sbatch --array=0-59 slurm/run_preempt.sbatch \
    experiments/20_multi_seed_random_depth2_small_models/run.py --resume
```
The output dir encodes `ep<E>` (and `L<L>H<H>`), so changing `--epochs` or the
config points at a fresh bucket rather than extending in place.

## Viewing all runs together in wandb
The launcher sets two env vars per subprocess (honored automatically), both
encoding (arch, config, epochs, wd):
- `WANDB_RUN_GROUP = 20_multi_seed_random_depth2_small_models_<arch>_L<L>H<H>_ep<E>_wd<WD>`
  — the 5 seeds of each (arch, config) share a **group** → one panel with a mean
  curve + min/max band across seeds.
- `WANDB_TAGS = 20_multi_seed_random_depth2_small_models,<arch>,L<L>H<H>,seed_<S>,ep<E>,wd<WD>`
  — filter all runs with the `20_multi_seed_random_depth2_small_models` tag.

## Caveats
- **One epoch, not a grokking horizon.** Read this as a capacity/seed comparison
  across the 4 configs, not a delayed-transition probe (cf. exp 18's 6 epochs).
- **In-distribution, not extrapolation.** No depth/structure is held out (cf. exp
  16). Expect exact-match to be high; the interesting read is how it degrades as
  layers/heads shrink, and the per-config seed spread.
- **`gpt2_rope` is the capacity-fragile one.** Exp 12 found the decoder-only arch
  degrades sharply at small shapes (mean 0.70 at L2/H2 on the v1 d3 split) while
  both Vaswani enc-decs stayed saturated — watch `L2H1` / `L2H2` here especially.
- **No train-accuracy curve.** Runners compute `exact_match` on val only; train
  performance is visible as `train_loss`.

## Results
_(pending — runs not yet launched)_

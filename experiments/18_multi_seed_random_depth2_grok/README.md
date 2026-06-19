# 18_multi_seed_random_depth2_grok — Observations

## Goal
Run experiment **16's grokking-probe machinery** (3 archs × 5 seeds, long-ish
horizon, `--weight-decay` lever, constant LR, **no early-stop**) on the v2
grammar's **random-frame depth-2** split instead of 16's held-out-depth split.

This is the **in-distribution control** for exp 16. Exp 16 withholds the deeper
depth-2 structures from training (train = shallow, val/test = deeper, held out) to
test depth *extrapolation*. 18 keeps everything identical but swaps in a **random
80/10/10 split over frames**: train/val/test are frame-disjoint, yet all drawn
from the **same depth-{0,1,2} distribution**, so there is no structural gap to
extrapolate across — only unseen frames/lexicalizations of seen depths. It tells
us how the same probe config behaves when generalization is in-distribution, the
baseline against which 16's extrapolation result is read.

Prior signal: exp 15 ran a *random* depth-2 split (k=100) with **early stopping**
and all 30 runs saturated test exact-match (0.984–0.999) quickly. 18 differs by
(a) the much larger, structurally exhaustive k=2 dataset below, and (b) the
grokking config (no early-stop, wd=1.0, constant LR) so the post-convergence
behavior is visible rather than truncated.

## Dataset
`kylelovesllms/hi-hf-v2-frames-1.3M-d2-k2-random`

| split | rows | content |
|---|---|---|
| train | **2,229,638** | depth-{0,1,2} frames, random 80% |
| validation | 278,704 | depth-{0,1,2}, frame-disjoint 10% |
| test | 278,706 | depth-{0,1,2}, frame-disjoint 10% |

Built from the **full depth-2 frame enumeration** (~1,393,524 frames) × **k=2**
lexicalizations each, split `random_frame` (80/10/10 over `frame_id`). Columns
used: `hi` (source) → `hf` (target). No depth/structure is withheld — val/test
are unseen *frames* of the *same* depths, so the gap is lexical/frame-level, not
structural.

**Scale note (drives the horizon below):** at batch 64 this train split is
**~34,838 steps/epoch** — ~9× exp 16's ~3,906. So the **epoch budget is retuned**
(6, not 16's 50) to match 16's ~195k-step horizon; the step-based eval/save
cadence is unchanged.

## Setup
- **Architectures:** drive local runner copies from exp 16 under `runners/` —
  one source of truth per arch:
  | arch key | local runner | based on | model |
  |---|---|---|---|
  | `vaswani` | `runners/vaswani_run.py` | 06 | Vaswani enc-dec, sinusoidal PE |
  | `vaswani_rope` | `runners/vaswani_rope_run.py` | 07 | Vaswani enc-dec, RoPE on Q/K |
  | `gpt2_rope` | `runners/gpt2_rope_run.py` | 08 | GPT2-style decoder-only, RoPE |

  These are 06/07/08 + the `--weight-decay` / `--lr-scheduler` knobs and **no
  early-stop callback** (a grokking probe must train *past* convergence). They
  are exp 16's runners with **one addition** — a `--max-eval-samples` flag — that
  caps the per-step generation-eval to a fixed random subset. Needed because this
  split's val/test are **278,704 / 278,706** rows each (vs exp 16's ~1,250), so
  full-split eval every step would dwarf training (measured ~13 min/eval × ~209
  evals ≈ 47 h). The launcher passes **`--max-eval-samples 2500`** by default →
  ~7 s/eval and a ±~1% exact-match estimate.
- **Tokenizers:** the **v2** tokenizers under `artifacts/` (copied from exp 16) —
  the **137-token** seq2seq tokenizer for `vaswani`/`vaswani_rope`, and the
  **138-token** `<sot>`-augmented one for `gpt2_rope`. Same v2 grammar as 16, so
  the vocab is identical; the launcher threads the right one to each arch.
- **Seeds:** 42–46 (5). **Epochs:** default **6** (~209k steps ≈ exp 16's ~195k;
  16 used 50 epochs on a ~9× smaller split). Extend via `--resume` if memorization
  is clean but eval stays flat.
- **Weight decay:** default **1.0** (matches 16). Constant LR means wd carries all
  the regularization pressure. Sweepable (see below).
- **LR schedule:** `constant_with_warmup` (flat LR after warmup), **not** the
  `linear` decay-to-0 of 06/07/08 — so late steps aren't starved of LR and runs
  extend cleanly.
- **Eval/checkpoint cadence** (step-based, unchanged from 16): `--eval-steps 1000`
  (~every 0.029 epoch → ~209 eval points over 6 epochs), `--save-steps 10000`
  (~21 checkpoints/run; must stay a multiple of `--eval-steps`).
- **Checkpoints:** local only (`--no-push`), under
  `results/<arch>/ep<E>_wd<WD>/seed_<S>/`.

`run.py` (copy `run_proposal.py` → `run.py`) is a thin launcher that shells out to
the runners with a different `--seed` each; it never redefines model or training
logic.

## How to run

**Recommended — packed (3 jobs, one per architecture, all 5 seeds share one GPU).**
The models are tiny (~1.8M params, 4 layers) and underutilize an A6000:
```bash
sbatch slurm/run_gpu.sbatch experiments/18_multi_seed_random_depth2_grok/run.py --arch vaswani
sbatch slurm/run_gpu.sbatch experiments/18_multi_seed_random_depth2_grok/run.py --arch vaswani_rope
sbatch slurm/run_gpu.sbatch experiments/18_multi_seed_random_depth2_grok/run.py --arch gpt2_rope
```

**Alternative — array (1 GPU per run, 15 tasks).** Lower wall-clock per run but
uses 15 GPUs; prefer when the cluster is idle:
```bash
sbatch --array=0-14 slurm/run_gpu.sbatch experiments/18_multi_seed_random_depth2_grok/run.py
# task -> (arch, seed):  arch = ARCHS[task // 5], seed = 42 + task % 5
```

**Local smoke test (2 seeds, 1 epoch, frequent eval/save, no wandb / no push):**
```bash
uv run python experiments/18_multi_seed_random_depth2_grok/run.py \
    --arch vaswani --seeds 42-43 --epochs 1 \
    --eval-steps 50 --save-steps 50 --weight-decay 1.0 \
    --max-parallel 2 --no-wandb
```

Per-run output goes to `logs/18_multi_seed_random_depth2_grok_<arch>_ep<E>_wd<WD>_seed_<S>.log`.

## Resuming from checkpoint
Each `checkpoint-<step>/` dir (written every `--save-steps`) holds `optimizer.pt`
+ scheduler + RNG + `trainer_state.json` — everything HF needs to continue exactly
where it left off (the final `best/` dir is weights-only and is *not* used for
resume). The launcher's sweep-wide `--resume` makes each `(arch, seed)`
auto-resume from the latest `checkpoint-*` in **its own**
`results/<arch>/ep<E>_wd<WD>/seed_<S>/` dir; a run with no checkpoint yet starts
fresh, so `--resume` is safe even on the first submission. Best paired with the
preemptable partition:
```bash
sbatch --array=0-14 slurm/run_preempt.sbatch \
    experiments/18_multi_seed_random_depth2_grok/run.py --resume
```
**Extending the horizon:** the output dir encodes `ep<E>`, so a *larger*
`--epochs` points at a new `ep<E>/` tree with no checkpoints and starts **fresh**.
Decide the full horizon up front, or resume a specific old checkpoint by hand
(`--resume-from-checkpoint results/<arch>/ep6_wd1.0/seed_<S>/checkpoint-<latest>`).

## Weight-decay sweep
WD is the key science knob. Each value writes to its own results subdir + wandb
group, so runs never collide or merge:
```bash
sbatch --array=0-14 slurm/run_gpu.sbatch experiments/18_multi_seed_random_depth2_grok/run.py --weight-decay 0.1
```

## Viewing all runs together in wandb
The launcher sets two env vars per subprocess (honored automatically by
`wandb.init` / the HF Trainer — no edits to the runners), both encoding (epochs, wd):
- `WANDB_RUN_GROUP = 18_multi_seed_random_depth2_grok_<arch>_ep<E>_wd<WD>` — the 5
  seeds of each (arch, epochs, wd) share a **group** → one panel with a mean curve
  + min/max band across seeds.
- `WANDB_TAGS = 18_multi_seed_random_depth2_grok,<arch>,seed_<S>,ep<E>,wd<WD>` —
  filter all runs of this experiment with the `18_multi_seed_random_depth2_grok` tag.

## Caveats
- **In-distribution, not extrapolation.** Unlike 16, no depth/structure is held
  out — val/test are unseen *frames* of *seen* depths. Expect exact-match to rise
  (cf. exp 15 saturating ~0.99), so the interesting read is the *rate*/seed-spread
  and whether the no-early-stop + wd=1.0 config shows any post-convergence drift,
  not a delayed grok transition. Treat "grok" in the name as the inherited probe
  config, not a prediction.
- **No train-accuracy curve.** Runners compute `exact_match` on the *val* set
  only; train performance is visible only as `train_loss`.
- **Horizon is the first knob to revisit.** 6 epochs is sized to 16's step budget;
  the `ep<E>` dir-encoding means a larger `--epochs` starts a fresh bucket (see
  *Resuming*), so prefer setting the full horizon up front.
- **Runtime.** ~34,838 steps/epoch + generation-eval on 278k val rows is far
  heavier per epoch than 16 — **eval over the full val split could dominate**;
  watch the first job's pace against the 2-day walltime and consider a smaller
  eval subset or array mode if a packed arch can't finish 6 epochs in time.

## Results
_(pending — runs not yet launched)_

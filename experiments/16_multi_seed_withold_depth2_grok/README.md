# 16_multi_seed_withold_depth2_grok — Observations

## Goal
Rerun experiment **10's grokking probe** on the **v2 grammar** (relative clauses
+ adjunction) using its **depth-withhold-2** split. We look for **grokking**
(delayed generalization): `train_loss` collapses early on the shallow train
frames while held-out `eval_exact_match` on the deeper structures stays low,
then — if grokking occurs — rises sharply much later.

Why this experiment exists: exp 10 ran the identical probe (3 archs × seeds,
long horizon, a `--weight-decay` lever, constant LR) on the *v1* grammar's
held-out-depth split and **did not grok at wd=0.1** — every run memorized
(`train_loss→0`) then *overfit* (held-out `eval_loss` climbed to 3–6, test
exact-match 0.0). 16 asks whether the richer **v2** grammar + a **stronger
weight decay (1.0)** recovers depth generalization.

## Dataset
`kylelovesllms/hi-hf-v2-frames-depth_withhold2_k_train100_k_eval5_eval_frame_cap500`

| split | rows | tree_height (structure depth) |
|---|---|---|
| train | **249,964** | shallow (≤ ~12) |
| validation | 1,250 | **deeper, held out** (9–17, mostly 12–16) |
| test | 1,250 | **deeper, held out** (9–17, mostly 12–16) |

Columns used: `hi` (source) → `hf` (target). The split withholds the deeper
depth-2 structures from training, so val/test are genuinely unseen depths — the
real generalization gap a grokking probe needs.

**Scale note (drives the cadence below):** at batch 64 this train split is
**~3,906 steps/epoch** — ~60× exp 10's ~66 steps/epoch. So the epoch budget and
eval/save cadence are set in *steps* appropriate to this size, not copied from
10.

## Setup
- **Architectures:** drive the **byte-identical** runner copies from exp 10
  under `runners/` (one source of truth per arch):
  | arch key | local runner | based on | model |
  |---|---|---|---|
  | `vaswani` | `runners/vaswani_run.py` | 06 | Vaswani enc-dec, sinusoidal PE |
  | `vaswani_rope` | `runners/vaswani_rope_run.py` | 07 | Vaswani enc-dec, RoPE on Q/K |
  | `gpt2_rope` | `runners/gpt2_rope_run.py` | 08 | GPT2-style decoder-only, RoPE |

  These are 06/07/08 + the two added knobs `--weight-decay` and `--lr-scheduler`
  threaded into `(Seq2Seq)TrainingArguments`, and — critically — **no early-stop
  callback** (a grokking probe must train *past* convergence, unlike exp 15).
  We copy rather than edit per repo convention.
- **Tokenizer (the only thing 16 changes vs 10's runner defaults):** the **v2**
  tokenizers under `artifacts/` (copied from exp 15) — the **137-token** seq2seq
  tokenizer for `vaswani`/`vaswani_rope`, and the **138-token** `<sot>`-augmented
  one for `gpt2_rope`. The launcher threads the right one to each arch via
  `--tokenizer`; the runners are otherwise unchanged.
- **Seeds:** 42–46 (5). **Epochs:** default **50** (~195k steps ≈ exp 10's total
  compute; extend if memorization is clean but eval stays flat).
- **Weight decay:** default **1.0** — the headline science knob. Exp 10 showed
  wd=0.1 fails to grok on the analogous v1 task, and constant LR means weight
  decay carries *all* the regularization pressure. Sweepable (see below).
- **LR schedule:** `constant_with_warmup` (flat 3e-4 after a 200-step warmup),
  **not** 06/07/08's `linear` decay-to-0 — a decaying LR is ≈0 by the late
  epochs where grokking would appear, and a flat LR lets runs extend cleanly.
- **Eval/checkpoint cadence** (set by the launcher): `--eval-steps 1000`
  (~every 0.26 epoch → ~195 eval points over 50 epochs), `--save-steps 10000`
  (~20 checkpoints/run, not ~195; must stay a multiple of `--eval-steps`).
- **Checkpoints:** local only (`--no-push`), under
  `results/<arch>/ep<E>_wd<WD>/seed_<S>/`.

`run.py` (copy `run_proposal.py` → `run.py`) is a thin launcher that shells out
to the runners with a different `--seed` each; it never redefines model or
training logic.

## How to run

**Recommended — packed (3 jobs, one per architecture, all 5 seeds share one GPU).**
The models are tiny (~1.8M params, 4 layers) and underutilize an A6000, so 5
seeds pack onto one GPU efficiently — 3 GPUs total instead of 15:
```bash
sbatch slurm/run_gpu.sbatch experiments/16_multi_seed_withold_depth2_grok/run.py --arch vaswani
sbatch slurm/run_gpu.sbatch experiments/16_multi_seed_withold_depth2_grok/run.py --arch vaswani_rope
sbatch slurm/run_gpu.sbatch experiments/16_multi_seed_withold_depth2_grok/run.py --arch gpt2_rope
```

**Alternative — array (1 GPU per run, 15 tasks).** Lower wall-clock per run but
uses 15 GPUs; prefer when the cluster is idle and you want results fastest:
```bash
sbatch --array=0-14 slurm/run_gpu.sbatch experiments/16_multi_seed_withold_depth2_grok/run.py
# task -> (arch, seed):  arch = ARCHS[task // 5], seed = 42 + task % 5
```

**Local smoke test (2 seeds, 1 epoch, frequent eval/save, no wandb / no push):**
```bash
uv run python experiments/16_multi_seed_withold_depth2_grok/run.py \
    --arch vaswani --seeds 42-43 --epochs 1 \
    --eval-steps 50 --save-steps 50 --weight-decay 1.0 \
    --max-parallel 2 --no-wandb
```

Per-run output goes to `logs/16_multi_seed_withold_depth2_grok_<arch>_ep<E>_wd<WD>_seed_<S>.log`.

## Resuming from checkpoint
Each `checkpoint-<step>/` dir (written every `--save-steps`) holds `optimizer.pt`
+ scheduler + RNG + `trainer_state.json` — everything HF needs to continue
training exactly where it left off. The final `best/` dir is **weights-only** (no
optimizer) and is *not* used for resume. The runners take
`--resume-from-checkpoint`; the launcher exposes a sweep-wide `--resume` that
makes each `(arch, seed)` auto-resume from the latest `checkpoint-*` in **its
own** `results/<arch>/ep<E>_wd<WD>/seed_<S>/` dir. A run with **no checkpoint yet
starts fresh**, so `--resume` is safe to pass even on the very first submission.

**Primary use — preemption / crash recovery (same `--epochs`).** This is the
robust case and the reason it matters here: the `nlp` partition is preemptable,
so a job can be killed mid-run. Submit on the preemptable partition with
`--resume` from the start; on any requeue it picks up from the last checkpoint
instead of restarting from step 0:
```bash
sbatch --array=0-14 slurm/run_preempt.sbatch \
    experiments/16_multi_seed_withold_depth2_grok/run.py --resume
```
(Manual single-run resume: pass the runner an explicit dir,
`--resume-from-checkpoint results/.../checkpoint-120000`.) Constant LR means the
schedule is well-defined at any step, so resume is exact.

**Extending the horizon (e.g. 50 → 150 epochs) — read this first.** The output
dir encodes `ep<E>`, so re-submitting with a *larger* `--epochs` points each run
at a **new** `ep150/` tree that has no checkpoints — it would start **fresh**, not
extend. This is intentional (different fixed budgets stay in separate buckets +
wandb groups, mirroring exp 10), but it means "extend" is not a one-flag change.
To genuinely train longer, either:
- **Decide the horizon up front** and set the full `--epochs` on the first run
  (simplest; recommended given constant LR), or
- Resume a *specific* old checkpoint into the new run by hand:
  `--resume-from-checkpoint results/<arch>/ep50_wd1.0/seed_<S>/checkpoint-<latest>`
  on a single runner (the sweep-wide `--resume` can't express per-run paths).

## Weight-decay sweep
WD is the key science knob. Re-submit with a different `--weight-decay`; each
value writes to its own results subdir and its own wandb group, so runs never
collide or merge:
```bash
sbatch --array=0-14 slurm/run_gpu.sbatch experiments/16_multi_seed_withold_depth2_grok/run.py --weight-decay 0.1
```

## Viewing all runs together in wandb
The launcher sets two env vars per subprocess (honored automatically by
`wandb.init` / the HF Trainer — **no edits to the runners**), both encoding
(epochs, wd) so different sweep points stay separate:
- `WANDB_RUN_GROUP = 16_multi_seed_withold_depth2_grok_<arch>_ep<E>_wd<WD>` — the
  5 seeds of each (arch, epochs, wd) setting share a **group** → one panel with
  a **mean curve + min/max band** across seeds.
- `WANDB_TAGS = 16_multi_seed_withold_depth2_grok,<arch>,seed_<S>,ep<E>,wd<WD>` —
  filter all runs of this experiment with the `16_multi_seed_withold_depth2_grok`
  tag.

In the wandb UI (project `constituent-geometry`): filter by tag
`16_multi_seed_withold_depth2_grok`, overlay the three arch groups, and watch for
`train_loss` collapsing early while `eval_exact_match` rises only much later.

## Caveats
- **No train-accuracy curve.** The runners compute `exact_match` on the *val*
  set only; train performance is visible only as `train_loss`. The grokking read
  is therefore `train_loss → ~0` vs delayed `eval_exact_match` rise.
- **Epoch budget is the first knob to revisit.** Grokking can be very delayed;
  50 epochs is a first probe sized to exp 10's compute. If you see clean
  memorization (`train_loss→0`) but flat held-out eval, train longer — but note
  the `ep<E>` dir-encoding means a larger `--epochs` starts a fresh bucket rather
  than extending in place (see *Resuming from checkpoint*), so prefer setting the
  full horizon up front.
- **WD value matters.** Default 1.0 maximizes the chance of a late transition,
  but too-large wd can stop the model memorizing at all. Sweep {0.1, 1.0}.
- **Runtime.** ~3,906 steps/epoch + generation-eval on 1,250 val rows makes each
  run heavier than exp 10's. Default is **packed** (3 GPUs): the 4-layer models
  underutilize an A6000, so 5 seeds share one GPU with little slowdown. Watch the
  first job's pace against the 2-day walltime; switch to array mode if a packed
  arch can't finish 50 epochs in time.

## Results
_(pending — runs not yet launched)_

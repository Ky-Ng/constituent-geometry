# 19_NP_adjunct_generalization — Observations

## Goal
Run experiment **16/18's grokking-probe machinery** (3 archs × 5 seeds, long
horizon, `--weight-decay` lever, constant LR, **no early-stop**) on a v2-grammar
split that holds out **relative clauses on the subject**.

The question: does a relative clause — a post-nominal **NP adjunct** — learned
only on **object** noun phrases transfer to the **subject** position? This is the
**position-transfer analog of exp 16's depth gap**, and the depth-2 incarnation
of exp 17's subject-position hypothesis, run with the grokking config rather than
early stopping.

Where 16/18 sit:
- **16** = held-out **depth** (extrapolate to deeper structures) — a real gap.
- **18** = random-frame depth-2 — the **in-distribution control** (no gap).
- **19** = held-out **modification position** (RC on subject) — a real
  *structural* gap, like 16, but along a different axis.

Grokking signature here: `train_loss` collapses early (memorization of the
object-only-adjunct train frames) while held-out `eval_exact_match` on the
subject-RC structures stays low, then — if grokking occurs — rises sharply much
later.

## Dataset
`kylelovesllms/hi-hf-v2-frames-object-rc-only-depth2`

| split | rows | `subject_branching` | `subject_mod` | depth | frames |
|---|---|---|---|---|---|
| train | **3,936** | `False` (all) | `''` (all) | {0,1,2} (mostly 2) | 1,312 |
| validation | 300 | `True` (all) | `rel` (54) / `adj+rel` (246) | 2 | 100 |
| test | 300 | `True` (all) | `rel` (48) / `adj+rel` (252) | 2 | 100 |

Columns used: `hi` (source) → `hf` (target). **Train subjects are always atomic**
(proper name or bare `D N`) — every adjective/relative clause in training lives on
an **object**. **Val/test subjects always carry a relative clause** (the held-out
structure); frames are disjoint from train. So the gap is genuinely structural —
an RC/adjunct on the *subject* is never seen during training — exactly what a
grokking probe needs. (`subject_branching`/`subject_mod` are exp 17's columns.)

**Scale note (drives the horizon below):** this train split is **tiny — 3,936
rows ≈ ~62 steps/epoch at batch 64**, the same scale as exp **10**'s grokking
probe (~66 steps/epoch), and ~60× smaller than 16's and ~570× smaller than 18's.
So the epoch budget is **not** copied from 16/18; it is matched in *epochs* to
exp 10 (see below).

## Setup
- **Architectures:** drive the **byte-identical** runner copies from exp 16/18
  (verified `diff`-clean) under `runners/` — one source of truth per arch:
  | arch key | local runner | based on | model |
  |---|---|---|---|
  | `vaswani` | `runners/vaswani_run.py` | 06 | Vaswani enc-dec, sinusoidal PE |
  | `vaswani_rope` | `runners/vaswani_rope_run.py` | 07 | Vaswani enc-dec, RoPE on Q/K |
  | `gpt2_rope` | `runners/gpt2_rope_run.py` | 08 | GPT2-style decoder-only, RoPE |

  These are 06/07/08 + the `--weight-decay` / `--lr-scheduler` knobs and **no
  early-stop callback** (a grokking probe must train *past* convergence). Copied,
  not edited, per repo convention.
- **Tokenizers:** the **v2** tokenizers under `artifacts/` (copied from exp 18,
  verified identical) — the **137-token** seq2seq tokenizer for
  `vaswani`/`vaswani_rope`, and the **138-token** `<sot>`-augmented one for
  `gpt2_rope`. Same v2 grammar as 15/16/18, so the vocab is identical; the
  launcher threads the right one to each arch.
- **Seeds:** 42–46 (5). **Epochs:** default **3000** (~186k steps ≈ exp
  10/16/18's ~195k-step horizon; matched in epochs to exp 10 since this split is
  exp-10 scale). Extend via `--resume` if memorization is clean but eval stays flat.
- **Weight decay:** default **1.0** (matches 16/18). Constant LR means wd carries
  all the regularization pressure. Sweepable (see below).
- **LR schedule:** `constant_with_warmup` (flat LR after a 200-step warmup),
  **not** the `linear` decay-to-0 of 06/07/08 — so late epochs aren't starved of
  LR and runs extend cleanly.
- **Eval/checkpoint cadence** (step-based, unchanged from 16/18): `--eval-steps
  1000` (~every 16 epochs → ~186 eval points over 3000 epochs), `--save-steps
  10000` (~18 checkpoints/run; must stay a multiple of `--eval-steps`).
- **Checkpoints:** local only (`--no-push`), under
  `results/<arch>/ep<E>_wd<WD>/seed_<S>/`.

`run.py` (copy `run_proposal.py` → `run.py`) is a thin launcher that shells out to
the runners with a different `--seed` each; it never redefines model or training
logic.

## How to run

**Recommended — packed (3 jobs, one per architecture, all 5 seeds share one GPU).**
The models are tiny (~1.8M params, 4 layers) and underutilize an A6000:
```bash
sbatch slurm/run_gpu.sbatch experiments/19_NP_adjunct_generalization/run.py --arch vaswani
sbatch slurm/run_gpu.sbatch experiments/19_NP_adjunct_generalization/run.py --arch vaswani_rope
sbatch slurm/run_gpu.sbatch experiments/19_NP_adjunct_generalization/run.py --arch gpt2_rope
```

**Alternative — array (1 GPU per run, 15 tasks).** Lower wall-clock per run but
uses 15 GPUs; prefer when the cluster is idle:
```bash
sbatch --array=0-14 slurm/run_gpu.sbatch experiments/19_NP_adjunct_generalization/run.py
# task -> (arch, seed):  arch = ARCHS[task // 5], seed = 42 + task % 5
```

**Local smoke test (2 seeds, short horizon, frequent eval/save, no wandb / no push):**
```bash
uv run python experiments/19_NP_adjunct_generalization/run.py \
    --arch vaswani --seeds 42-43 --epochs 5 \
    --eval-steps 50 --save-steps 50 --weight-decay 1.0 \
    --max-parallel 2 --no-wandb
```

Per-run output goes to `logs/19_NP_adjunct_generalization_<arch>_ep<E>_wd<WD>_seed_<S>.log`.

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
    experiments/19_NP_adjunct_generalization/run.py --resume
```
**Extending the horizon:** the output dir encodes `ep<E>`, so a *larger*
`--epochs` points at a new `ep<E>/` tree with no checkpoints and starts **fresh**.
Decide the full horizon up front, or resume a specific old checkpoint by hand
(`--resume-from-checkpoint results/<arch>/ep3000_wd1.0/seed_<S>/checkpoint-<latest>`).

## Weight-decay sweep
WD is the key science knob. Each value writes to its own results subdir + wandb
group, so runs never collide or merge:
```bash
sbatch --array=0-14 slurm/run_gpu.sbatch experiments/19_NP_adjunct_generalization/run.py --weight-decay 0.1
```

## Viewing all runs together in wandb
The launcher sets two env vars per subprocess (honored automatically by
`wandb.init` / the HF Trainer — no edits to the runners), both encoding (epochs, wd):
- `WANDB_RUN_GROUP = 19_NP_adjunct_generalization_<arch>_ep<E>_wd<WD>` — the 5
  seeds of each (arch, epochs, wd) share a **group** → one panel with a mean curve
  + min/max band across seeds.
- `WANDB_TAGS = 19_NP_adjunct_generalization,<arch>,seed_<S>,ep<E>,wd<WD>` —
  filter all runs of this experiment with the `19_NP_adjunct_generalization` tag.

## Caveats
- **Structural gap, like 16 — not the in-distribution control (18).** The
  held-out axis is *which NP carries the adjunct* (object→subject), not depth.
  Expect held-out `eval_exact_match` to start low; the interesting read is whether
  it grokks (delayed rise) or stays flat / overfits (the exp-10 outcome on the v1
  depth gap).
- **No train-accuracy curve.** Runners compute `exact_match` on the *val* set
  only; train performance is visible only as `train_loss`. The grokking read is
  therefore `train_loss → ~0` vs a delayed `eval_exact_match` rise.
- **Held-out subjects always have an RC.** `subject_mod` is `rel` or `adj+rel`
  (never `adj`-only), so this isolates **relative-clause** transfer to the subject,
  not bare-adjective transfer. A pure-`adj` subject split would be a separate probe.
- **Horizon is the first knob to revisit.** 3000 epochs is matched to exp 10's
  step budget; the `ep<E>` dir-encoding means a larger `--epochs` starts a fresh
  bucket (see *Resuming*), so prefer setting the full horizon up front.
- **Runtime.** Tiny dataset (3,936 train, 300 val) + tiny models → far lighter
  per epoch than 16/18; generation-eval on only 300 val rows is cheap. 3000 epochs
  at this scale fit comfortably in one 2-day job (cf. exp 10's 3000-epoch runs).

## Results

Two horizons run so far (2026-06-18), both packed (3 jobs × 5 seeds), wd=1.0,
`constant_with_warmup`, no early-stop. Walltime was capped (`--time`) on
submission to fit before the `jun26maint` reservation (06-19 13:00 → 06-21 18:00),
which the template's 2-day default would otherwise have blocked.

### 30-epoch smoke run — pipeline validation ✅
Jobs 4202145/46/47, `--epochs 30 --eval-steps 100 --save-steps 500`, ~2 min each.
All 15 runs `COMPLETED` (exit 0). Purpose was to validate the machinery, not to
probe generalization (30 ep ≈ 1.6% of the horizon). Confirmed: v2 dataset loads,
both v2 tokenizers thread through, all 3 archs × 5 seeds train + held-out eval
with no errors, and checkpoints (`checkpoint-500/1000/1500/1860` + `best/`) are
written per run. test_exact_match = 0.0 everywhere (expected this early).

### 1000-epoch run — clean negative ❌ (no transfer through 1000 epochs)
Jobs 4202151/52/53, `--epochs 1000` (≈62,000 steps), default cadence
(`--eval-steps 1000`/`--save-steps 10000` → 62 eval points + 6 checkpoints/run).
All 15 runs `COMPLETED` (exit 0); elapsed vaswani 44:53, vaswani_rope 48:25,
gpt2_rope 36:03.

| arch | final `train_loss` | held-out `eval_exact_match` | `test_exact_match` |
|---|---|---|---|
| vaswani | ~0.002–0.006 | **0** at all 62 evals × 5 seeds | **0.0** (×5) |
| vaswani_rope | ~0.0009–0.005 | **0** at all 62 evals × 5 seeds | **0.0** (×5) |
| gpt2_rope | ~0.005–0.009 | **0** at all 62 evals × 5 seeds | **0.0** (×5) |

- **Memorization is complete** — `train_loss → ~10⁻³` on all 15 runs (the
  object-adjunct / atomic-subject train distribution is fully learned).
- **Subject-RC transfer never appears** — across **930 evaluations** (15 runs ×
  62 eval points), held-out `eval_exact_match` is *only ever exactly 0*; it never
  lifts off the floor. Final `test_exact_match` = 0.0 on all 15.
- **Held-out loss shows the overfitting U-shape**, not a plateau: e.g.
  vaswani_rope seed 42 `eval_loss` climbs **5.38 → ~7.8** over training. Same
  memorize-then-overfit shape as exp 10's v1 depth-gap negative.

**Reading:** at 1000 epochs / wd=1.0 the object→subject relative-clause (NP
adjunct) gap is **not** crossed by any architecture or seed — the position-transfer
analog reproduces exp 10's "memorize then overfit, generalization stays at 0"
pattern. Caveat: this is ⅓ of the planned 3000-epoch horizon, so grokking is not
formally ruled out (it can follow long overfitting) — but `eval_loss` *climbing*
rather than plateauing is a mildly discouraging sign for a late transition.

**Open / next:** (a) the full **3000-epoch** horizon (default `run.py`, capped
walltime); (b) a **wd sweep** (e.g. wd=0.1) since 1.0 may be over-regularizing.
Buckets `results/<arch>/ep30_wd1.0/` and `results/<arch>/ep1000_wd1.0/` hold these
runs' checkpoints + `best/`.

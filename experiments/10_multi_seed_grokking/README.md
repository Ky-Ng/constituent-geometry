# 10_multi_seed_grokking — Observations

## Goal
Look for **grokking** (delayed generalization) on the **held-out-depth** HI→HF
frame task, across seeds and the three architectures from 06/07/08. Experiment
09 only trained 30 epochs on the `random_frame` split, where the models already
reach ~1.0 test exact-match — there is no gap to "grok". Grokking needs (a) a
real generalization gap and (b) a long training horizon, so 10 changes three
things relative to 09:

1. **Dataset** → `kylelovesllms/hi_hf_frames_d3_100_heldoutdepth_3`
   (train 4,164 / val 2,400 / test 2,400; train sees depths 0–2, val/test are
   **unseen depth-3** frames).
2. **Epochs** → **3000** (vs 30).
3. **Weight decay** exposed as a sweepable knob (the main grokking lever).

**Grokking signature here:** `train_loss` collapses toward ~0 early
(memorization) while `eval_exact_match` on the held-out depth-3 split stays low,
then — if grokking occurs — rises sharply much later.

## Setup
- **Architectures:** reuse the 06/07/08 model/training logic via **local runner
  copies** under `runners/` (one source of truth per arch):
  | arch key | local runner | based on | model |
  |---|---|---|---|
  | `vaswani` | `runners/vaswani_run.py` | 06 | Vaswani enc-dec, sinusoidal PE |
  | `vaswani_rope` | `runners/vaswani_rope_run.py` | 07 | Vaswani enc-dec, RoPE on Q/K |
  | `gpt2_rope` | `runners/gpt2_rope_run.py` | 08 | GPT2-style decoder-only, RoPE |

  Each runner is a **verbatim copy** of its 06/07/08 source with **two** added
  arguments, `--weight-decay` and `--lr-scheduler`, threaded into
  `(Seq2Seq)TrainingArguments`. We copy rather than edit 06/07/08 because (a)
  repo convention forbids editing existing experiments and (b) those scripts
  hardcode `weight_decay=0.0` and `lr_scheduler_type="linear"`, both of which
  grokking depends on. Everything else (tokenizer defaults, model code, eval
  metric, the GPT2 `GenEvalTrainer`) is unchanged, so 10 stays directly
  comparable to 06/07/08/09.
- **Seeds:** 42–46 (5). **Epochs:** 3000. **Weight decay:** default 0.1.
- **LR schedule:** `constant_with_warmup` (flat 3e-4 after a 200-step warmup) —
  **not** 06/07/08's `linear` decay-to-0. A decaying LR is ≈0 by the late epochs
  where grokking would appear, starving the transition; a flat LR also lets runs
  be extended cleanly (see *Continuing / resuming training*). The `--weight-decay`
  knob supplies the regularization pressure that LR decay otherwise would.
- **Eval/checkpoint frequency** (set by the launcher; see *Why these defaults*):
  `--eval-steps 330` (~every 5 epochs → ~600 eval points), `--save-steps 9900`
  (~20 checkpoints/run instead of ~990).
- **Checkpoints:** local only (`--no-push`), under
  `results/<arch>/ep<E>_wd<WD>/seed_<S>/`.

`run.py` (copy `run_proposal.py` → `run.py`; likewise each `runners/*_run_proposal.py`
→ `runners/*_run.py`) is a thin launcher that shells out to the runners with a
different `--seed` each; it never redefines model or training logic.

## Why these defaults (eval/save frequency, disk)
On this dataset there are **66 steps/epoch** (4,164 train / batch 64), so 3000
epochs ≈ **198,000 steps/run**. The child scripts default to `save_steps=200`
with **no `save_total_limit`**, which we cannot change without editing them.
Left alone that is ~990 checkpoint dirs/run × 15 runs ≈ hundreds of GB. So the
launcher passes a **large** `--save-steps` (default 9,900 = 30 × eval_steps) to
keep ~20 checkpoints/run while still evaluating often enough (`--eval-steps 330`)
to resolve a grok transition. `--save-steps` must remain a round multiple of
`--eval-steps` (HF enforces this under `load_best_model_at_end=True`); the
launcher asserts this before submitting.

## How to run

**Default — packed (3 jobs, one per architecture, all 5 seeds share one GPU):**
```bash
sbatch slurm/run_gpu.sbatch experiments/10_multi_seed_grokking/run.py --arch vaswani
sbatch slurm/run_gpu.sbatch experiments/10_multi_seed_grokking/run.py --arch vaswani_rope
sbatch slurm/run_gpu.sbatch experiments/10_multi_seed_grokking/run.py --arch gpt2_rope
```

**Alternative — array (1 GPU per run, 15 tasks):**
```bash
sbatch --array=0-14 slurm/run_gpu.sbatch experiments/10_multi_seed_grokking/run.py
# task -> (arch, seed):  arch = ARCHS[task // 5], seed = 42 + task % 5
```

**Local smoke test (2 seeds, 1 epoch, frequent eval/save, no wandb / no push):**
```bash
uv run python experiments/10_multi_seed_grokking/run.py \
    --arch vaswani --seeds 42-43 --epochs 1 \
    --eval-steps 20 --save-steps 20 --weight-decay 0.1 \
    --max-parallel 2 --no-wandb
```

Per-run output goes to `logs/10_multi_seed_grokking_<arch>_ep<E>_wd<WD>_seed_<S>.log`.

## Weight-decay sweep
WD is the key science knob. Sweep it by re-submitting with a different
`--weight-decay`; each value writes to its own results subdir and its own wandb
group, so runs never collide or merge:
```bash
sbatch slurm/run_gpu.sbatch experiments/10_multi_seed_grokking/run.py --arch vaswani --weight-decay 0.01
sbatch slurm/run_gpu.sbatch experiments/10_multi_seed_grokking/run.py --arch vaswani --weight-decay 1.0
```

## Viewing all runs together in wandb
The launcher sets two env vars per subprocess (honored automatically by
`wandb.init` / the HF Trainer — **no edits to the runners**), both encoding
(epochs, wd) so different sweep points stay separate:
- `WANDB_RUN_GROUP = 10_multi_seed_grokking_<arch>_ep<E>_wd<WD>` — the 5 seeds of
  each (arch, epochs, wd) setting share a **group** → one panel with a **mean
  curve + min/max band** across seeds.
- `WANDB_TAGS = 10_multi_seed_grokking,<arch>,seed_<S>,ep<E>,wd<WD>` — filter all
  runs of this experiment with the `10_multi_seed_grokking` tag.

In the wandb UI (project `constituent-geometry`): filter by tag
`10_multi_seed_grokking`, overlay the three arch groups, and watch for
`train_loss` collapsing early while `eval_exact_match` rises only much later.

## Continuing / resuming training (e.g. 3000 → 5000 epochs)
**Is the optimizer state saved?** Yes — each `checkpoint-<step>/` dir written at
`--save-steps` contains `optimizer.pt` (+ scheduler, RNG, `trainer_state.json`),
which is everything `Trainer.train(resume_from_checkpoint=...)` needs. The
`best/` dir saved at the end is **weights-only** (no optimizer), so resume from
the **latest `checkpoint-<step>/`**, not from `best/`. With the default
`--save-steps 9900` we keep ~20 such dirs per run, so a resumable checkpoint near
the end always exists.

**Caveat before relying on extend-and-resume:**
- **The runners don't wire `resume_from_checkpoint` yet** — they call
  `trainer.train()`. Continuing needs a small opt-in flag (a one-line change to
  the runners); not added yet. Until then, the simplest path is to set the full
  epoch budget up front.

The **LR-schedule** obstacle is already handled: this experiment defaults to
`constant_with_warmup` (flat LR after warmup), so the LR at any step does not
depend on the planned total — extending 3000→5000 needs no schedule recompute
and the late epochs aren't starved of LR. (06/07/08's default `linear`
decay-to-0 would have made both a problem.)

## Caveats
- **No train-accuracy curve.** The runners only compute `exact_match` on the
  *val* set; train performance is visible only as `train_loss`. The grokking
  read is therefore `train_loss → ~0` vs delayed `eval_exact_match` rise. Adding
  a periodic train-subset eval is a clean follow-up.
- **WD value matters.** Default 0.1 is a middle ground; too small may never
  grok, too large may stop the model memorizing at all. Sweep {0.01, 0.1, 1.0}.
- **Runtime.** Packed, 5 seeds × 3000 epochs ≈ ~4 h wall per arch (3 archs on 3
  GPUs in parallel). A 3-value WD sweep is ~12 h of GPU time.

## Results
_(15-run sweep complete — 3 archs × 5 seeds, **wd=0.1**, full 3000 epochs / 198,000
steps each. `held_out_depth` depth-3: train depths 0–2, val/test unseen depth-3.)_

**No grokking at wd=0.1.** The grokking signature — early `train_loss→0` followed
by a *delayed* `eval` rise — did not appear. Instead every run **memorized and then
overfit**:

| arch | train_loss (final) | eval_loss (min → final) | eval_exact_match (max → final) | test exact-match |
|---|---|---|---|---|
| `vaswani` | ~0.000 | ~0.9–1.4 → **4.2–4.8** | 0.06–0.14 → **0.0** | **0.0** (all seeds) |
| `vaswani_rope` | ~0.000 | ~0.95–1.2 → **2.9–3.8** | 0.00 → **0.0** | **0.0** (all seeds) |
| `gpt2_rope` | ~0.000 | ~1.6–1.8 → **5.2–6.4** | ~0.00 → **0.0** | **0.0** (all seeds) |

What the curves show (per-run details from each `checkpoint-198000/trainer_state.json`):
- **Full memorization:** `train_loss` collapses to ~0 for all 15 runs (train depths
  0–2 are fit exactly).
- **U-shaped eval, not grokking:** `eval_loss` reaches an early minimum then *climbs*
  steadily to a large final value — classic overfitting on the held-out depth-3
  frames. The held-out loss gets **worse**, not better, with more training.
- **Held-out exact-match never takes off:** `gpt2_rope` and `vaswani_rope` stay at
  ~0 throughout; only `vaswani` flickers to a transient **0.05–0.14** mid-run before
  decaying back to 0. Final test exact-match is **0.0 for all 3 archs × 5 seeds**.
- The original `vaswani` reaches the lowest eval minimum and the only nonzero
  exact-match, `vaswani_rope` has the lowest *final* eval_loss (~3), and `gpt2_rope`
  overfits hardest (final eval_loss ~5–6).

**Interpretation.** Depth-3 compositional generalization is *not* recovered by long
training + wd=0.1 alone on this task — the models find a non-recursive memorizing
solution. This is a clean negative result that sets up the intended **weight-decay
sweep** (`--weight-decay 0.01 / 1.0`, see above) and possibly other levers (LR,
longer horizon, train-subset eval) as the next steps. The constant-LR schedule means
these 3000-epoch runs can also be cleanly extended once `resume_from_checkpoint` is
wired (see *Continuing / resuming*).

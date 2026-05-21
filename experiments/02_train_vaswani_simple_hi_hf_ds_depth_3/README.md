# 02 — Train Vaswani encoder-decoder on HI → HF (depth 3)

Train the from-scratch Vaswani encoder-decoder (`src/architecture/`) to translate
Head-Initial sentences into their Head-Final counterparts, on the depth-3 toy-CFG
dataset from experiment 01.

## Setup

Prerequisites (build once):

```bash
uv run python experiments/01_build_simple_hi_hf_ds/run.py            # -> data/hi_hf_dataset_depth_3
uv run python -m architecture.build_tokenizer --out data/tokenizer   # -> data/tokenizer
```

Run:

```bash
# local
uv run python experiments/02_train_vaswani_simple_hi_hf_ds_depth_3/run.py --epochs 5

# cluster
sbatch slurm/run_gpu.sbatch experiments/02_train_vaswani_simple_hi_hf_ds_depth_3/run.py --epochs 5
```

`--no-wandb` (or `WANDB_MODE=disabled`) turns off tracking.

## What it does

- **Task:** seq2seq, `hi` (encoder input) → `hf` (decoder target). Swap with `--src hf --tgt hi`.
- **Model knobs:** `--d-model --n-heads --n-layers --d-ff --dropout` (see `run.py --help`).
  `vocab_size` is taken from the tokenizer; `n_layers` is shared by encoder and decoder.
- **Reusable loop:** `src/training_loops/seq2seq.py` (AdamW, linear warmup schedule,
  gradient clipping, checkpointing).
- **Metrics (per eval):**
  - `val/token_acc` — teacher-forced next-token accuracy (every eval, full val set).
  - `val/exact_match` — full-sequence accuracy via `.generate()` (capped to a few batches).
- **Checkpoints:** best (by exact-match) in `results/best/`, last in `results/final/` —
  each a full HF model dir (`config.json` + `model.safetensors` + tokenizer).

## Results

- **Eval loss ≈ 0.0033** on the random validation split — i.e. the model fits the
  HI→HF transformation essentially perfectly *within the training distribution*.

## Observations

**The low loss is not informative about generalization.** The 80/10/10 split is
random, so train and validation share the same sentence *structures* and the full
vocabulary (see `experiments/01_build_simple_hi_hf_ds/run.py`). Validation is
therefore i.i.d. with train, and a near-zero loss cannot distinguish a genuine
recursive reordering algorithm from a shallow, structure-bound pattern matcher.
This is what motivated **experiment 03** (a depth-held-out split that turns the
eval into a real distribution shift).

> Note: "memorization" is unlikely in the literal sense — the model has ~1–2M
> params vs. ~1.34M training pairs, so it cannot store a lookup table. It
> compressed *something*; the i.i.d. eval just can't tell us *what*.

### Known issues with this experiment (carried as lessons into 03/04)

1. **The committed `run.py` won't run as-is.** It imports
   `from training_loops.seq2seq import prepare_splits`, but `src/training_loops/`
   contains only a `.gitkeep` — `seq2seq.py`/`prepare_splits` were never committed.
   The 0.0033 number came from a local copy that no longer exists. Experiments
   03/04 inline their data prep to avoid this hidden dependency.
2. **`metric_for_best_model="eval_loss"` paired with `greater_is_better=True`**
   (lines ~84–85) selects the checkpoint with the *highest* eval loss — the worst
   model. HF only auto-corrects this when `greater_is_better` is left `None`. Fixed
   in 03/04 by selecting on `exact_match` (where higher really is better).

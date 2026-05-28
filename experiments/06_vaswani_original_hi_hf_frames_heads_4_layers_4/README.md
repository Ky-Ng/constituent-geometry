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

## Artifacts (HF Hub)

After training finishes, the **best checkpoint** (selected on val
exact-match) and the tokenizer are pushed to a **public** HF Hub repo whose
name matches the wandb run name:

| | |
|---|---|
| repo id (default) | `kylelovesllms/06_vaswani_original_hi_hf_frames_heads_4_layers_4_random_depth_3` |
| visibility | public |
| contents | `config.json`, `model.safetensors` (best ckpt), `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` |

The repo id is derived from `--run-name`, so changing the run name (e.g.
when swapping `--dataset` to the depth-2 build) also swaps the destination
repo — no extra flag needed. Override explicitly with `--hub-repo-id
<owner>/<name>`. Disable the push with `--no-push` (always pair with
`--no-wandb` for local smoke tests so you don't pollute either backend).

**Prerequisites** (one-time, before the first submit):

```bash
huggingface-cli login                    # interactive: paste a write-scoped token
# or in slurm/run_gpu.sbatch, export HF_TOKEN=hf_xxx before `uv run`
```

The token needs *write* scope. The Hub repo is created on first push, so no
manual setup on huggingface.co is required.

To later load the pushed checkpoint:

```python
from architecture.modeling_vaswani import VaswaniForConditionalGeneration
from transformers import AutoTokenizer
repo = "kylelovesllms/06_vaswani_original_hi_hf_frames_heads_4_layers_4_random_depth_3"
model = VaswaniForConditionalGeneration.from_pretrained(repo)
tok = AutoTokenizer.from_pretrained(repo)
```

## Setup

```bash
# Local smoke test (1 epoch, no wandb, no Hub push)
uv run python experiments/06_vaswani_original_hi_hf_frames_heads_4_layers_4/run.py \
    --epochs 1 --no-wandb --no-push

# Cluster (high-priority queue, single GPU, full 30-epoch run + Hub push)
sbatch slurm/run_gpu.sbatch \
    experiments/06_vaswani_original_hi_hf_frames_heads_4_layers_4/run.py

# Point at a different frame dataset (e.g. depth-2 random) -- the run name
# update is what re-routes the Hub push to a depth-2-specific repo.
sbatch slurm/run_gpu.sbatch \
    experiments/06_vaswani_original_hi_hf_frames_heads_4_layers_4/run.py \
    --dataset kylelovesllms/hi_hf_frames_d2_random_100 \
    --run-name 06_vaswani_original_hi_hf_frames_heads_4_layers_4_random_depth_2
```

## Results

### 06_vaswani_original_hi_hf_frames_heads_4_layers_4_random_depth_3

Wandb link: <https://wandb.ai/kgng-usc/constituent-geometry/runs/dzb85qxg>

Huggingface Repo Link: <https://huggingface.co/kylelovesllms/06_vaswani_original_hi_hf_frames_heads_4_layers_4_random_depth_3>

Summary of training results (train and validation loss and final evaluation metric):

- **Dataset**: `kylelovesllms/hi_hf_frames_d3_random_100` — train 7,200 / val 864 / test 900 rows (split by *frame*, so train and val/test share no parse skeletons).
- **Model**: `d_model=128, n_heads=4, n_layers=4, d_ff=512` → **1,858,816 params**.
- **Train loss**: 0.3354 averaged over the full run; last logged step ≈ 0.006 (smoothed; see wandb for the curve).
- **Validation** (in-loop, every 200 steps): exact-match first hit **1.0 at epoch 7.08** (eval_loss 0.0465) and stayed at 1.0 for the remainder. Final epoch-30 val: `eval_loss=0.00199, eval_exact_match=1.0`.
- **Test (held-out frames)**: `test_loss=0.0441, test_exact_match=1.0`.
- **Wall-clock**: `train_runtime=97s`, total SLURM elapsed `2:27` on a single A6000 (`c06-02`, `nlp_hiprio`).

Command:

```
sbatch slurm/run_gpu.sbatch \
experiments/06_vaswani_original_hi_hf_frames_heads_4_layers_4/run.py \
--tokenizer experiments/06_vaswani_original_hi_hf_frames_heads_4_layers_4/artifacts/tokenizer \
--dataset kylelovesllms/hi_hf_frames_d3_random_100 \
--run-name 06_vaswani_original_hi_hf_frames_heads_4_layers_4_random_depth_3
```

### 06_vaswani_original_hi_hf_frames_d3_100_heldoutdepth_3

Wandb link: <https://wandb.ai/kgng-usc/constituent-geometry/runs/u9ljm1ij>

Huggingface Repo Link: <https://huggingface.co/kylelovesllms/06_vaswani_original_hi_hf_frames_d3_100_heldoutdepth_3>

Summary of training results (train and validation loss and final evaluation metric):

- **Dataset**: `kylelovesllms/hi_hf_frames_d3_100_heldoutdepth_3` — train 4,164 / val 2,400 / test 2,400 rows. Split policy = `held_out_depth`: train sees frames at depths 0–2 only; val and test are 50/50 splits of **depth-3 frames** the model never saw at training time.
- **Model**: `d_model=128, n_heads=4, n_layers=4, d_ff=512` → **1,858,816 params** (same as the random-split run above).
- **Train loss**: 0.6301 averaged over the full run (vs 0.335 on the random split — the model fits the depth-{0,1,2} pool, but slower / noisier).
- **Validation** (in-loop, every 200 steps, on held-out **depth-3** frames):
  - **best val_exact_match across the run = 0.1663**; final epoch-30 val: `eval_loss=1.412, eval_exact_match=0.0588`.
  - Val curve never plateaus near 1 — fluctuates between ~0.01 and ~0.17. `load_best_model_at_end=True` selected the 0.1663 checkpoint for the test eval below.
- **Test (held-out depth-3 frames, never tuned on)**: `test_loss=0.5195, test_exact_match=0.2475`.
- **Wall-clock**: `train_runtime=60s`, total SLURM elapsed `1:22` on a single A6000 (`c06-02`, `nlp_hiprio`).

Command:

```
sbatch slurm/run_gpu.sbatch \
experiments/06_vaswani_original_hi_hf_frames_heads_4_layers_4/run.py \
--tokenizer experiments/06_vaswani_original_hi_hf_frames_heads_4_layers_4/artifacts/tokenizer \
--dataset kylelovesllms/hi_hf_frames_d3_100_heldoutdepth_3 \
--run-name 06_vaswani_original_hi_hf_frames_d3_100_heldoutdepth_3
```

### 06_vaswani_original_hi_hf_frames_d4_100_heldoutdepth_4

Wandb link: <https://wandb.ai/kgng-usc/constituent-geometry/runs/yrya9rsn>

Huggingface Repo Link: <https://huggingface.co/kylelovesllms/06_vaswani_original_hi_hf_frames_d4_100_heldoutdepth_4>

Summary of training results (train and validation loss and final evaluation metric):

- **Dataset**: `kylelovesllms/hi_hf_frames_d4_100_heldoutdepth_4` — train 8,964 / val 4,800 / test 4,800 rows. Split policy = `held_out_depth` at max-depth 4: train sees frames at depths 0–3; val and test are 50/50 splits of **depth-4 frames**.
- **Model**: `d_model=128, n_heads=4, n_layers=4, d_ff=512` → **1,858,816 params**.
- **Train loss**: 0.3024 averaged over the full run.
- **Validation** (in-loop, every 200 steps, on held-out **depth-4** frames):
  - **best val_exact_match across the run = 0.1942**; final epoch-30 val: `eval_loss=1.226, eval_exact_match=0.1338`.
  - Same qualitative pattern as the d3 heldout run — val exact-match never approaches 1, hovering in the 0.10–0.20 band.
- **Test (held-out depth-4 frames, never tuned on)**: `test_loss=0.7006, test_exact_match=0.2158`.
- **Wall-clock**: `train_runtime=200s`, total SLURM elapsed `4:06` on a single A6000 (`c06-05`, `nlp_hiprio`).

Caveat: `max_position_embeddings=64` and the tokenizer truncates at `--max-length 64`. Some depth-4 sentences likely exceed that and are silently truncated — the model is seeing partial sentences on the long tail. If 4:1 generalization is the actual question, the next run should bump both knobs (e.g. to 128) so depth-4 sentences fit end-to-end.

Command:

```
sbatch slurm/run_gpu.sbatch \
experiments/06_vaswani_original_hi_hf_frames_heads_4_layers_4/run.py \
--tokenizer experiments/06_vaswani_original_hi_hf_frames_heads_4_layers_4/artifacts/tokenizer \
--dataset kylelovesllms/hi_hf_frames_d4_100_heldoutdepth_4 \
--run-name 06_vaswani_original_hi_hf_frames_d4_100_heldoutdepth_4
```

## Observations

_TODO._

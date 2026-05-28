# 07 — Vaswani (RoPE) on frame-based HI → HF, `n_heads=4, n_layers=4`

Direct positional-encoding ablation of [experiment 06](../06_vaswani_original_hi_hf_frames_heads_4_layers_4/README.md):
same data, same width, same shape, same optimizer/eval/Hub-push wiring — the
**only** thing that changes is that the fixed sinusoidal positional table is
replaced by RoPE applied to Q/K inside self-attention. Cross-attention is left
un-rotated (no shared position frame between encoder and decoder sequences).

## Why

Experiment 06 showed that the original Vaswani encoder-decoder solves the
`random_frame` split (test exact-match 1.0) but fails to generalize across
depth boundaries (`held_out_depth_3` test EM ≈ 0.25; `held_out_depth_4` ≈
0.22). One natural hypothesis for the depth-holdout failure is that absolute
sinusoidal PE encodes "where in the sequence" rather than "how far from my
neighbor", and that a relative scheme like RoPE — which the literature
associates with better length / structural generalization — could change the
picture. Experiment 07 replays the same three runs with RoPE so the
comparison is one-knob.

## Model

`architecture.modeling_vaswani_rope.VaswaniRoPEForConditionalGeneration`.
Config (matches 06): `d_model=128, n_heads=4, n_layers=4, d_ff=512,
dropout=0.1`. `head_dim = 128 / 4 = 32` (even, as RoPE requires).

A single `RotaryEmbedding(head_dim, max_position_embeddings)` table is shared
across every self-attention layer (encoder and decoder both rotate; the two
cross-attention layers per decoder block do *not*).

Total params should be slightly *lower* than 06 (no learned/buffered
sinusoidal table is added to the embedding pipeline, and the RoPE cos/sin
buffers are persistent but tiny). Verify on first run.

## Data

Three runs, mirroring 06:

| run-name suffix | dataset (HF Hub) | split policy |
|---|---|---|
| `random_depth_3` | `kylelovesllms/hi_hf_frames_d3_random_100` | `random_frame` — generalization to unseen frames at the *same* depths |
| `heldoutdepth_3` | `kylelovesllms/hi_hf_frames_d3_100_heldoutdepth_3` | `held_out_depth` — train depths 0-2, val/test depth 3 |
| `heldoutdepth_4` | `kylelovesllms/hi_hf_frames_d4_100_heldoutdepth_4` | `held_out_depth` — train depths 0-3, val/test depth 4 |

Tokenizer: `artifacts/tokenizer/` (copied from experiment 06's artifacts so
this experiment is self-contained per the project convention).

## Tracking

- wandb project: `constituent-geometry`
- wandb run name: derived from `--run-name`

## Artifacts (HF Hub)

After training, the **best checkpoint** (selected on val exact-match) plus
the tokenizer are pushed to a **public** HF Hub repo whose name matches
`--run-name`, default `kylelovesllms/<run-name>`. Override with
`--hub-repo-id`; disable with `--no-push`.

**Prerequisites** (one-time):

```bash
huggingface-cli login                    # paste a write-scoped token
# or in the sbatch script: export HF_TOKEN=hf_xxx
```

## Setup

```bash
# Local smoke test (1 epoch, no wandb, no Hub push)
uv run python experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/run.py \
    --epochs 1 --no-wandb --no-push
```

### Run 1 — random-frame split, depth 3

```bash
sbatch slurm/run_gpu.sbatch \
    experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/run.py \
    --tokenizer experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/artifacts/tokenizer \
    --dataset kylelovesllms/hi_hf_frames_d3_random_100 \
    --run-name 07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4_random_depth_3
```

### Run 2 — held-out depth 3

```bash
sbatch slurm/run_gpu.sbatch \
    experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/run.py \
    --tokenizer experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/artifacts/tokenizer \
    --dataset kylelovesllms/hi_hf_frames_d3_100_heldoutdepth_3 \
    --run-name 07_vaswani_RoPE_hi_hf_frames_d3_100_heldoutdepth_3
```

### Run 3 — held-out depth 4

```bash
sbatch slurm/run_gpu.sbatch \
    experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/run.py \
    --tokenizer experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/artifacts/tokenizer \
    --dataset kylelovesllms/hi_hf_frames_d4_100_heldoutdepth_4 \
    --run-name 07_vaswani_RoPE_hi_hf_frames_d4_100_heldoutdepth_4
```

Caveat carried over from 06's depth-4 run: `max_position_embeddings=64` and
`--max-length 64` will silently truncate the long tail of depth-4 sentences.
If 4:1 generalization is the actual question, bump both to ~128 (and rebuild
the RoPE table — done automatically because it's keyed off
`max_position_embeddings`).

## Results

### 07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4_random_depth_3

Wandb link: <https://wandb.ai/kgng-usc/constituent-geometry/runs/32f2wefq>

Huggingface Repo Link: <https://huggingface.co/kylelovesllms/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4_random_depth_3>

Summary of training results (train and validation loss and final evaluation metric):

- **Dataset**: `kylelovesllms/hi_hf_frames_d3_random_100` — train 7,200 / val 864 / test 900 rows (split by *frame*; train and val/test share no parse skeletons).
- **Model**: `d_model=128, n_heads=4, n_layers=4, d_ff=512` → **1,858,816 params** (same as 06; RoPE adds no learnable params).
- **Train loss**: 0.2683 averaged over the full run.
- **Validation** (in-loop, every 200 steps): exact-match first hit **1.0 at epoch 5.31** (eval_loss 0.04315) and stayed at 1.0 for the remainder. Final epoch-30 val: `eval_loss=0.001396, eval_exact_match=1.0`. Slightly faster to saturate than 06 (which hit 1.0 at epoch 7.08).
- **Test (held-out frames)**: `test_loss=0.0346, test_exact_match=1.0`.
- **Wall-clock**: `train_runtime=110s`, total SLURM elapsed `2:22` on a single A6000 (`c05-01`, `nlp_hiprio`).

Command:

```
sbatch slurm/run_gpu.sbatch \
experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/run.py \
--tokenizer experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/artifacts/tokenizer \
--dataset kylelovesllms/hi_hf_frames_d3_random_100 \
--run-name 07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4_random_depth_3
```

### 07_vaswani_RoPE_hi_hf_frames_d3_100_heldoutdepth_3

Wandb link: <https://wandb.ai/kgng-usc/constituent-geometry/runs/m2zzd8zh>

Huggingface Repo Link: <https://huggingface.co/kylelovesllms/07_vaswani_RoPE_hi_hf_frames_d3_100_heldoutdepth_3>

Summary of training results (train and validation loss and final evaluation metric):

- **Dataset**: `kylelovesllms/hi_hf_frames_d3_100_heldoutdepth_3` — train 4,164 / val 2,400 / test 2,400 rows. Split policy = `held_out_depth`: train sees frames at depths 0–2 only; val and test are 50/50 splits of **depth-3 frames** the model never saw at training time.
- **Model**: `d_model=128, n_heads=4, n_layers=4, d_ff=512` → **1,858,816 params**.
- **Train loss**: 0.4253 averaged over the full run (vs 06's 0.6301 on the same split — RoPE fits the depth-{0,1,2} training pool *better*, but that doesn't translate to held-out depth-3).
- **Validation** (in-loop, every 200 steps, on held-out **depth-3** frames):
  - **eval_exact_match was 0.0 at every single in-loop evaluation** (10/10 evals) — never solved a single held-out depth-3 example.
  - Best `eval_loss=0.8943` at epoch 6.061; from there the val loss drifts upward to ~1.3–1.5 by epoch 30. Final: `eval_loss=1.423, eval_exact_match=0.0`.
- **Test (held-out depth-3 frames, never tuned on)**: `test_loss=1.909, test_exact_match=0.0`.
- **Wall-clock**: `train_runtime=72s`, total SLURM elapsed `1:46` on a single A6000 (`c05-01`, `nlp_hiprio`).

Note on best-model selection: with `metric_for_best_model="exact_match"` + `greater_is_better=True` and every checkpoint tied at 0.0, HF Trainer ties on the first qualifying checkpoint. The "best" model used for the test eval is therefore effectively the first eval checkpoint (step 200, ~3 epochs in). Even on a hypothetical later checkpoint with the lowest eval_loss (0.8943 @ ep 6.06), test_loss would still be much worse than 06's (06: 0.5195) — selection is not the load-bearing issue here.

Command:

```
sbatch slurm/run_gpu.sbatch \
experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/run.py \
--tokenizer experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/artifacts/tokenizer \
--dataset kylelovesllms/hi_hf_frames_d3_100_heldoutdepth_3 \
--run-name 07_vaswani_RoPE_hi_hf_frames_d3_100_heldoutdepth_3
```

### 07_vaswani_RoPE_hi_hf_frames_d4_100_heldoutdepth_4

Wandb link: <https://wandb.ai/kgng-usc/constituent-geometry/runs/0ws7mj62>

Huggingface Repo Link: <https://huggingface.co/kylelovesllms/07_vaswani_RoPE_hi_hf_frames_d4_100_heldoutdepth_4>

Summary of training results (train and validation loss and final evaluation metric):

- **Dataset**: `kylelovesllms/hi_hf_frames_d4_100_heldoutdepth_4` — train 8,964 / val 4,800 / test 4,800 rows. Split policy = `held_out_depth` at max-depth 4: train sees frames at depths 0–3; val and test are 50/50 splits of **depth-4 frames**.
- **Model**: `d_model=128, n_heads=4, n_layers=4, d_ff=512` → **1,858,816 params**.
- **Train loss**: 0.2131 averaged over the full run (vs 06's 0.3024 — same "fits training pool better" pattern as the d3 heldout).
- **Validation** (in-loop, every 200 steps, on held-out **depth-4** frames):
  - **eval_exact_match was 0.0 at every single in-loop evaluation** (22/22 evals) — same total failure pattern as the d3 heldout.
  - Best `eval_loss=0.8697` at epoch 2.837 (very early), then noisy in the 0.9–1.5 band. Final: `eval_loss=1.323, eval_exact_match=0.0`.
- **Test (held-out depth-4 frames, never tuned on)**: `test_loss=2.055, test_exact_match=0.0`.
- **Wall-clock**: `train_runtime=208s`, total SLURM elapsed `3:56` on a single A6000 (`c05-02`, `nlp_hiprio`).

Caveat carried over from 06: `max_position_embeddings=64` and `--max-length 64` silently truncate the long tail of depth-4 sentences. But this affects both 06 and 07 identically, so it doesn't explain the gap between them (06=0.2158 vs 07=0.0).

Command:

```
sbatch slurm/run_gpu.sbatch \
experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/run.py \
--tokenizer experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/artifacts/tokenizer \
--dataset kylelovesllms/hi_hf_frames_d4_100_heldoutdepth_4 \
--run-name 07_vaswani_RoPE_hi_hf_frames_d4_100_heldoutdepth_4
```

## Observations

Headline comparison vs experiment 06 (sinusoidal absolute PE, otherwise identical):

| Split | 06 test EM | 07 test EM | Δ |
|---|---|---|---|
| `random_depth_3` | 1.0 | **1.0** | tied |
| `heldoutdepth_3` | 0.2475 | **0.0** | RoPE worse by 0.25 |
| `heldoutdepth_4` | 0.2158 | **0.0** | RoPE worse by 0.22 |

Three things to note:

1. **Random-frame is a non-discriminator at this scale.** Both PE schemes hit EM=1.0 on `random_depth_3` with room to spare. If you want this experiment to *distinguish* PE schemes, the held-out-depth splits are where the signal lives.
2. **RoPE does *not* help depth generalization here — it actively hurts it.** This cuts against the standard intuition that relative-position encodings extrapolate better than absolute ones. Two non-exclusive hypotheses to consider:
   - **The small EM that 06 achieved on heldout splits may have been an absolute-PE artifact, not "generalization."** With only depths 0–2 (or 0–3) seen during training, the absolute sinusoidal PE assigns specific position vectors to specific sentence lengths; some held-out depth-3 (or depth-4) sentences may happen to fall in the same length range as some training examples, letting absolute-PE-based memorization produce the right output on a small slice. RoPE has no such "length-coincidence" leverage — it's strictly relative — so this leakage vanishes and EM drops to 0.
   - **RoPE's relative bias may be a worse inductive prior than absolute PE for this specific symbolic-transduction task at this scale.** The HI→HF map is deterministic and structural; the model needs to learn a constituent-shuffling rule, not extrapolate to longer sequences. RoPE's "rotate Q/K by absolute position so attention scores depend on relative position" optimization may simply not align with what the model needs to encode.
3. **Train-loss inversion.** RoPE achieves *lower* train loss than 06 on both heldout splits (d3: 0.43 vs 0.63; d4: 0.21 vs 0.30) — i.e. RoPE fits the depths it has seen *better* than 06, but generalizes to unseen depths *worse*. This is the textbook signature of a model with a more flexible/expressive PE scheme overfitting harder to the training-depth distribution.

The next experiment to disambiguate hypothesis 2.1 vs 2.2 would be a length-controlled probe: re-run both 06 and 07 with depth-3 holdout, but additionally filter the held-out depth-3 test set to *only* sentences whose length falls inside the train-set length range (i.e., depths-{0,1,2} length envelope). If 06's 0.2475 collapses to ~0 once that overlap is removed, the small win was a length-leakage artifact (hypothesis 2.1). If 06 still beats 07 on the length-controlled subset, RoPE is a genuinely worse prior here (hypothesis 2.2).

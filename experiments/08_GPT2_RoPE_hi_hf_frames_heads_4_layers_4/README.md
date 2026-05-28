# 08 — GPT2-style decoder-only + RoPE on frame-based HI → HF, `n_heads=4, n_layers=4`

**This is a new baseline, NOT a one-knob ablation against 06 or 07.** Three
things change at once vs. [experiment 07](../07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/README.md):

1. encoder-decoder → **decoder-only** causal LM.
2. cross-attention → single causal self-attention stream over the
   concatenated source+target sequence.
3. seq2seq loss on the target → **masked causal-LM loss** on tokens strictly
   after `<sot>` (inclusive of `<eos>`).

So "08 beats / loses to 07" does not isolate the effect of any single change.
Experiment 08 exists to characterize how a GPT2-style decoder-only model with
RoPE handles the frame splits on its own terms.

## Sequence format

```
input_ids = [<bos>] + hi + [<sot>] + hf + [<eos>]
labels    = [ -100, -100, ..., -100,   hf_1, ..., hf_m, <eos>]
             ^---- everything up to and including <sot> ----^
```

`<sot>` ("start of translation") is a new special token at id=4. The model's
CE loss shifts internally (position `t` predicts `labels[t+1]`), so masking
label positions `0..sot_pos` (inclusive) means the first un-masked prediction
uses input `<sot>` at position `sot_pos` to predict `hf_1` at position
`sot_pos + 1`. The final `<eos>` is unmasked so the model learns when to halt.

Example, mirroring the user prompt:

```
"Robin hates that the dog chases the researcher"  ->  the dog the researcher

<bos> Robin hates that the dog chases the researcher <sot> the dog the researcher <eos>
                                                       ^-- loss starts here --^
```

## Model

`architecture.modeling_gpt2_rope.GPT2RoPEForCausalLM`.

GPT2 block design: **pre-LN**, **GELU**, **biases on linears**, **final LN**,
**tied LM head**. Diverges from a vanilla GPT2 only in that the learned
absolute positional embedding is removed; positional signal comes entirely
from **RoPE applied to Q/K** in every self-attention layer (one shared cos/sin
table, same `head_dim` as 07).

Config (matches 06/07 width and shape):
`d_model=128, n_heads=4, n_layers=4, d_ff=512, dropout=0.1`,
`head_dim = 128 / 4 = 32` (even, as RoPE requires),
`max_position_embeddings=128` (bumped from 64 because the concatenated
sequence is roughly twice as long as 07's encoder-only or decoder-only
sequence).

Parameter count is reported on first run — expect it to be near 07's, give or
take the difference between (encoder + decoder stacks at depth 4 each) and (a
single decoder-only stack at depth 4).

## Data

Same three runs as 06 and 07:

| run-name suffix    | dataset (HF Hub)                                       | split policy                                              |
|--------------------|--------------------------------------------------------|-----------------------------------------------------------|
| `random_depth_3`   | `kylelovesllms/hi_hf_frames_d3_random_100`             | `random_frame` — unseen frames at the same depths         |
| `heldoutdepth_3`   | `kylelovesllms/hi_hf_frames_d3_100_heldoutdepth_3`     | `held_out_depth` — train depths 0–2, val/test depth 3     |
| `heldoutdepth_4`   | `kylelovesllms/hi_hf_frames_d4_100_heldoutdepth_4`     | `held_out_depth` — train depths 0–3, val/test depth 4     |

Tokenizer: `artifacts/tokenizer/`, built once via `build_tokenizer.py` in this
folder. **This is NOT the same tokenizer as 06/07.** It adds `<sot>` at id 4,
shifting grammar terminals to start at id 5. (Experiments 06 and 07 ship
pre-built tokenizers in their own `artifacts/`, so they remain reproducible.)

## Evaluation

For exact-match during/after training, the trainer (a custom subclass of
`Trainer`) does:

1. For each eval row, slice the prompt `[<bos>] + hi + [<sot>]` out of the
   concatenated input (everything up to and including the first un-masked
   label position).
2. Left-pad prompts within a batch (HF causal-LM generation requires left
   padding) and call `model.generate(..., do_sample=False, num_beams=1)`.
3. Strip the prompt prefix from the generated ids; that's the model's
   continuation.
4. `tokenizer.batch_decode(..., skip_special_tokens=True)` on both
   continuation and gold, then string equality.

`metric_for_best_model="exact_match"` selects the best checkpoint the same way
as 06 and 07.

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
# 0. Build the SOT-augmented tokenizer (one-time)
uv run python experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/build_tokenizer.py

# 1. Local smoke test (1 epoch, no wandb, no Hub push)
uv run python experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/run.py \
    --epochs 1 --no-wandb --no-push
```

### Run 1 — random-frame split, depth 3

```bash
sbatch slurm/run_gpu.sbatch \
    experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/run.py \
    --tokenizer experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/artifacts/tokenizer \
    --dataset kylelovesllms/hi_hf_frames_d3_random_100 \
    --run-name 08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4_random_depth_3
```

### Run 2 — held-out depth 3

```bash
sbatch slurm/run_gpu.sbatch \
    experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/run.py \
    --tokenizer experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/artifacts/tokenizer \
    --dataset kylelovesllms/hi_hf_frames_d3_100_heldoutdepth_3 \
    --run-name 08_GPT2_RoPE_hi_hf_frames_d3_100_heldoutdepth_3
```

### Run 3 — held-out depth 4

```bash
sbatch slurm/run_gpu.sbatch \
    experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/run.py \
    --tokenizer experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/artifacts/tokenizer \
    --dataset kylelovesllms/hi_hf_frames_d4_100_heldoutdepth_4 \
    --run-name 08_GPT2_RoPE_hi_hf_frames_d4_100_heldoutdepth_4
```

Caveat carried over from 06/07's depth-4 run: depth-4 sentences can run long.
The concatenated sequence here is ~2x the source/target alone, so if you see
silent right-truncation in the tokenized columns, bump `--max-length` (which
also resizes the RoPE table because the config keys off it).

## Results

Headline test exact-match (with 06 / 07 comparisons for context):

| split | 06 (Vaswani, sinusoidal) | 07 (Vaswani, RoPE) | **08 (GPT2-RoPE decoder-only)** |
|---|---|---|---|
| `random_depth_3` (unseen frames, same depths) | 1.0 | 1.0 | **0.9678** |
| `heldoutdepth_3` (train d0-2 / test d3) | ~0.25 | 0.0 | **0.0** |
| `heldoutdepth_4` (train d0-3 / test d4) | ~0.22 | 0.0 | **0.0** |

Param count is roughly half: **800,896** (decoder-only stack of 4 layers, no cross-attention) vs **1,858,816** in 06 / 07 (encoder + decoder + cross-attention). With the masked causal-LM loss only counting positions after `<sot>`, the effective "decoder-side" parameter budget is comparable, but the source-side encoder Vaswani gets in 06 / 07 is gone — the same 4-layer stack does both jobs in 08.

### 08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4_random_depth_3

Wandb link: <https://wandb.ai/kgng-usc/constituent-geometry/runs/2j6jthpr>

Huggingface Repo Link: <https://huggingface.co/kylelovesllms/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4_random_depth_3>

Summary of training results (train and validation loss and final evaluation metric):

- **Dataset**: `kylelovesllms/hi_hf_frames_d3_random_100` — train 7,200 / val 864 / test 900 rows (split by *frame*; train and val/test share no parse skeletons).
- **Model**: `d_model=128, n_heads=4, n_layers=4, d_ff=512` → **800,896 params**. Roughly half of 06 / 07's 1,858,816 — there is no encoder stack and no cross-attention.
- **Train loss**: 0.3136 averaged over the full run.
- **Validation** (in-loop, every 200 steps): exact-match first hit **1.0 at epoch 14.16** (eval_loss 0.005952). After that it dipped to 0.999 once at epoch 15.93, then stayed at 1.0 for the remainder. Final epoch-30 val: `eval_loss=0.001686, eval_exact_match=1.0`. Noticeably slower to saturate than 07 (epoch 5.31) and 06 (epoch 7.08) — the decoder-only model takes ~2-3x as many epochs to learn the random-frame mapping at the same width / shape.
- **Test (held-out frames)**: `test_loss=0.01261, test_exact_match=0.9678`. Crosses below 1.0 — about 29 / 900 test frames are wrong. 06 / 07 both saturated at 1.0.
- **Wall-clock**: `train_runtime=60s`, total SLURM elapsed `1:17` on a single A6000 (`c05-01`, `nlp_hiprio`).

Command:

```
sbatch slurm/run_gpu.sbatch \
experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/run.py \
--tokenizer experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/artifacts/tokenizer \
--dataset kylelovesllms/hi_hf_frames_d3_random_100 \
--run-name 08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4_random_depth_3
```

### 08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4_d3_100_heldoutdepth_3

Wandb link: <https://wandb.ai/kgng-usc/constituent-geometry/runs/m9cpsuo1>

Huggingface Repo Link: <https://huggingface.co/kylelovesllms/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4_d3_100_heldoutdepth_3>

Summary of training results (train and validation loss and final evaluation metric):

- **Dataset**: `kylelovesllms/hi_hf_frames_d3_100_heldoutdepth_3` — train 4,164 / val 2,400 / test 2,400 rows. Split policy = `held_out_depth`: train sees frames at depths 0–2 only; val and test are 50 / 50 splits of **depth-3 frames** the model never saw at training time.
- **Model**: `d_model=128, n_heads=4, n_layers=4, d_ff=512` → **800,896 params**.
- **Train loss**: 0.4287 averaged over the full run. Similar to 07's 0.4253 on the same split — the decoder-only model fits the depth-{0,1,2} training pool about as well as 07, even with half the parameters.
- **Validation** (in-loop, every 200 steps, on held-out **depth-3** frames):
  - **eval_exact_match was 0.0 at every single in-loop evaluation** — same total failure mode as 07.
  - Best `eval_loss=1.904` at epoch 3.03; from there val loss drifts upward to ~2.6 by epoch 30. Final: `eval_loss=2.649, eval_exact_match=0.0`. Notably worse than 07 (whose best eval_loss was 0.8943 at epoch 6.06) — the decoder-only model overfits the lower-depth pool harder.
- **Test (held-out depth-3 frames, never tuned on)**: `test_loss=1.385, test_exact_match=0.0`.
- **Wall-clock**: `train_runtime=47s`, total SLURM elapsed `1:03` on a single A6000 (`c05-01`, `nlp_hiprio`).

Selection footgun carries over from 07: with `metric_for_best_model="exact_match"` + `greater_is_better=True` and every checkpoint tied at 0.0, the "best" model used for the test eval is effectively the first eval checkpoint. Choosing the lowest-eval_loss checkpoint instead would still produce a test loss well above 06's (06: 0.5195 on the same split).

Command:

```
sbatch slurm/run_gpu.sbatch \
experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/run.py \
--tokenizer experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/artifacts/tokenizer \
--dataset kylelovesllms/hi_hf_frames_d3_100_heldoutdepth_3 \
--run-name 08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4_d3_100_heldoutdepth_3
```

### 08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4_d4_100_heldoutdepth_4

Wandb link: <https://wandb.ai/kgng-usc/constituent-geometry/runs/6lmo486s>

Huggingface Repo Link: <https://huggingface.co/kylelovesllms/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4_d4_100_heldoutdepth_4>

Summary of training results (train and validation loss and final evaluation metric):

- **Dataset**: `kylelovesllms/hi_hf_frames_d4_100_heldoutdepth_4` — train 8,964 / val 4,800 / test 4,800 rows. Split policy = `held_out_depth` at depth 4: train sees frames at depths 0–3; val and test are depth-4 frames the model never saw at training time. (Same caveat as 06 / 07: depth-4 sentences are the longest in any of the three datasets, so check that `--max-length=128` is not silently truncating the right tail before reading too much into the result.)
- **Model**: `d_model=128, n_heads=4, n_layers=4, d_ff=512` → **800,896 params**.
- **Train loss**: 0.2443 averaged over the full run.
- **Validation** (in-loop, every 200 steps, on held-out **depth-4** frames):
  - Peak `eval_exact_match = 0.0002083` (literally 1 / 4800 examples) at epoch 2.84 — indistinguishable from chance. After that, exact-match returns to 0 and stays there.
  - Best `eval_loss=1.297` at epoch 5.67; val loss drifts upward to ~1.7 by epoch 30. Final: `eval_loss=1.742, eval_exact_match=0.0`.
- **Test (held-out depth-4 frames, never tuned on)**: `test_loss=1.576, test_exact_match=0.0`.
- **Wall-clock**: `train_runtime=129s`, total SLURM elapsed `2:28` on a single A6000 (`c05-01`, `nlp_hiprio`).

Command:

```
sbatch slurm/run_gpu.sbatch \
experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/run.py \
--tokenizer experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/artifacts/tokenizer \
--dataset kylelovesllms/hi_hf_frames_d4_100_heldoutdepth_4 \
--run-name 08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4_d4_100_heldoutdepth_4
```

## Observations

**Random-frame split (in-distribution depths).** GPT2-RoPE decoder-only nearly solves it (test EM 0.9678) but doesn't quite saturate, where 06 / 07 both hit 1.0. It also takes ~2-3x as many epochs to converge on val EM (epoch 14 vs 5-7). Two confounds make this hard to attribute: (i) ~half the parameter count (800k vs 1.86M — no encoder, no cross-attention), and (ii) the optimization problem is harder (one stack has to encode the source AND decode the target through the causal bottleneck, vs 06 / 07 where the encoder gets to look at the whole source bidirectionally before the decoder generates).

**Depth-holdout splits.** Total failure — exact-match is 0.0 at every single in-loop eval on both held-out depth 3 and held-out depth 4. This is *worse* than 06 (~0.22-0.25) and matches 07's failure (0.0). So:

- The depth cliff that 07 introduced relative to 06 (collapse from ~0.22-0.25 to 0.0 on held-out depth) is *also* present in 08. RoPE does not, on its own, recover the partial generalization 06 showed.
- Switching from encoder-decoder to decoder-only on top of RoPE doesn't break things further — both 07 and 08 are at 0.0 on held-out depth — but it doesn't help either.
- Best **eval_loss** on the heldout-depth-3 split is *worse* in 08 (1.904) than in 07 (0.894), suggesting the decoder-only architecture overfits the lower-depth training pool more strongly even though the parameter budget is half.

**On the comparison being three-knob, not one-knob.** Caveat (see top of README) — 08 vs 07 changes three things at once (architecture, attention pattern, objective), so "GPT2-RoPE is worse at random-frame" or "GPT2-RoPE has the same depth cliff" cannot be attributed cleanly to any single change. The 0.0 vs 0.22 gap on heldout-depth-3 between 08 / 07 and 06 looks like a property of the RoPE setup (both 07 and 08 are at 0.0, 06 with sinusoidal is at ~0.22), but the decoder-only / encoder-decoder dimension is also varied across 06 and 08, so this is suggestive rather than conclusive.

**Truncation check.** `train: 8,964 / val: 4,800 / test: 4,800` for the depth-4 split shows all rows tokenize, but the per-row length distribution is not logged. Concatenated sequences at depth 4 can exceed `--max-length=128` quietly; a follow-up that bumps `--max-length` to e.g. 256 would falsify whether truncation is masking any real depth-4 generalization.

# 21 — Causal tracing via input-embedding ablation

## Question

> When the model predicts the **first content noun** of the HF target, which input
> tokens is that prediction actually using?

Working hypothesis: **everything past the matrix verb in the source is irrelevant.**
For the running example

```
HI (source):  the cat that a dog likes chases this researcher
HF (target):  the a dog likes that cat this researcher chases
```

the prediction of the first noun **"dog"** should not depend on the trailing
`this researcher` (they sit after the matrix verb `chases`).

## Method

This is the **zero-ablation** variant of the perturbation in the experiment brief
(`perturbed_token = 0`, positional encoding still added).

1. Tokenize HI (encoder input) and the gold HF. `labels` = HF with the leading
   `<bos>` stripped; `decoder_input_ids = shift_right(labels)` (teacher forcing).
2. Find the decoder step `t*` that **predicts** the target word. For "dog":
   `labels = [the, a, dog, ...]` so `t* = 2`, and the model predicts it *from*
   `decoder_input[2] = "a"`. Record the clean logit `L0 = logits[0, t*, id(dog)]`.
3. For each input position `i`, replace that position's **token embedding** with
   the zero vector, re-run, and record `Δ_i = logit_ablated − L0`.
   - `Δ_i ≪ 0` ⇒ token `i` was **important** (zeroing it hurt the prediction).
   - `Δ_i ≈ 0` ⇒ token `i` was **irrelevant**.
4. Two input streams are swept (`--ablate`): the **encoder** (source / HI — where
   the matrix-verb hypothesis lives) and the **decoder prefix** (`[<bos>, the, a]`
   up to `t*`; later positions can't affect step `t*` under the causal mask).

### How the embedding is zeroed

`VaswaniForConditionalGeneration.forward` has no `inputs_embeds` path, and the
encoder/decoder both *reference* the shared embedding (`model.model.shared`, also
tied to `lm_head`). `run_proposal.py` temporarily reassigns only the relevant
stack's `embed_tokens` to a wrapper that zeroes one row of the lookup output, runs
the forward, then restores it. This perturbs a single stream while leaving the
other stream and the tied head untouched, and it reproduces the model's own
embedding scaling + positional encoding exactly. (Verified: decoder logits are
bit-identical before/after an encoder ablation.)

## How to run

```bash
# copy the proposal into place (project convention), then run:
cp experiments/21_causal_tracing_ablations/run_proposal.py \
   experiments/21_causal_tracing_ablations/run.py

uv run python experiments/21_causal_tracing_ablations/run.py \
    --model experiments/16_multi_seed_withold_depth2_grok/results/vaswani/ep50_wd1.0/seed_42/best \
    --model-type vaswani \
    --hi "the cat that a dog likes chases this researcher" \
    --hf "the a dog likes that cat this researcher chases" \
    --target-word dog \
    --ablate both \
    --out-dir experiments/21_causal_tracing_ablations
```

Runs on CPU in seconds (single sentence, no training). Swap `--model` for any
Vaswani / Vaswani-RoPE checkpoint (`--model-type vaswani_rope`). Omit `--hf` to
trace the model's own greedy decode instead of the gold target. `--mode noise`
switches to the Gaussian-noise variant (`e_i += N(0, --noise-std)`, averaged over
`--noise-samples`).

### Outputs
| Path | Contents |
|---|---|
| `results/<slug>__<target>.pt`  | raw bundle (per-position Δlogit/Δprob, tokens, meta) |
| `results/<slug>__<target>.csv` | one row per ablated position |
| `figures/<slug>__<target>.png` | bar chart of Δlogit(target) vs position |

## Smoke test (mechanism check, not the official run)

A quick CPU check on the exp-16 `vaswani/seed_42/best` checkpoint, zeroing source
positions, already reproduces the hypothesis cleanly:

| ablated source token | Δ logit(dog) |
|---|---:|
| `dog` (the source noun)      | **−14.93** |
| `chases` (matrix verb)       | −0.15 |
| `researcher` (post-verb)     | +0.13 |
| `<eos>`                      | +0.09 |

i.e. the source **`dog`** is decisive, while the matrix verb and everything after
it are ~irrelevant — consistent with "tokens past the matrix verb don't matter."

## Results

_To be filled in after running `run.py` (per project convention)._

## Notes / open questions

- The checkpoint above (exp 16) trains on **depth-2-held-out**; this depth-1 rel-clause
  sentence is in-distribution for it, so "dog" is predicted confidently — good for a
  clean trace. Worth repeating on a few seeds / other splits and checking robustness.
- Decoder-prefix ablations are short here (`[<bos>, the, a]`); the interesting signal
  is on the encoder stream.
- `gpt2_rope` (decoder-only) is **not** supported yet — it concatenates HI/HF into one
  causal stream, so `t*` and the position map differ. Add a separate path if needed.

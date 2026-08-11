# 01_2_layer_simple_grammar — Observations

## Goal
First end-to-end run of the HI↔HF translation task on the toy CFG. Tests whether
a minimal GPT-2-style decoder (2 layers, 1 head, `d_model=3`, attention-only, no
FFN, untied unembed) can learn the bijection between head-initial and
head-final renderings of the same derivation tree.

## Setup

- **Data:** `data/grammar_samples.csv` (generated via
  `uv run --no-sync python src/entrypoints/generate_grammar.py --n_samples N --seed S`).
  Each row contributes two training sequences (HI→HF and HF→HI):

  ```
  <hi> s_hi <translate> s_hf <eos>
  <hf> s_hf <translate> s_hi <eos>
  ```

  `<pad>` fills up to `max_seq_len`. Loss is computed only on target-side tokens
  (positions after `<translate>`, including `<eos>`) by default — set
  `--loss_on all` to supervise everything.

- **Model:** `src/architecture/decoder.py`
  - `DecoderConfig(vocab_size=V, d_model=3, n_layers=2, n_heads=1, max_seq_len=16)`
  - Pre-LN, learned token + positional embeddings, attention-only blocks
    (no MLP), final LayerNorm, untied `Linear` unembedding.
  - `causal_lm_loss` handles the standard shift and masks loss on non-target
    positions via `loss_mask`.

- **Training:** `src/training_loops/causal_lm.py`
  - AdamW, grad clip 1.0, no weight decay, no warmup.
  - Defaults: `epochs=60`, `batch_size=128`, `lr=3e-3`.

- **Plots:** `src/visualization/training_curves.py` writes `figures/loss_curve.png`
  (log-scale CE plus val token accuracy).

## Run

```bash
# 1. Generate/regenerate data (idempotent given seed)
uv run --no-sync python src/entrypoints/generate_grammar.py --n_samples 5000 --seed 0

# 2. Train
uv run --no-sync python experiments/01_2_layer_simple_grammar/run.py
```

Outputs land in `results/history.json`, `results/model.pt`, `figures/loss_curve.png`.

## Results

See `figures/loss_curve.png` and `results/history.json`. Summary metrics
(`final_val_loss`, `final_val_accuracy`) are logged at the end of `run.py`.

## Notes

- Unique HI sentences in this grammar: 100 (intransitive) + 2500 (transitive) = 2600.
  5000 samples ≈ 2× coverage with repeats; perfect memorization is achievable.
- If val accuracy plateaus well below 1.0, the `d_model=3` bottleneck is the
  suspect — per the planning doc, scale up `d_model` / `n_layers` as needed.
- Probing (QK/OV, residual geometry) happens in follow-up experiments; this
  one just confirms the training rig works end-to-end.

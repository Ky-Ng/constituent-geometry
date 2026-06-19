"""Experiment 21 — Causal tracing via input-embedding ablation (Vaswani seq2seq).

QUESTION
--------
When the model predicts the first content noun of the HF (head-final) target,
*which input tokens does that prediction actually depend on?* The working
hypothesis (yours) is that **everything past the matrix verb in the source is
irrelevant** — e.g. for

    HI : the cat that a dog likes chases this researcher
    HF : the a dog  likes that cat this researcher chases

the prediction of the first noun "dog" should not care about the trailing
"this researcher" (they come after the matrix verb "chases").

METHOD (the perturbation in the experiment write-up, zero-ablation variant)
--------------------------------------------------------------------------
Let the decoder position that PREDICTS the target word be ``t*`` (here the model
predicts "dog" at decoder step 2 from the prefix ``[<bos>, the, a]``). We record
the clean logit

    L0 = logits[0, t*, id("dog")]

then, **one input position at a time**, replace that position's *token
embedding* with the zero vector (the positional encoding is still added, exactly
as in the pseudocode ``perturbed_token = 0`` / ``tokenized[:i] + 0 + ...``), run
the model again, and record

    Δ_i = logits_ablate_i[0, t*, id("dog")] - L0.

A large **negative** Δ_i means "zeroing token i hurt the 'dog' prediction, so
token i mattered". A Δ_i ≈ 0 means token i was irrelevant — what we expect for
the post-matrix-verb source tokens.

We ablate two input streams (selectable with --ablate):
  * encoder (the SOURCE / HI sentence)  — where the matrix-verb hypothesis lives;
  * decoder prefix (the already-emitted HF tokens ``[<bos>, the, a]`` up to t*) —
    positions after t* can't affect step t* under the causal mask, so we only
    sweep 0..t*.

WHY AN EMBEDDING SWAP (not inputs_embeds)
-----------------------------------------
``VaswaniForConditionalGeneration.forward`` only accepts ``input_ids`` — there is
no ``inputs_embeds`` path. The encoder and decoder both hold a *reference* to the
shared embedding (``model.model.shared``). We therefore temporarily reassign just
the relevant stack's ``embed_tokens`` to a thin wrapper that zeroes one row of the
lookup output, run the forward, then restore it. Reassigning ``encoder.embed_tokens``
does NOT touch ``decoder.embed_tokens`` or the tied ``lm_head`` (all still point at
``shared``), so only the chosen stream is perturbed. This reproduces the model's
own embedding scaling + positional encoding exactly.

OUTPUTS (into this experiment folder)
-------------------------------------
  results/<slug>__<target>.pt    raw bundle (diffs, logits, probs, tokens, meta)
  results/<slug>__<target>.csv   one row per ablated position
  figures/<slug>__<target>.png   Δ logit(target) vs position (bar chart)

USAGE (the concrete request: zero-ablation for "dog")
-----------------------------------------------------
    uv run python experiments/21_causal_tracing_ablations/run.py \
        --model experiments/16_multi_seed_withold_depth2_grok/results/vaswani/ep50_wd1.0/seed_42/best \
        --model-type vaswani \
        --hi "the cat that a dog likes chases this researcher" \
        --hf "the a dog likes that cat this researcher chases" \
        --target-word dog \
        --ablate both \
        --out-dir experiments/21_causal_tracing_ablations
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Model loading. Both Vaswani variants expose the SAME structure we rely on:
#   model.model.encoder.embed_tokens  /  model.model.decoder.embed_tokens
#   both referencing model.model.shared (also tied to lm_head).
# ---------------------------------------------------------------------------
def load_model(model_type: str, path: str, device: str):
    if model_type == "vaswani":
        from architecture.modeling_vaswani import VaswaniForConditionalGeneration as M
    elif model_type == "vaswani_rope":
        from architecture.modeling_vaswani_rope import (
            VaswaniRoPEForConditionalGeneration as M,
        )
    else:
        raise ValueError(
            f"model_type {model_type!r} not supported (use vaswani | vaswani_rope). "
            "gpt2_rope is decoder-only and would need a different position map."
        )
    model = M.from_pretrained(path).to(device).eval()
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok


# ---------------------------------------------------------------------------
# The ablation primitive: a wrapper that returns the base embedding lookup with
# ONE row replaced by the zero vector (mode="zero") or perturbed by Gaussian
# noise (mode="noise"). Position is in the *fed-in* sequence (token, not byte).
# ---------------------------------------------------------------------------
class _AblatedEmbedding(nn.Module):
    def __init__(self, base: nn.Embedding, pos: int, mode: str,
                 noise_std: float, generator: torch.Generator | None) -> None:
        super().__init__()
        self.base = base
        self.pos = pos
        self.mode = mode
        self.noise_std = noise_std
        self.generator = generator

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.base(input_ids).clone()          # [B, T, d_model]
        if self.mode == "zero":
            emb[:, self.pos, :] = 0.0
        else:                                        # additive Gaussian noise
            row = emb[:, self.pos, :]
            noise = torch.randn(row.shape, generator=self.generator,
                                device=row.device, dtype=row.dtype) * self.noise_std
            emb[:, self.pos, :] = row + noise
        return emb


def forward_with_ablation(model, stack: str, pos: int, *, mode: str,
                          noise_std: float, generator, **fwd) -> torch.Tensor:
    """Run a forward with ``model.model.<stack>``'s embedding row ``pos`` ablated.

    Restores the original embedding reference in a ``finally`` so a single model
    instance can be reused across every position. Returns logits [B, T_dec, V].
    """
    sub = getattr(model.model, stack)               # encoder or decoder
    original = sub.embed_tokens
    sub.embed_tokens = _AblatedEmbedding(original, pos, mode, noise_std, generator)
    try:
        with torch.no_grad():
            return model(**fwd).logits
    finally:
        sub.embed_tokens = original


# ---------------------------------------------------------------------------
# Tokenization helpers (mirror visualization/attention_heatmaps._extract_vaswani)
# ---------------------------------------------------------------------------
def build_inputs(tok, model, hi: str, hf: str | None, device: str,
                 max_new_tokens: int):
    """Return (enc_ids, enc_mask, labels, dec_input_ids) on ``device``.

    labels        = HF tokens with the leading <bos> stripped (the prediction targets)
    dec_input_ids = shift_right(labels) = [<bos>, label_0, ..., label_{-2}]  (teacher forcing)
    """
    enc = tok(hi, return_tensors="pt").to(device)
    enc_ids, enc_mask = enc["input_ids"], enc["attention_mask"]

    if hf is None:                                   # let the model pick the target
        with torch.no_grad():
            gen = model.generate(input_ids=enc_ids, attention_mask=enc_mask,
                                  max_new_tokens=max_new_tokens, num_beams=1,
                                  do_sample=False)
        labels = gen[:, 1:]                          # drop decoder_start (==<bos>)
    else:
        labels = tok(hf, return_tensors="pt").to(device)["input_ids"][:, 1:]

    dec_input_ids = labels.new_zeros(labels.shape)   # shift_tokens_right (no -100 here)
    dec_input_ids[:, 1:] = labels[:, :-1].clone()
    dec_input_ids[:, 0] = model.config.decoder_start_token_id
    return enc_ids, enc_mask, labels, dec_input_ids


def find_target_position(tok, labels: torch.Tensor, word: str, occurrence: int) -> int:
    """Index in ``labels`` of the ``occurrence``-th (1-based) appearance of ``word``."""
    tid = tok.convert_tokens_to_ids(word)
    if tid is None or tid == tok.unk_token_id:
        raise ValueError(f"target word {word!r} is not a single known token.")
    hits = (labels[0] == tid).nonzero(as_tuple=True)[0].tolist()
    if len(hits) < occurrence:
        toks = tok.convert_ids_to_tokens(labels[0].tolist())
        raise ValueError(
            f"{word!r} appears {len(hits)} time(s) in HF labels {toks}; "
            f"--target-occurrence {occurrence} is out of range."
        )
    return hits[occurrence - 1]


def _slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return s[:max_len] or "prompt"


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_trace(rows, target_word: str, hi: str, t_star_token: str, save_path: Path) -> None:
    """Bar chart of Δ logit(target) per ablated position. Encoder positions and
    decoder-prefix positions are drawn as two segments split by a dashed line."""
    import matplotlib.pyplot as plt

    enc = [r for r in rows if r["stream"] == "encoder"]
    dec = [r for r in rows if r["stream"] == "decoder"]
    ordered = enc + dec
    xs = list(range(len(ordered)))
    diffs = [r["delta_logit"] for r in ordered]
    labels = [r["token"] for r in ordered]
    colors = ["#4C72B0" if r["stream"] == "encoder" else "#C44E52" for r in ordered]

    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(ordered) + 2.0), 4.2))
    ax.bar(xs, diffs, color=colors)
    ax.axhline(0.0, color="black", lw=0.8)
    if enc and dec:                                  # separator between the two streams
        ax.axvline(len(enc) - 0.5, ls="--", lw=0.9, color="0.5")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel(f"Δ logit({target_word})  =  ablated − clean")
    ax.set_title(
        f"Causal trace of '{target_word}' (predicted at '{t_star_token}')\n"
        f"HI: {hi!r}   "
        f"[blue=encoder/source, red=decoder prefix]  more negative ⇒ more important",
        fontsize=9,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="Checkpoint dir (model + tokenizer).")
    p.add_argument("--model-type", default="vaswani",
                   choices=["vaswani", "vaswani_rope"])
    p.add_argument("--hi", required=True, help="Source (encoder) sentence.")
    p.add_argument("--hf", default=None,
                   help="Gold HF target. If omitted, the model generates one and "
                        "we trace against its own prediction.")
    p.add_argument("--target-word", default="dog",
                   help="Word whose prediction logit we trace (the 'first N').")
    p.add_argument("--target-occurrence", type=int, default=1,
                   help="Which occurrence of --target-word in HF to trace (1-based).")
    p.add_argument("--ablate", choices=["encoder", "decoder", "both"], default="both",
                   help="Which input stream(s) to sweep.")
    p.add_argument("--mode", choices=["zero", "noise"], default="zero",
                   help="zero: e_i := 0 (requested). noise: e_i += N(0, std), "
                        "averaged over --noise-samples.")
    p.add_argument("--noise-std", type=float, default=1.0, help="Std for --mode noise.")
    p.add_argument("--noise-samples", type=int, default=20, help="Samples for --mode noise.")
    p.add_argument("--noise-seed", type=int, default=0, help="RNG seed for --mode noise.")
    p.add_argument("--device", default="cpu", help="cpu | cuda.")
    p.add_argument("--out-dir", required=True,
                   help="Experiment dir; writes results/ and figures/ under it.")
    args = p.parse_args()

    model, tok = load_model(args.model_type, args.model, args.device)

    enc_ids, enc_mask, labels, dec_input_ids = build_inputs(
        tok, model, args.hi, args.hf, args.device, max_new_tokens=64
    )
    t_star = find_target_position(tok, labels, args.target_word, args.target_occurrence)
    target_id = int(labels[0, t_star])

    enc_tokens = tok.convert_ids_to_tokens(enc_ids[0].tolist())
    dec_tokens = tok.convert_ids_to_tokens(dec_input_ids[0].tolist())
    t_star_token = dec_tokens[t_star]                # the token the model predicts FROM

    fwd = dict(input_ids=enc_ids, attention_mask=enc_mask, decoder_input_ids=dec_input_ids)

    # --- clean reference logit ------------------------------------------------
    with torch.no_grad():
        clean_logits = model(**fwd).logits
    L0 = float(clean_logits[0, t_star, target_id])
    P0 = float(torch.softmax(clean_logits[0, t_star], dim=-1)[target_id])
    pred_id = int(clean_logits[0, t_star].argmax())
    print(f"[clean] target '{args.target_word}' predicted at decoder step {t_star} "
          f"(from input token '{t_star_token}')")
    print(f"[clean] logit={L0:.4f}  prob={P0:.4f}  "
          f"argmax='{tok.convert_ids_to_tokens([pred_id])[0]}' "
          f"({'CORRECT' if pred_id == target_id else 'WRONG'})")

    gen = torch.Generator(device=args.device)

    def logit_after(stack: str, pos: int) -> tuple[float, float]:
        if args.mode == "zero":
            lg = forward_with_ablation(model, stack, pos, mode="zero",
                                       noise_std=0.0, generator=None, **fwd)
            return float(lg[0, t_star, target_id]), \
                float(torch.softmax(lg[0, t_star], dim=-1)[target_id])
        vals, probs = [], []
        for s in range(args.noise_samples):          # average noise samples
            gen.manual_seed(args.noise_seed + 1000 * pos + s)
            lg = forward_with_ablation(model, stack, pos, mode="noise",
                                       noise_std=args.noise_std, generator=gen, **fwd)
            vals.append(float(lg[0, t_star, target_id]))
            probs.append(float(torch.softmax(lg[0, t_star], dim=-1)[target_id]))
        return sum(vals) / len(vals), sum(probs) / len(probs)

    rows: list[dict] = []
    if args.ablate in ("encoder", "both"):
        for i, t in enumerate(enc_tokens):           # sweep every source position
            lg, pr = logit_after("encoder", i)
            rows.append(dict(stream="encoder", pos=i, token=t,
                             logit=lg, prob=pr, delta_logit=lg - L0, delta_prob=pr - P0))
    if args.ablate in ("decoder", "both"):
        for j in range(t_star + 1):                  # only the causal prefix matters
            lg, pr = logit_after("decoder", j)
            rows.append(dict(stream="decoder", pos=j, token=dec_tokens[j],
                             logit=lg, prob=pr, delta_logit=lg - L0, delta_prob=pr - P0))

    # --- console summary ------------------------------------------------------
    print(f"\n{'stream':8} {'pos':>3} {'token':<14} {'Δlogit':>9} {'Δprob':>9}")
    for r in rows:
        print(f"{r['stream']:8} {r['pos']:>3} {r['token']:<14} "
              f"{r['delta_logit']:>9.4f} {r['delta_prob']:>9.4f}")

    # --- persist (re-plottable) ----------------------------------------------
    out_dir = Path(args.out_dir)
    results_dir, fig_dir = out_dir / "results", out_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{_slugify(args.hi)}__{args.target_word}"
    if args.mode == "noise":
        slug += f"__noise{args.noise_std}"

    torch.save(
        dict(hi=args.hi, hf=args.hf, model=args.model, model_type=args.model_type,
             target_word=args.target_word, target_occurrence=args.target_occurrence,
             t_star=t_star, t_star_token=t_star_token, target_id=target_id,
             clean_logit=L0, clean_prob=P0, mode=args.mode, rows=rows,
             enc_tokens=enc_tokens, dec_tokens=dec_tokens),
        results_dir / f"{slug}.pt",
    )
    with open(results_dir / f"{slug}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stream", "pos", "token", "logit", "prob",
                                          "delta_logit", "delta_prob"])
        w.writeheader()
        w.writerows(rows)

    plot_trace(rows, args.target_word, args.hi, t_star_token, fig_dir / f"{slug}.png")
    print(f"\nwrote results/{slug}.pt, results/{slug}.csv, figures/{slug}.png")


if __name__ == "__main__":
    main()

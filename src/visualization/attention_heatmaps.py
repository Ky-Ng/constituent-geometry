"""Per-layer attention heatmaps for a saved seq2seq / causal checkpoint.

Three model families are wired up:

* ``vaswani``      -- the Vaswani encoder-decoder with sinusoidal PE
                      (``modeling_vaswani.VaswaniForConditionalGeneration``).
                      Returns three attention families: encoder self,
                      decoder self (causal), and cross.
* ``vaswani_rope`` -- the same encoder-decoder but with rotary PE
                      (``modeling_vaswani_rope.VaswaniRoPEForConditionalGeneration``).
                      Identical forward / output_attentions API to ``vaswani``
                      (they differ only internally in how position enters
                      attention), so it reuses ``_extract_vaswani`` and returns
                      the same three families.
* ``gpt2_rope``    -- the GPT2-style decoder-only with RoPE
                      (``modeling_gpt2_rope.GPT2RoPEForCausalLM``). Returns
                      one family, ``self``, over the concatenated sequence
                      ``[<bos>] + hi + [<sot>] + hf + [<eos>]``. The ``<sot>``
                      token in the axis labels divides the HI block from the
                      HF block; the lower-left rectangle (HF rows x HI cols)
                      is the decoder-only analog of Vaswani's ``cross``.

The two Vaswani variants write the SAME ``model_type = "vaswani"`` into
config.json and are distinguished only by the ``architectures`` class name, so
we cannot auto-pick them apart -- choose with ``--model-type``.

Adding a new family is a matter of writing ``_load_<name>`` and
``_extract_<name>`` and adding an entry to ``ADAPTERS`` (or reusing an existing
``_extract_*`` when the forward API matches, as ``vaswani_rope`` does). The
plotting code reads ``AttentionBundle.groups`` generically, so it does not change.

What it renders
---------------
For each prompt we forward through the model with ``output_attentions=True``
and collect the family-specific attention tensors:

* Vaswani / Vaswani-RoPE (3 families):
    - ``encoder_self``  -- [L, H, T_enc, T_enc]  HI tokens attending to HI
    - ``decoder_self``  -- [L, H, T_dec, T_dec]  HF tokens attending to HF (causal)
    - ``cross``         -- [L, H, T_dec, T_enc]  HF tokens attending to HI

* GPT2-RoPE (1 family):
    - ``self``          -- [L, H, T_full, T_full]  full sequence, causal

For each family we dump up to two figures into ``--output-dir``:

* ``{slug}_{family}_per_head.png``  -- (n_layers x n_heads) grid of
  heatmaps, axes labelled with the surface tokens.
* ``{slug}_{family}_head_avg.png``  -- (1 x n_layers) row, averaged across
  heads. Useful as the at-a-glance "where does layer L look".

If ``--hf`` is omitted we first call ``model.generate(...)`` and then
re-forward against the generated sequence -- so the attentions reflect
what the model **actually does** on this input. Pass ``--hf`` explicitly
when you want attentions on the *gold* parse instead (e.g. comparing
in-distribution vs held-out frames on the same target).

Padding-free: one prompt per invocation. No attention-mask gymnastics,
no rows/cols of zeros from <pad>.

Caveats
-------
* ``matplotlib`` must be installed (added to pyproject as ``matplotlib>=3.10.9``).
* For Vaswani (sinusoidal), ``max_position_embeddings`` is the PE table size --
  an HI input longer than that crashes with a shape mismatch. We warn before
  forwarding. For Vaswani-RoPE the bound is the rotary table length and for
  GPT2-RoPE it applies to the *concatenated* sequence
  ``len(<bos> + hi + <sot> + hf + <eos>)``, not just HI, so the warning here is
  a conservative lower bound for those model types.
* Vaswani decoder rows are labelled with the **decoder inputs**
  (``[<bos>, hf[0], ..., hf[-2]]``), not the labels the model is trying
  to predict at that position. Row 0's attention is what the model uses
  to predict ``hf[0]`` from ``<bos>``.
* GPT2-RoPE rows are labelled with the full concatenated input (one row
  per fed-in token). The causal mask zeros the upper-right triangle, so
  every figure shows a lower-triangular pattern by construction.
* Attention weights are taken **before** the attention-dropout, and
  ``model.eval()`` disables dropout anyway -- so what you see is the raw
  softmax distribution that actually weighted V at inference time.

Usage
-----
    # In-distribution prompt against the Vaswani random-split checkpoint:
    uv run python -m visualization.attention_heatmaps \\
        --model kylelovesllms/06_vaswani_original_hi_hf_frames_heads_4_layers_4_random_depth_3 \\
        --hi "the dog sees the cat" \\
        --output-dir experiments/06_vaswani_original_hi_hf_frames_heads_4_layers_4/figures/attn_indist

    # Same prompt against the Vaswani-RoPE checkpoint (reuses the vaswani extract):
    uv run python -m visualization.attention_heatmaps \\
        --model-type vaswani_rope \\
        --model experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/results/best \\
        --hi "the dog sees the cat" \\
        --output-dir experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/figures/attn_indist

    # Same prompt against the GPT2-RoPE random-split checkpoint:
    uv run python -m visualization.attention_heatmaps \\
        --model-type gpt2_rope \\
        --model kylelovesllms/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4_random_depth_3 \\
        --hi "the dog sees the cat" \\
        --output-dir experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/figures/attn_indist

    # Held-out depth-3 (Vaswani) with the gold HF (fixed target):
    uv run python -m visualization.attention_heatmaps \\
        --model kylelovesllms/06_vaswani_original_hi_hf_frames_d3_100_heldoutdepth_3 \\
        --hi "the bird that the cat likes sees the dog" \\
        --hf "<gold-hf-sequence-here>" \\
        --output-dir experiments/06_vaswani_original_hi_hf_frames_heads_4_layers_4/figures/attn_heldout_d3_gold
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class AttentionBundle:
    """Per-prompt attention extraction result, normalized across model types.

    ``groups`` maps a family name -> stacked tensor of shape
    ``(n_layers, n_heads, n_queries, n_keys)`` on CPU.

    ``query_tokens[name]`` / ``key_tokens[name]`` are the surface tokens
    used to label the heatmap axes for family ``name``. They are kept per-
    family because cross-attention has different query / key spaces
    (decoder side / encoder side).
    """

    groups: dict[str, torch.Tensor]
    query_tokens: dict[str, list[str]]
    key_tokens: dict[str, list[str]]


# ============================================================================
# Model adapters. Each one knows how to load a checkpoint and how to forward
# a (HI, HF) pair to get the attention bundle. Add an entry to ADAPTERS to
# support a new model family.
# ============================================================================
def _load_vaswani(path: str, device: str):
    # Imported lazily so the script does not pull every architecture every run.
    from architecture.modeling_vaswani import VaswaniForConditionalGeneration

    model = VaswaniForConditionalGeneration.from_pretrained(path).to(device).eval()
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok


def _load_vaswani_rope(path: str, device: str):
    # Same encoder-decoder API as _load_vaswani; only the internal position
    # encoding differs (rotary vs sinusoidal). Because the forward / generate /
    # output_attentions surface is identical, this family reuses _extract_vaswani
    # in the ADAPTERS table -- no separate extractor needed.
    from architecture.modeling_vaswani_rope import (
        VaswaniRoPEForConditionalGeneration,
    )

    model = (
        VaswaniRoPEForConditionalGeneration.from_pretrained(path).to(device).eval()
    )
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok


def _extract_vaswani(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    hi: str,
    hf: str | None,
    device: str,
    max_new_tokens: int,
) -> AttentionBundle:
    enc = tok(hi, return_tensors="pt").to(device)  # [1, T_enc]
    enc_ids = enc["input_ids"]

    # If no gold HF: generate one so the decoder attentions correspond to
    # what the model would actually predict on this input.
    if hf is None:
        with torch.no_grad():
            gen = model.generate(
                input_ids=enc_ids,
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
            )
        # generate() returns [<bos>, ..., <eos>?]. Strip the leading
        # decoder_start_token (== <bos>) to convert into labels form, which
        # is what shift_tokens_right inside the model expects.
        hf_labels = gen[:, 1:]
    else:
        hf_tok = tok(hf, return_tensors="pt").to(device)
        # Same convention as experiments/06_.../run.py: drop the tokenizer's
        # leading <bos> so the model's shift_tokens_right does not double-
        # count it.
        hf_labels = hf_tok["input_ids"][:, 1:]

    with torch.no_grad():
        out = model(
            input_ids=enc_ids,
            attention_mask=enc["attention_mask"],
            labels=hf_labels,
            output_attentions=True,
        )

    # encoder_attentions: tuple of length L, each [1, H, T_enc, T_enc].
    # We drop the singleton batch dim and stack along a new layer dim.
    enc_self = torch.stack(out.encoder_attentions, dim=0)[:, 0]   # [L, H, Tq, Tk]
    dec_self = torch.stack(out.decoder_attentions, dim=0)[:, 0]
    cross    = torch.stack(out.cross_attentions,   dim=0)[:, 0]

    enc_tokens = tok.convert_ids_to_tokens(enc_ids[0].tolist())
    # The decoder's INPUTS are shift_right(hf_labels) = [<bos>, hf[0], ...].
    # That is what rows of decoder_self and cross correspond to.
    dec_input_ids = torch.cat(
        [
            torch.tensor([[model.config.decoder_start_token_id]], device=device),
            hf_labels[:, :-1],
        ],
        dim=1,
    )
    dec_tokens = tok.convert_ids_to_tokens(dec_input_ids[0].tolist())

    return AttentionBundle(
        groups={
            "encoder_self": enc_self.cpu(),
            "decoder_self": dec_self.cpu(),
            "cross":        cross.cpu(),
        },
        query_tokens={
            "encoder_self": enc_tokens,
            "decoder_self": dec_tokens,
            "cross":        dec_tokens,
        },
        key_tokens={
            "encoder_self": enc_tokens,
            "decoder_self": dec_tokens,
            "cross":        enc_tokens,
        },
    )


def _load_gpt2_rope(path: str, device: str):
    from architecture.modeling_gpt2_rope import GPT2RoPEForCausalLM

    model = GPT2RoPEForCausalLM.from_pretrained(path).to(device).eval()
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok


def _extract_gpt2_rope(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    hi: str,
    hf: str | None,
    device: str,
    max_new_tokens: int,
) -> AttentionBundle:
    # Mirror experiments/08_.../run.py's tokenization EXACTLY. The training
    # script splices special tokens by hand (``add_special_tokens=False`` on
    # both halves, then ``[<bos>] + hi_ids + [<sot>] + hf_ids + [<eos>]``)
    # because the joined format does not match any built-in template; if we
    # let the tokenizer auto-add specials here, we would end up with double
    # <bos> and no <sot>, and the attention pattern would be off by 1+.
    sot_id = tok.convert_tokens_to_ids("<sot>")
    if sot_id is None or sot_id == tok.unk_token_id:
        raise ValueError(
            "Tokenizer has no <sot> token. The GPT2-RoPE checkpoint expects "
            "the SOT-augmented tokenizer built by "
            "experiments/08_.../build_tokenizer.py."
        )
    bos_id = tok.bos_token_id
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id

    hi_ids = tok(hi, add_special_tokens=False)["input_ids"]
    prompt = [bos_id] + hi_ids + [sot_id]

    if hf is None:
        # Generate the continuation from the prompt; ``gen`` already includes
        # the prompt prefix, so we forward it as-is to get attentions over
        # the full sequence (no re-concatenation needed).
        prompt_t = torch.tensor([prompt], device=device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=prompt_t,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        full_ids = gen
    else:
        hf_ids = tok(hf, add_special_tokens=False)["input_ids"]
        full_ids = torch.tensor([prompt + hf_ids + [eos_id]], device=device)

    with torch.no_grad():
        out = model(input_ids=full_ids, output_attentions=True)

    # attentions: tuple of length L, each [1, H, T, T]. Causal mask zeros
    # the upper-right triangle in the softmax output -- expect a lower-
    # triangular wedge in every heatmap.
    self_attn = torch.stack(out.attentions, dim=0)[:, 0]  # [L, H, T, T]
    tokens = tok.convert_ids_to_tokens(full_ids[0].tolist())

    return AttentionBundle(
        groups={"self": self_attn.cpu()},
        query_tokens={"self": tokens},
        key_tokens={"self": tokens},
    )


ADAPTERS: dict[str, dict[str, Callable]] = {
    "vaswani":      {"load": _load_vaswani,      "extract": _extract_vaswani},
    "vaswani_rope": {"load": _load_vaswani_rope, "extract": _extract_vaswani},
    "gpt2_rope":    {"load": _load_gpt2_rope,    "extract": _extract_gpt2_rope},
}


# ============================================================================
# Plotting
# ============================================================================
def _slugify(text: str, max_len: int = 32) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return s[:max_len] or "prompt"


def _plot_per_head(
    attn: torch.Tensor,            # [L, H, Tq, Tk]
    query_tokens: list[str],
    key_tokens: list[str],
    title: str,
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    n_layers, n_heads = attn.shape[0], attn.shape[1]
    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(2.2 * n_heads + 1.5, 2.2 * n_layers + 0.8),
        squeeze=False,
    )
    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer][head]
            # vmin=0, vmax=1: attention weights are a probability distribution
            # over keys, so this is the principled fixed scale. A sharp peak
            # reads as bright yellow against deep purple -- exactly the
            # "which positions get attended to" signal we want.
            ax.imshow(attn[layer, head], cmap="viridis", aspect="auto",
                      vmin=0.0, vmax=1.0)
            if layer == 0:
                ax.set_title(f"head {head}", fontsize=9)
            if head == 0:
                ax.set_ylabel(f"layer {layer}\nquery", fontsize=9)
            ax.set_xticks(range(len(key_tokens)))
            ax.set_xticklabels(key_tokens, rotation=90, fontsize=6)
            ax.set_yticks(range(len(query_tokens)))
            ax.set_yticklabels(query_tokens, fontsize=6)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_head_avg(
    attn: torch.Tensor,            # [L, H, Tq, Tk]
    query_tokens: list[str],
    key_tokens: list[str],
    title: str,
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    n_layers = attn.shape[0]
    avg = attn.mean(dim=1)         # [L, Tq, Tk]
    fig, axes = plt.subplots(
        1, n_layers,
        figsize=(2.6 * n_layers + 1.5, 2.8),
        squeeze=False,
    )
    for layer in range(n_layers):
        ax = axes[0][layer]
        ax.imshow(avg[layer], cmap="viridis", aspect="auto",
                  vmin=0.0, vmax=1.0)
        ax.set_title(f"layer {layer}", fontsize=10)
        ax.set_xticks(range(len(key_tokens)))
        ax.set_xticklabels(key_tokens, rotation=90, fontsize=7)
        ax.set_yticks(range(len(query_tokens)))
        ax.set_yticklabels(query_tokens, fontsize=7)
    axes[0][0].set_ylabel("query", fontsize=9)
    fig.suptitle(title + " (head-averaged)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="HF Hub repo id or local checkpoint dir. Must contain "
                        "both model and tokenizer artifacts.")
    p.add_argument("--model-type", default="vaswani", choices=sorted(ADAPTERS),
                   help="Which adapter to use: 'vaswani' (sinusoidal seq2seq), "
                        "'vaswani_rope' (RoPE seq2seq), or 'gpt2_rope' "
                        "(decoder-only with RoPE).")
    p.add_argument("--hi", required=True,
                   help="HI prompt. For Vaswani this is the encoder input; "
                        "for GPT2-RoPE this is the part before <sot>.")
    p.add_argument("--hf", default=None,
                   help="HF target. If omitted, we generate one and re-forward "
                        "so the attentions reflect the model's actual "
                        "prediction. For Vaswani this becomes the decoder "
                        "input via shift_right; for GPT2-RoPE it is appended "
                        "after <sot>.")
    p.add_argument("--output-dir", required=True,
                   help="Where to dump the PNGs. Created if missing.")
    p.add_argument("--device", default="cpu",
                   help="cpu | cuda | mps. CPU is fine for single-prompt "
                        "inspection.")
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="Only used when --hf is omitted (generation cap). "
                        "Mirror the experiment's --max-length default.")
    p.add_argument("--show", default="both",
                   choices=("per_head", "head_avg", "both"),
                   help="Which figure(s) to render per attention family.")
    args = p.parse_args()

    adapter = ADAPTERS[args.model_type]
    model, tok = adapter["load"](args.model, args.device)

    # PE-table bounds check. For Vaswani (sinusoidal) this is exact (the PE
    # table covers exactly the encoder input). For Vaswani-RoPE / GPT2-RoPE this
    # is a conservative lower bound -- the actual constraint is the rotary table
    # length, and for GPT2-RoPE it is on the FULL concatenated sequence
    # (<bos> + hi + <sot> + hf + <eos>), not just HI; that path will raise a
    # clear RoPE indexing error if exceeded.
    max_pos = getattr(model.config, "max_position_embeddings", None)
    n_in = len(tok(args.hi)["input_ids"])
    if max_pos is not None and n_in > max_pos:
        print(f"WARNING: HI input length {n_in} exceeds "
              f"max_position_embeddings={max_pos}; the forward will fail. "
              f"Trim the prompt or retrain with a larger PE table.")

    bundle = adapter["extract"](
        model, tok, args.hi, args.hf, args.device, args.max_new_tokens
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(args.hi)

    written: list[Path] = []
    for name, attn in bundle.groups.items():
        title = f"{args.model_type} {name}  |  HI: {args.hi!r}"
        if args.show in ("per_head", "both"):
            path = out_dir / f"{slug}_{name}_per_head.png"
            _plot_per_head(attn, bundle.query_tokens[name],
                           bundle.key_tokens[name], title, path)
            written.append(path)
        if args.show in ("head_avg", "both"):
            path = out_dir / f"{slug}_{name}_head_avg.png"
            _plot_head_avg(attn, bundle.query_tokens[name],
                           bundle.key_tokens[name], title, path)
            written.append(path)

    print(f"saved {len(written)} figure(s) into {out_dir}/  (slug={slug})")
    for path in written:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()

"""Translate a prompt through a saved Vaswani encoder-decoder checkpoint.

Loads the model AND tokenizer from the same source (an HF Hub repo id or a
local directory), runs greedy / beam decoding on one or more prompts, and
prints the input, its tokenization, and the prediction side-by-side. Designed
for quickly probing "what does this checkpoint actually output on this
sentence", e.g. for the heldout-depth-3/4 runs in experiment 06.

The output now exposes the tokenizer's view of both sides so a confusing
prediction can be debugged against the actual input ids. Warnings flag the
two failure modes that look identical to "the model is dumb" but are
actually tokenizer / generation issues:

  * <unk> tokens in the input  -- a word the closed-vocab tokenizer doesn't
    know got mapped to <unk>. The model never saw that token in training,
    so its prediction is meaningless. Listed by surface word.
  * input longer than max_position_embeddings -- the sinusoidal PE table
    is sized at construction time; an over-long input indexes past the end
    and produces NaNs / garbage. We warn before invoking the model.
  * generation did not emit <eos> within --max-new-tokens -- the printed
    prediction is a truncated prefix, not the model's actual stopping
    point. Bump --max-new-tokens and re-run if you see this.

Examples:
    # Greedy, single prompt, against a pushed checkpoint:
    uv run python -m visualization.translate \\
        --model kylelovesllms/06_vaswani_original_hi_hf_frames_d3_100_heldoutdepth_3 \\
        --prompt "the cat that the dog likes sees the bird"

    # Batch a few prompts in one call:
    uv run python -m visualization.translate \\
        --model kylelovesllms/06_vaswani_original_hi_hf_frames_heads_4_layers_4_random_depth_3 \\
        --prompt "the dog sees the cat" \\
        --prompt "the bird that the cat likes sees the dog"

    # Against a local checkpoint directory (no Hub access needed):
    uv run python -m visualization.translate \\
        --model experiments/06_vaswani_original_hi_hf_frames_heads_4_layers_4/results/best \\
        --prompt "the dog sees the cat"

Private Hub repos require `huggingface-cli login` first (same token as the
training push). Local directories work offline.

The script imports VaswaniForConditionalGeneration directly rather than going
through AutoModelForSeq2SeqLM, because the Vaswani config/model are not
registered with the HF auto classes -- relying on the explicit class avoids
"Unrecognized model type" errors at load time.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from architecture.modeling_vaswani import VaswaniForConditionalGeneration


def find_unk_words(prompt: str, vocab: dict, unk_token: str) -> list[str]:
    """Return the surface words that aren't in the WordLevel tokenizer vocab.

    The tokenizer uses Whitespace pre-tokenization, so word-to-id is 1:1
    (modulo <bos>/<eos>). Splitting on whitespace is exactly what the
    tokenizer does at encode time, so what we list here is what actually
    became <unk> in the input ids.
    """
    return [w for w in prompt.split() if w not in vocab and w != unk_token]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        required=True,
        help="HF Hub repo id (e.g. kylelovesllms/06_...) or local checkpoint dir. "
             "Must contain both model and tokenizer artifacts.",
    )
    p.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="Input sentence (HI side). Pass multiple times to batch.",
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="1 = greedy decoding; >1 = beam search",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="cpu | cuda | mps. CPU is fine for single-prompt inspection.",
    )
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = (
        VaswaniForConditionalGeneration.from_pretrained(args.model)
        .to(args.device)
        .eval()
    )

    vocab = tok.get_vocab()
    unk_token = tok.unk_token or "<unk>"
    max_pos = getattr(model.config, "max_position_embeddings", None)

    enc = tok(args.prompt, return_tensors="pt", padding=True).to(args.device)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=False,
        )

    for i, prompt in enumerate(args.prompt):
        # --- input side -----------------------------------------------------
        # Use the un-padded ids for this row by clipping to attention_mask
        # length; otherwise the printed token list ends in a wall of <pad>.
        n_in = int(enc["attention_mask"][i].sum())
        in_ids = enc["input_ids"][i, :n_in].tolist()
        in_tokens = tok.convert_ids_to_tokens(in_ids)

        # --- output side ----------------------------------------------------
        out_row = out_ids[i].tolist()
        out_tokens = tok.convert_ids_to_tokens(out_row)
        out_decoded = tok.decode(out_row, skip_special_tokens=True)
        eos_in_output = (
            tok.eos_token_id is not None and tok.eos_token_id in out_row
        )

        print(f"=== prompt {i + 1} ===")
        print(f"INPUT  : {prompt!r}")
        print(f"  ids    : {in_ids}")
        print(f"  tokens : {in_tokens}  (len {n_in})")
        print()
        print(f"OUTPUT : {out_decoded!r}")
        print(f"  ids    : {out_row}")
        print(f"  tokens : {out_tokens}  (len {len(out_row)})")
        print()

        # --- warnings -------------------------------------------------------
        warnings = []
        unk_words = find_unk_words(prompt, vocab, unk_token)
        if unk_words:
            warnings.append(
                f"<unk> tokens in input: {unk_words} -- the closed-vocab "
                f"tokenizer doesn't know these words, so the prediction is "
                f"meaningless on them."
            )
        if max_pos is not None and n_in > max_pos:
            warnings.append(
                f"input length {n_in} exceeds model.config.max_position_embeddings "
                f"({max_pos}); positional embeddings will index out of bounds."
            )
        if not eos_in_output:
            warnings.append(
                f"generation did not emit <eos> within --max-new-tokens="
                f"{args.max_new_tokens}; the printed output is a truncated prefix. "
                f"Bump --max-new-tokens and re-run if the model needed more room."
            )

        if warnings:
            for w in warnings:
                print(f"  WARNING: {w}")
        else:
            print("  (no warnings)")
        print()


if __name__ == "__main__":
    main()

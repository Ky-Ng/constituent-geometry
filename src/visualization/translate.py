"""Translate a prompt through a saved checkpoint (any of our model families).

Loads the model AND tokenizer from the same source (an HF Hub repo id or a
local directory), runs greedy / beam decoding on one or more prompts, and
prints the input, its tokenization, and the prediction side-by-side. Designed
for quickly probing "what does this checkpoint actually output on this
sentence", e.g. for the heldout-depth-3/4 runs in experiment 06.

Three model families are wired up via the ``ADAPTERS`` registry (mirroring
``attention_heatmaps.py``):

* ``vaswani``      -- the original encoder-decoder with sinusoidal PE
                      (``modeling_vaswani.VaswaniForConditionalGeneration``).
* ``vaswani_rope`` -- the same encoder-decoder but with rotary PE
                      (``modeling_vaswani_rope.VaswaniRoPEForConditionalGeneration``).
                      Identical generate() API to ``vaswani`` -- they share the
                      seq2seq translate path and differ only internally in how
                      position enters attention.
* ``gpt2_rope``    -- the GPT2-style decoder-only with RoPE
                      (``modeling_gpt2_rope.GPT2RoPEForCausalLM``). Decoder-only,
                      so it has its own translate path: we feed
                      ``[<bos>] + hi + [<sot>]`` and the prediction is the
                      continuation generated after ``<sot>``.

Pick the family with ``--model-type`` (default ``vaswani``). The two Vaswani
variants write the SAME ``model_type = "vaswani"`` into config.json and are only
distinguished by the ``architectures`` class name, so we cannot auto-pick them
apart safely -- hence the explicit flag.

The output exposes the tokenizer's view of both sides so a confusing prediction
can be debugged against the actual input ids. Warnings flag the failure modes
that look identical to "the model is dumb" but are actually tokenizer /
generation issues:

  * <unk> tokens in the input  -- a word the closed-vocab tokenizer doesn't
    know got mapped to <unk>. The model never saw that token in training,
    so its prediction is meaningless. Listed by surface word.
  * input longer than max_position_embeddings -- for vaswani the sinusoidal PE
    table is sized at construction time; an over-long input indexes past the
    end and produces NaNs / garbage. For gpt2_rope the bound is on the full
    concatenated sequence, so the HI-only check here is a conservative lower
    bound. We warn before invoking the model.
  * generation did not emit <eos> within --max-new-tokens -- the printed
    prediction is a truncated prefix, not the model's actual stopping
    point. Bump --max-new-tokens and re-run if you see this.

Examples:
    # Greedy, single prompt, against a pushed (sinusoidal) checkpoint:
    uv run python -m visualization.translate \\
        --model kylelovesllms/06_vaswani_original_hi_hf_frames_d3_100_heldoutdepth_3 \\
        --prompt "the cat that the dog likes sees the bird"

    # A RoPE checkpoint (the path that used to crash on load):
    uv run python -m visualization.translate \\
        --model-type vaswani_rope \\
        --model experiments/07_vaswani_RoPE_hi_hf_frames_heads_4_layers_4/results/best \\
        --prompt "the dog sees the cat"

    # The GPT2-RoPE decoder-only checkpoint:
    uv run python -m visualization.translate \\
        --model-type gpt2_rope \\
        --model kylelovesllms/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4_random_depth_3 \\
        --prompt "the dog sees the cat"

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

We import each model class lazily inside its loader rather than going through
AutoModelForSeq2SeqLM, because the custom configs/models are not registered with
the HF auto classes -- relying on the explicit class avoids "Unrecognized model
type" errors at load time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import torch
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class PromptResult:
    """Per-prompt translation result, normalized across model families.

    The display/warning loop in ``main`` reads only these fields, so each
    family's translate function just has to populate them however its
    generation API works.

    ``in_ids`` / ``in_tokens`` are the (un-padded) input side as the tokenizer
    sees it; ``n_in`` is their length (used for the PE-bound warning). For
    gpt2_rope the input side is the spliced ``[<bos>] + hi + [<sot>]`` prompt.

    ``out_ids`` / ``out_tokens`` / ``out_decoded`` are the prediction side --
    for seq2seq the full decoder output, for gpt2_rope only the continuation
    generated after ``<sot>`` (the prompt prefix is stripped). ``eos_in_output``
    flags whether generation actually stopped on <eos>.
    """

    in_ids: list[int]
    in_tokens: list[str]
    n_in: int
    out_ids: list[int]
    out_tokens: list[str]
    out_decoded: str
    eos_in_output: bool


# ============================================================================
# Model adapters. Each family provides a `load` (checkpoint -> (model, tok)) and
# a `translate` (prompts -> list[PromptResult]). Add an entry to ADAPTERS to
# support a new family. Both Vaswani variants share `_translate_seq2seq`; only
# the decoder-only gpt2_rope needs its own `_translate_causal`.
# ============================================================================
def _load_vaswani(path: str, device: str):
    # Imported lazily so the script does not pull every architecture every run.
    from architecture.modeling_vaswani import VaswaniForConditionalGeneration

    model = VaswaniForConditionalGeneration.from_pretrained(path).to(device).eval()
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok


def _load_vaswani_rope(path: str, device: str):
    from architecture.modeling_vaswani_rope import (
        VaswaniRoPEForConditionalGeneration,
    )

    model = (
        VaswaniRoPEForConditionalGeneration.from_pretrained(path).to(device).eval()
    )
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok


def _load_gpt2_rope(path: str, device: str):
    # NOTE: src/architecture/modeling_gpt2_rope.py does not exist yet. This
    # loader (and _translate_causal below) are wired up to mirror
    # attention_heatmaps.py so that gpt2_rope "just works" once the module
    # lands. Until then, invoking --model-type gpt2_rope raises a clear error
    # instead of an opaque ImportError.
    try:
        from architecture.modeling_gpt2_rope import GPT2RoPEForCausalLM
    except ImportError as e:
        raise SystemExit(
            "--model-type gpt2_rope requested but "
            "architecture/modeling_gpt2_rope.py is not present in this repo "
            "yet. Use --model-type vaswani or vaswani_rope, or add the GPT2-RoPE "
            f"module first. (underlying import error: {e})"
        )

    model = GPT2RoPEForCausalLM.from_pretrained(path).to(device).eval()
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok


def _translate_seq2seq(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: list[str],
    device: str,
    max_new_tokens: int,
    num_beams: int,
) -> list[PromptResult]:
    """Encoder-decoder translate: shared by vaswani and vaswani_rope.

    Batches all prompts with padding (the encoder takes an attention_mask), then
    splits the per-row results back out for display.
    """
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )

    results: list[PromptResult] = []
    for i in range(len(prompts)):
        # Clip the input row to its attention_mask length so the printed token
        # list does not end in a wall of <pad>.
        n_in = int(enc["attention_mask"][i].sum())
        in_ids = enc["input_ids"][i, :n_in].tolist()
        in_tokens = tok.convert_ids_to_tokens(in_ids)

        out_row = out_ids[i].tolist()
        out_tokens = tok.convert_ids_to_tokens(out_row)
        out_decoded = tok.decode(out_row, skip_special_tokens=True)
        eos_in_output = (
            tok.eos_token_id is not None and tok.eos_token_id in out_row
        )

        results.append(
            PromptResult(
                in_ids=in_ids,
                in_tokens=in_tokens,
                n_in=n_in,
                out_ids=out_row,
                out_tokens=out_tokens,
                out_decoded=out_decoded,
                eos_in_output=eos_in_output,
            )
        )
    return results


def _translate_causal(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: list[str],
    device: str,
    max_new_tokens: int,
    num_beams: int,
) -> list[PromptResult]:
    """Decoder-only translate: gpt2_rope.

    Mirrors experiments/08_.../run.py and attention_heatmaps.py's tokenization:
    splice ``[<bos>] + hi_ids + [<sot>]`` by hand (``add_special_tokens=False``)
    because the joined HI/HF format matches no built-in template. The prediction
    is the continuation generated AFTER ``<sot>`` -- we strip the prompt prefix
    before decoding so the output is the HF parse, not the echoed input.

    One prompt per forward (no padding) to keep the splicing unambiguous, same
    as the heatmaps script.
    """
    sot_id = tok.convert_tokens_to_ids("<sot>")
    if sot_id is None or sot_id == tok.unk_token_id:
        raise SystemExit(
            "Tokenizer has no <sot> token. The GPT2-RoPE checkpoint expects "
            "the SOT-augmented tokenizer built by "
            "experiments/08_.../build_tokenizer.py."
        )
    bos_id = tok.bos_token_id
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id

    results: list[PromptResult] = []
    for prompt in prompts:
        hi_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        prompt_ids = [bos_id] + hi_ids + [sot_id]
        prompt_t = torch.tensor([prompt_ids], device=device)

        with torch.no_grad():
            gen = model.generate(
                input_ids=prompt_t,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )

        # gen includes the prompt prefix; the prediction is everything after it.
        full_row = gen[0].tolist()
        cont_ids = full_row[len(prompt_ids):]

        in_tokens = tok.convert_ids_to_tokens(prompt_ids)
        out_tokens = tok.convert_ids_to_tokens(cont_ids)
        out_decoded = tok.decode(cont_ids, skip_special_tokens=True)
        eos_in_output = eos_id is not None and eos_id in cont_ids

        results.append(
            PromptResult(
                in_ids=prompt_ids,
                in_tokens=in_tokens,
                n_in=len(prompt_ids),
                out_ids=cont_ids,
                out_tokens=out_tokens,
                out_decoded=out_decoded,
                eos_in_output=eos_in_output,
            )
        )
    return results


ADAPTERS: dict[str, dict[str, Callable]] = {
    "vaswani":      {"load": _load_vaswani,      "translate": _translate_seq2seq},
    "vaswani_rope": {"load": _load_vaswani_rope, "translate": _translate_seq2seq},
    "gpt2_rope":    {"load": _load_gpt2_rope,    "translate": _translate_causal},
}


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
        "--model-type",
        default="vaswani",
        choices=sorted(ADAPTERS),
        help="Which model family: vaswani (sinusoidal seq2seq), vaswani_rope "
             "(RoPE seq2seq), or gpt2_rope (decoder-only with RoPE). Default "
             "vaswani.",
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

    adapter = ADAPTERS[args.model_type]
    model, tok = adapter["load"](args.model, args.device)

    vocab = tok.get_vocab()
    unk_token = tok.unk_token or "<unk>"
    max_pos = getattr(model.config, "max_position_embeddings", None)

    results = adapter["translate"](
        model, tok, args.prompt, args.device, args.max_new_tokens, args.num_beams
    )

    for i, (prompt, res) in enumerate(zip(args.prompt, results)):
        # --- input side -----------------------------------------------------
        print(f"=== prompt {i + 1} ===")
        print(f"INPUT  : {prompt!r}")
        print(f"  ids    : {res.in_ids}")
        print(f"  tokens : {res.in_tokens}  (len {res.n_in})")
        print()

        # --- output side ----------------------------------------------------
        print(f"OUTPUT : {res.out_decoded!r}")
        print(f"  ids    : {res.out_ids}")
        print(f"  tokens : {res.out_tokens}  (len {len(res.out_ids)})")
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
        if max_pos is not None and res.n_in > max_pos:
            warnings.append(
                f"input length {res.n_in} exceeds model.config.max_position_embeddings "
                f"({max_pos}); positional embeddings will index out of bounds."
            )
        if not res.eos_in_output:
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

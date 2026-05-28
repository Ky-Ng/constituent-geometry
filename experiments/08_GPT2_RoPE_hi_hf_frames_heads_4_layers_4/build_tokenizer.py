"""Build a WordLevel tokenizer that adds a <sot> token for experiment 08.

What this differs from src/architecture/build_tokenizer.py:
* Adds ``<sot>`` ("start of translation") to ``SPECIAL_TOKENS`` at id=4, between
  UNK (id=3) and the first grammar terminal. Grammar terminals shift to start
  at id=5 (e.g. "the" = 5 instead of "the" = 4). The pre-built tokenizers in
  ``experiments/06_*/artifacts/tokenizer`` and
  ``experiments/07_*/artifacts/tokenizer`` are checked in to git and are
  unaffected by this script -- 06/07 stay reproducible.
* **No ``TemplateProcessing`` post-processor.** Run.py for experiment 08
  assembles the concatenated sequence
  ``[<bos>] + hi_ids + [<sot>] + hf_ids + [<eos>]`` by hand, so an auto-wrap
  template would double-add specials.

Run:
    uv run python experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/build_tokenizer.py

The tokenizer is written to ``artifacts/tokenizer/`` inside this experiment
folder so the experiment is self-contained per the project convention.

To copy: rename to build_tokenizer.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

# Reuse cfg_vocab for the grammar terminals so this experiment cannot drift
# from the grammar in src/grammar/. We only ADD a new special token here; the
# grammar word list is untouched.
from grammar.cfg_vocab import BOS, EOS, PAD, UNK, grammar_words

SOT = "<sot>"
SPECIAL_TOKENS: list[str] = [PAD, BOS, EOS, UNK, SOT]


def build_vocab() -> dict[str, int]:
    """Token -> id map. Specials first (PAD=0, BOS=1, EOS=2, UNK=3, SOT=4),
    then the grammar terminals starting at id=5."""
    flat = SPECIAL_TOKENS + grammar_words()
    return {tok: i for i, tok in enumerate(flat)}


def build_tokenizer() -> PreTrainedTokenizerFast:
    vocab = build_vocab()

    backend = Tokenizer(WordLevel(vocab=vocab, unk_token=UNK))
    backend.pre_tokenizer = Whitespace()
    # Intentionally NO post_processor: run.py builds the concatenated sequence
    # by hand. Calling tokenizer(text, add_special_tokens=False) returns the raw
    # word ids, which is exactly what we want.

    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token=PAD,
        bos_token=BOS,
        eos_token=EOS,
        unk_token=UNK,
        # additional_special_tokens makes <sot> show up in
        # ``tokenizer.all_special_tokens`` and in ``skip_special_tokens`` during
        # decoding, which is what we want for the exact-match compare in eval.
        additional_special_tokens=[SOT],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/artifacts/tokenizer"
        ),
    )
    args = parser.parse_args()

    tokenizer = build_tokenizer()
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.out)

    # Sanity check: simulate the concatenation that run.py performs.
    hi = "the dog likes the cat"
    hf = "the dog the cat likes"
    hi_ids = tokenizer(hi, add_special_tokens=False)["input_ids"]
    hf_ids = tokenizer(hf, add_special_tokens=False)["input_ids"]
    sot_id = tokenizer.convert_tokens_to_ids(SOT)
    full = [tokenizer.bos_token_id] + hi_ids + [sot_id] + hf_ids + [tokenizer.eos_token_id]

    print(f"vocab size : {tokenizer.vocab_size}")
    print(f"saved to   : {args.out}")
    print(f"<sot> id   : {sot_id}    (expected 4)")
    print(f"sample hi  : '{hi}'  ->  {hi_ids}")
    print(f"sample hf  : '{hf}'  ->  {hf_ids}")
    print(f"concat ids : {full}")
    print(f"decode     : '{tokenizer.decode(full)}'")


if __name__ == "__main__":
    main()

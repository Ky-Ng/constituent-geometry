"""Build a word-level tokenizer for the toy CFG and save it HF-style.

Your grammar emits a small *closed* set of whole words, so a ``WordLevel`` model
is exact: every word maps to exactly one id, with no subword splitting to muddy a
later constituent-geometry analysis. We derive the vocabulary directly from
``src.grammar.generate`` so the tokenizer can never drift from the grammar.

Running this script writes a directory containing the same tokenizer artifacts a
real HF repo ships: ``tokenizer.json`` (the fast tokenizer), ``tokenizer_config.json``,
and ``special_tokens_map.json``. That directory can then be loaded with
``AutoTokenizer.from_pretrained(dir)``.

To copy: rename this file to ``build_tokenizer.py``.

Run:  uv run python -m src.architecture.build_tokenizer --out data/tokenizer
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

# vocab.py is the single source of truth for tokens, ids, and special tokens.
from grammar.grammar import BOS, EOS, PAD, UNK, token_to_id


def build_tokenizer() -> PreTrainedTokenizerFast:
    vocab = token_to_id()

    # WordLevel = whole-word lookup; anything missing falls back to <unk>.
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token=UNK))
    backend.pre_tokenizer = Whitespace()  # split on whitespace/punctuation boundaries

    # Wrap every encoded single sequence as: <bos> ...tokens... <eos>.
    # The decoder is separately primed with decoder_start_token_id (=<bos>), and
    # <eos> is the generation stop signal.
    backend.post_processor = TemplateProcessing(
        single=f"{BOS} $A {EOS}",
        pair=f"{BOS} $A {EOS} $B:1 {EOS}:1",  # unused here, but well-formed
        special_tokens=[(BOS, vocab[BOS]), (EOS, vocab[EOS])],
    )

    # Wrap the raw backend in the HF interface so it gains save_pretrained,
    # batching, padding, attention masks, etc.
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token=PAD,
        bos_token=BOS,
        eos_token=EOS,
        unk_token=UNK,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/tokenizer"))
    args = parser.parse_args()

    tokenizer = build_tokenizer()
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.out)

    # Quick sanity check on a generated-style sentence.
    sample = "the dog likes the cat"
    enc = tokenizer(sample)
    print(f"vocab size : {tokenizer.vocab_size}")
    print(f"saved to   : {args.out}")
    print(f"'{sample}' -> {enc['input_ids']}")
    print(f"round-trip -> '{tokenizer.decode(enc['input_ids'])}'")


if __name__ == "__main__":
    main()

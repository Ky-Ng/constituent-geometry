"""Build a word-level tokenizer for the v2 toy CFG (adjuncts + relative clauses).

Identical to src/architecture/build_tokenizer.py except that vocabulary comes
from grammar.v2.cfg_vocab instead of grammar.cfg_vocab.  The v2 grammar adds
adjectives and adverbs, so the resulting tokenizer has ~137 tokens vs the
58-token v1 tokenizer.

Run:
    uv run python -m src.architecture.build_tokenizer_v2 \\
        --out experiments/15_multi_seed_grammar_v2_depth_2_random/artifacts/tokenizer

To copy: rename this file to build_tokenizer_v2.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from grammar.v2.cfg_vocab import BOS, EOS, PAD, UNK, token_to_id


def build_tokenizer() -> PreTrainedTokenizerFast:
    vocab = token_to_id()
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token=UNK))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single=f"{BOS} $A {EOS}",
        pair=f"{BOS} $A {EOS} $B:1 {EOS}:1",
        special_tokens=[(BOS, vocab[BOS]), (EOS, vocab[EOS])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token=PAD,
        bos_token=BOS,
        eos_token=EOS,
        unk_token=UNK,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output directory (e.g. experiments/15_.../artifacts/tokenizer)",
    )
    args = parser.parse_args()

    tokenizer = build_tokenizer()
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.out)

    sample = "the father gracefully hugs this artist"
    enc = tokenizer(sample)
    print(f"vocab size : {tokenizer.vocab_size}")
    print(f"saved to   : {args.out}")
    print(f"'{sample}' -> {enc['input_ids']}")
    print(f"round-trip -> '{tokenizer.decode(enc['input_ids'])}'")


if __name__ == "__main__":
    main()

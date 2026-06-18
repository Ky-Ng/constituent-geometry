"""Build a WordLevel tokenizer for gpt2_rope in experiment 15.

Like experiments/08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4/build_tokenizer.py,
this adds <sot> to the special tokens so run.py can build the concatenated
sequence [<bos>] + hi + [<sot>] + hf + [<eos>] by hand.  The difference:
vocabulary comes from grammar.v2.cfg_vocab (with Adj/Adv terminals) rather
than grammar.cfg_vocab.

Run:
    uv run python experiments/15_multi_seed_grammar_v2_depth_2_random/build_tokenizer_gpt2.py

The tokenizer is written to artifacts/tokenizer_gpt2/ inside this experiment
folder so the experiment is self-contained.

To copy: rename this file to build_tokenizer_gpt2.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from grammar.v2.cfg_vocab import BOS, EOS, PAD, UNK, grammar_words

EXP_DIR = Path(__file__).resolve().parent

SOT = "<sot>"
SPECIAL_TOKENS: list[str] = [PAD, BOS, EOS, UNK, SOT]


def build_vocab() -> dict[str, int]:
    """PAD=0, BOS=1, EOS=2, UNK=3, SOT=4, then grammar terminals from id=5."""
    flat = SPECIAL_TOKENS + grammar_words()
    return {tok: i for i, tok in enumerate(flat)}


def build_tokenizer() -> PreTrainedTokenizerFast:
    vocab = build_vocab()
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token=UNK))
    backend.pre_tokenizer = Whitespace()
    # No TemplateProcessing: run.py assembles the concatenated sequence by hand.
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token=PAD,
        bos_token=BOS,
        eos_token=EOS,
        unk_token=UNK,
        additional_special_tokens=[SOT],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=EXP_DIR / "artifacts" / "tokenizer_gpt2",
    )
    args = parser.parse_args()

    tokenizer = build_tokenizer()
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.out)

    hi = "the father gracefully hugs this artist"
    hf = "this artist the father gracefully hugs"
    hi_ids = tokenizer(hi, add_special_tokens=False)["input_ids"]
    hf_ids = tokenizer(hf, add_special_tokens=False)["input_ids"]
    sot_id = tokenizer.convert_tokens_to_ids(SOT)
    full = [tokenizer.bos_token_id] + hi_ids + [sot_id] + hf_ids + [tokenizer.eos_token_id]

    print(f"vocab size : {tokenizer.vocab_size}")
    print(f"saved to   : {args.out}")
    print(f"<sot> id   : {sot_id}    (expected 4)")
    print(f"sample hi  : '{hi}'  ->  {hi_ids}")
    print(f"concat ids : {full}")
    print(f"decode     : '{tokenizer.decode(full)}'")


if __name__ == "__main__":
    main()

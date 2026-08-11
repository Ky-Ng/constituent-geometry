"""Vocabulary: special tokens + lexicon terminals, with integer ids."""
from .rules import TERMINALS

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
HI = "<hi>"
HF = "<hf>"
TRANSLATE = "<translate>"

SPECIAL_TOKENS: list[str] = [PAD, BOS, EOS, HI, HF, TRANSLATE]


def all_terminals() -> list[str]:
    return [t for cat in TERMINALS.values() for t in cat]


def build_vocab() -> tuple[list[str], dict[str, int]]:
    tokens = SPECIAL_TOKENS + all_terminals()
    return tokens, {tok: i for i, tok in enumerate(tokens)}

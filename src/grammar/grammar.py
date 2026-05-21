"""Single source of truth for the toy CFG's vocabulary.

Three consumers share this file, so it lives in one place to prevent drift:

* ``generate.py``        — samples a random word per part-of-speech (POS).
* ``build_tokenizer.py`` — flattens every word into one id space.
* analysis / plotting    — asks ``pos_of(token)`` to group points by category.

The vocabulary is stored as a POS -> words mapping (not flat lists) precisely so
that third consumer survives: a flat list cannot tell you a token's category
later, but this dict can. Special tokens live here too, since their *ids* are the
one thing the config and tokenizer must agree on.

To copy: rename this file to ``vocab.py``.
"""

from __future__ import annotations

# --- Grammar terminals, keyed by the POS label used in generate.py -----------
# The keys ("D", "NP", "V", "C", "P") are exactly the Terminal labels the
# derivation tree carries, so pos_of() returns labels that match the brackets.
VOCAB: dict[str, list[str]] = {
    "D": ["the", "a"],            # determiners
    "NP": ["dog", "cat", "boy", "girl"],  # nouns
    "V": ["likes", "believes"],   # verbs
    "C": ["that"],                # complementizers
    "P": ["to", "at"],            # prepositions
}

# --- Special tokens. Order defines id: <pad>=0, <bos>=1, <eos>=2, <unk>=3 -----
# These ids MUST match the *_token_id defaults in VaswaniConfig.
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS: list[str] = [PAD, BOS, EOS, UNK]


def grammar_words() -> list[str]:
    """Every terminal the grammar can emit, deduplicated, first-seen order."""
    flat = [w for words in VOCAB.values() for w in words]
    return list(dict.fromkeys(flat))


def all_tokens() -> list[str]:
    """Full ordered token list: special tokens first, then grammar words."""
    return SPECIAL_TOKENS + grammar_words()


def token_to_id() -> dict[str, int]:
    """The canonical token -> id map. Both the tokenizer and config defer to this."""
    return {tok: i for i, tok in enumerate(all_tokens())}


# Reverse lookup word -> POS, built once at import. Special tokens map to None.
_POS_OF: dict[str, str] = {w: pos for pos, words in VOCAB.items() for w in words}


def pos_of(token: str) -> str | None:
    """Return the POS label of a grammar word, or ``None`` for special tokens.

    Useful for coloring constituent-geometry plots by category.
    """
    return _POS_OF.get(token)

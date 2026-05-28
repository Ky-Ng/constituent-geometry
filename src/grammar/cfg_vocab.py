"""Single source of truth for the toy CFG's vocabulary.

Three consumers share this file, so it lives in one place to prevent drift:

* ``generate.py``        — samples a random word per part-of-speech (POS).
* ``build_tokenizer.py`` — flattens every word into one id space.
* analysis / plotting    — asks ``pos_of(token)`` to group points by category.

The vocabulary is stored as a POS -> words mapping (not flat lists) precisely so
that third consumer survives: a flat list cannot tell you a token's category
later, but this dict can. Special tokens live here too, since their *ids* are the
one thing the config and tokenizer must agree on.

Note on duplicates: ``likes``, ``believes``, ``hates``, ``knows`` appear in both
``V_dp`` and ``V_cp``. ``grammar_words()`` deduplicates them (first-seen order),
so each surface form gets exactly one token id. ``pos_of`` returns whichever
fine-grained POS appears LAST in VOCAB iteration order — for the four ambiguous
verbs that is ``V_cp``. If a coarse ``V`` is more useful for plotting, map the
fine label down at the call site.

To copy: rename this file to ``vocab.py``.
"""

from __future__ import annotations

# --- Grammar terminals, keyed by the POS label used in generate.py -----------
# Keys are the FINE-grained CFG categories from GRAMMAR.md. Surface bracket
# labels (what shows up in trees) are coarser:
#   D            -> [D ...]
#   NP_singular  -> [NP ...]
#   NP_proper    -> bare under [DP X] (no inner POS label)
#   V_dp / V_cp / V_intrans -> [V ...]
#   C            -> [C ...]
VOCAB: dict[str, list[str]] = {
    "D": ["the", "a", "this"],
    "NP_singular": ["dog", "cat", "boy", "girl"],
    "NP_proper": ["John", "Mary", "Iskarous", "Jia"],
    "V_dp": [
        "likes", "believes", "hates", "knows",
        "faces", "kisses", "chases", "pursues", "loves",
    ],
    "V_cp": [
        "likes", "believes", "hates", "thinks",
        "knows", "assumes", "claims",
    ],
    "V_intrans": ["swims", "dances", "sings"],
    "C": ["that"],
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
# For words appearing under multiple POS keys (e.g. ``likes`` in V_dp and V_cp),
# the LAST POS in VOCAB iteration order wins.
_POS_OF: dict[str, str] = {w: pos for pos, words in VOCAB.items() for w in words}


def pos_of(token: str) -> str | None:
    """Return the fine-grained POS label of a grammar word, or ``None`` for
    special tokens.

    Useful for coloring constituent-geometry plots by category. Returns the
    fine-grained POS (e.g. ``V_dp``, ``NP_proper``). For verbs that belong to
    both ``V_dp`` and ``V_cp`` (``likes``, ``believes``, ``hates``, ``knows``),
    this returns ``V_cp`` by iteration order — collapse to coarse ``V`` at the
    call site if you don't want to distinguish.
    """
    return _POS_OF.get(token)

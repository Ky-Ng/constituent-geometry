"""Single source of truth for the v2 toy CFG's vocabulary.

v2 changes from v1 (grammar/cfg_vocab.py):
  - ``NP_singular`` renamed to ``N_singular`` (NP_singular is now a phrasal rule,
    N_singular is the terminal).
  - ``NP_proper`` renamed to ``N_proper`` (same reason).
  - New POS: ``C_rel``, ``Adj``, ``Adv``.
  - All other entries (D, V_dp, V_cp, V_intrans, C) are unchanged.

The same consumers apply:
  * ``generate_with_frames.py``  — samples words per POS.
  * ``build_tokenizer.py``       — flattens words into one id space.
  * analysis / plotting          — uses ``pos_of`` to group tokens by category.

Note on duplicates: ``likes``, ``believes``, ``hates``, ``knows`` appear in both
``V_dp`` and ``V_cp``. ``grammar_words()`` deduplicates them (first-seen order).
``pos_of`` returns the LAST POS in VOCAB iteration order for ambiguous verbs —
that is ``V_cp``. Collapse to coarse ``V`` at the call site when needed.

To install: copy this file to ``src/grammar/v2/cfg_vocab.py``.
"""

from __future__ import annotations

VOCAB: dict[str, list[str]] = {
    "D": ["the", "a", "this"],
    # v2: renamed from NP_singular
    "N_singular": [
        "dog", "cat", "boy", "girl",
        "teacher", "student", "friend", "researcher",
        "dancer", "artist", "musician", "engineer",
        "father", "mother", "sister", "brother",
    ],
    # v2: renamed from NP_proper
    "N_proper": [
        "John", "Mary", "Iskarous", "Jia",
        "James", "Hamilton", "Betty", "Shri",
    ],
    "V_dp": [
        "likes", "believes", "hates", "knows",
        "faces", "kisses", "chases", "pursues", "loves",
        "soothes", "hugs", "consoles", "tickles", "bedazzles", "vexes",
    ],
    "V_cp": [
        "likes", "believes", "hates", "thinks",
        "knows", "assumes", "claims",
    ],
    "V_intrans": [
        "swims", "dances", "sings",
        "laughs", "smiles", "claps", "jeers", "applauds",
    ],
    "C": ["that"],
    # v2 new: relative clause complementizer (same surface form as C but distinct POS)
    "C_rel": ["that"],
    # v2 new: adjectives (AdjP -> Adj)
    "Adj": [
        "attractive", "bald", "beautiful", "chubby", "clean",
        "dazzling", "drab", "elegant", "fancy", "fit",
        "flabby", "glamorous", "gorgeous", "handsome", "magnificent",
        "muscular", "plain", "plump", "scruffy", "shapely",
        "skinny", "stocky", "unkempt", "unsightly",
        "agreeable", "ambitious", "brave", "calm", "delightful",
        "eager", "faithful", "gentle", "happy", "jolly",
        "kind", "lively", "nice", "obedient", "polite",
        "proud", "silly", "thankful", "victorious", "witty",
        "wonderful", "zealous",
        "angry", "bewildered", "clumsy", "defeated", "embarrassed",
        "fierce", "grumpy", "helpless", "itchy", "jealous",
        "lazy", "mysterious", "nervous", "obnoxious", "panicky",
        "pitiful", "repulsive", "scary", "thoughtless", "uptight",
        "worried",
    ],
    # v2 new: adverbs (AdvP -> Adv)
    "Adv": [
        "quickly", "slowly", "quietly", "loudly",
        "happily", "sadly", "eagerly", "gently",
        "fiercely", "gracefully", "carefully", "clumsily",
    ],
}

# Special tokens. Order defines id: <pad>=0, <bos>=1, <eos>=2, <unk>=3.
# These ids MUST match the *_token_id defaults in any model config.
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


# Reverse lookup word -> POS. For ambiguous words (e.g. "that" in C and C_rel,
# "likes" in V_dp and V_cp), the LAST POS in VOCAB iteration order wins.
_POS_OF: dict[str, str] = {w: pos for pos, words in VOCAB.items() for w in words}


def pos_of(token: str) -> str | None:
    """Return the fine-grained POS of a grammar word, or None for special tokens."""
    return _POS_OF.get(token)

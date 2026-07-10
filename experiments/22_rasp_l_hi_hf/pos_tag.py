"""Stage 1 of the RASP-L program: local part-of-speech disambiguation.

The flat HI string is lexically ambiguous: `that` is both C (CP_sent) and C_rel
(relative clause), and believes/hates/knows/likes are both V_dp and V_cp. Yet
the *fine* POS of every token is decidable from a tiny fixed window (prev, next-1,
next-2), so this whole stage is elementwise maps + relative-position reads --
the length-generalizing core of RASP-L. Verified 100% vs grammar_oracle.

Disambiguation rules (see LOG.md for why they are exhaustive):
  * `that`  -> C_rel iff the previous token is an N_singular, else C.
  * a verb  -> V_cp        if the next non-adverb token is `that`
               V_dp        if it is a determiner or proper name
               V_intrans   else, if the word is lexically intransitive
               V_dp        else  (a V_dp-type verb with no complement = object-gap)
"""
from __future__ import annotations
import numpy as np
from grammar.v2.cfg_vocab import VOCAB, all_tokens
import rasp_l as R

_D = set(VOCAB["D"]); _NSING = set(VOCAB["N_singular"]); _NPROP = set(VOCAB["N_proper"])
_ADJ = set(VOCAB["Adj"]); _ADV = set(VOCAB["Adv"])
_VINT = set(VOCAB["V_intrans"]); _VDP = set(VOCAB["V_dp"]); _VCP = set(VOCAB["V_cp"])

# lexical (context-free) class codes -- one fixed token-embedding lookup
LX_OTHER, LX_D, LX_NSING, LX_NPROP, LX_ADJ, LX_ADV, LX_THAT, LX_VINT, LX_VTRANS = range(9)

def _lex(w):
    if w in _D:     return LX_D
    if w in _NSING: return LX_NSING
    if w in _NPROP: return LX_NPROP
    if w in _ADJ:   return LX_ADJ
    if w in _ADV:   return LX_ADV
    if w == "that": return LX_THAT
    if w in _VINT:  return LX_VINT           # lexically intransitive-only
    if w in _VDP or w in _VCP: return LX_VTRANS  # transitive-capable (V_dp and/or V_cp)
    return LX_OTHER

_TOKENS = all_tokens()
_ID = {w: i for i, w in enumerate(_TOKENS)}
_LEX_BY_ID = np.array([_lex(w) for w in _TOKENS], dtype=int)

# fine POS codes (match grammar_oracle leaf labels)
POS = {"OTHER": 0, "D": 1, "N_singular": 2, "N_proper": 3, "Adj": 4, "Adv": 5,
       "C": 6, "C_rel": 7, "V_intrans": 8, "V_dp": 9, "V_cp": 10}
POS_NAME = {v: k for k, v in POS.items()}

def encode(hi_tokens):
    """Words -> integer ids (the transformer's input tokens)."""
    return np.array([_ID[w] for w in hi_tokens], dtype=int)

def pos_codes(ids):
    """Integer fine-POS s-op, computed with RASP-L primitives only."""
    lex = R.tok_map(ids, lambda i: int(_LEX_BY_ID[i]))       # lexical class per token (MLP/embedding)
    prev = R.read_rel(lex, -1, default=LX_OTHER)             # class of previous token (causal)
    nx1 = R.read_rel(lex, +1, default=LX_OTHER)              # class of next token
    nx2 = R.read_rel(lex, +2, default=LX_OTHER)              # class of next-next token
    # next non-adverb class = nx1 unless nx1 is an adverb, then nx2
    nna = R.seq_map(nx1, nx2, lambda a, b: b if a == LX_ADV else a)

    def classify(l, p, n):
        # l=own lex class, p=prev lex class, n=next-non-adverb lex class
        if l == LX_D:     return POS["D"]
        if l == LX_NSING: return POS["N_singular"]
        if l == LX_NPROP: return POS["N_proper"]
        if l == LX_ADJ:   return POS["Adj"]
        if l == LX_ADV:   return POS["Adv"]
        if l == LX_THAT:  return POS["C_rel"] if p == LX_NSING else POS["C"]
        if l == LX_VINT:  return POS["V_intrans"]     # lexically intransitive
        if l == LX_VTRANS:                            # transitive-capable verb
            if n == LX_THAT:  return POS["V_cp"]
            if n in (LX_D, LX_NPROP): return POS["V_dp"]
            return POS["V_dp"]                        # no complement -> object-gap V_dp
        return POS["OTHER"]

    # three-way elementwise combine (one MLP over (lex, prev, next-non-adv))
    lp = R.seq_map(lex, prev, lambda a, b: a * 16 + b)
    return R.seq_map(lp, nna, lambda ab, n: classify(ab // 16, ab % 16, n))

def pos_tags(hi_tokens):
    """Convenience: list[str] HI -> list[str] fine POS labels."""
    codes = pos_codes(encode(hi_tokens))
    return [POS_NAME[int(c)] for c in codes]

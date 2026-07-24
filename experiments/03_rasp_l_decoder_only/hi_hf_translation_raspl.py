"""Decoder-only RASP-L program for Head-Initial -> Head-Final (Zapanese) translation.

Contract (matches README.md):
  * Input is the token sequence  <bos> w1 .. wn <sep> v1 .. vk  (the HF prefix
    emitted so far; k may be 0).  The program is a next-token predictor: run it
    on the prefix, read the value at the LAST position, append, repeat.  It
    emits the HF permutation of w1..wn followed by <eos>.
  * Cross-position information flows ONLY through the unmodified vendored
    library `rasp_core/` (Apple ml-np-rasp `core.py` + `lib.py`).  Those
    primitives are causal by construction: `kqv` never exposes a non-causal
    selector and `index_select(x, idx)` yields its default whenever idx[i] > i.
    This file passes no `causal` flag anywhere because the API has none.
  * The hardcoded word -> id and word -> lexical-class tables are the token
    embedding (sanctioned by README: "hardcoded POS lexicons are allowed").
    The program reads nothing but the plain token sequence.

Why the computation lives at the generation frontier (the causality argument):
  A value stored at HI position i may only depend on tokens <= i.  But every
  interesting structural fact here — "where does the clause opened at this
  'that' end?", "how far does this verb's object extend?" — depends on tokens
  AFTER i ("John knows that Mary swims" vs "John knows that Mary chases the
  dog" agree on every prefix through "Mary", yet the spans differ).  So those
  facts simply cannot exist in HI residual streams.  They CAN exist at any
  position at or after <sep>, where the entire HI block is past context.  Every
  post-<sep> position therefore independently resolves its own output slot:

      r  = index (1-based) of the HF token this position must emit,
           computed as a count of positions since <sep>;
      src = HI position holding that token, found by walking the constituent
            tree top-down from S, narrowing one child per step;
      emit index_select(ids, src), or <eos> once r == n+1.

  The walk needs one nontrivial span oracle: the extent of a SUBJECT DP (every
  other constituent is the last child of its parent, so its right edge is
  inherited for free).  That oracle is a recursion unrolled to a fixed clause-
  nesting budget DEPTH — the O(d) "program family" the README allows.  A
  sentence nested deeper than the budget is resolved wrongly: that is the
  length-generalization wall of Zhou et al. (2023), made explicit.

Every retrieval is an `index_select` at a pointer that is unique by
construction, so every attention selector has width exactly 1 — no reliance on
`aggr_mean` averaging, and repeated words are harmless because addressing is
structural (pointer-valued), never by token identity.  Pointers are built only
from: a landmark found by content (`firsts` at <sep>), counts (`cumsum`), and
constant offsets — the sanctioned RASP-L idioms; raw `indices()` is never
touched by this file.

References:
  * Zhou et al., 2023. "What Algorithms Can Transformers Learn? A Study in
    Length Generalization." arXiv:2310.16028 (RASP-L, length generalization).
  * apple/ml-np-rasp — the vendored `rasp_core/core.py`, `rasp_core/lib.py`.
  * grammar/GRAMMAR.md and grammar/generate_with_frames.py — the v2 toy CFG
    this program inverts (view-only ground truth for the HI->HF mapping).
"""
from __future__ import annotations

import numpy as np

from rasp_core.core import full, seq_map, tok_map
from rasp_core.lib import cumsum, firsts, has_seen, index_select, where

from grammar.cfg_vocab import VOCAB, grammar_words

# ---------------------------------------------------------------------------
# Token table (the "embedding"): specials + every grammar word.
# cfg_vocab.py fixes <pad>=0 <bos>=1 <eos>=2 <unk>=3 but has no <sep>; the
# README format requires one, so it is appended as id 4 (deviation documented
# in the experiment notes).  Grammar words follow from id 5.
# ---------------------------------------------------------------------------
PAD, BOS, EOS, UNK, SEP = "<pad>", "<bos>", "<eos>", "<unk>", "<sep>"
SPECIALS = [PAD, BOS, EOS, UNK, SEP]

WORDS: list[str] = list(dict.fromkeys(grammar_words()))
W2I: dict[str, int] = {w: i for i, w in enumerate(SPECIALS + WORDS)}
I2W: dict[int, str] = {i: w for w, i in W2I.items()}

PAD_ID, BOS_ID, EOS_ID, UNK_ID, SEP_ID = (W2I[t] for t in SPECIALS)

# Lexical classes (the hardcoded POS lexicon).  A word's class is context-free;
# context-dependent role (V_dp vs V_cp, C vs C_rel) is recovered structurally.
OTHER, D, NPROP, NSING, ADJ, ADV, THAT, V = range(8)

_CLASS_OF_WORD: dict[str, int] = {}
for _pos, _cls in (
    ("D", D), ("N_proper", NPROP), ("N_singular", NSING),
    ("Adj", ADJ), ("Adv", ADV), ("C", THAT), ("C_rel", THAT),
    ("V_dp", V), ("V_cp", V), ("V_intrans", V),
):
    for _w in VOCAB[_pos]:
        _CLASS_OF_WORD.setdefault(_w, _cls)


def _class_of_id(tok_id: int) -> int:
    return _CLASS_OF_WORD.get(I2W.get(int(tok_id), ""), OTHER)


# Node kinds carried through the descent.
K_S, K_DP, K_CPREL, K_CPSENT, K_VP = 1, 2, 3, 4, 5

DEPTH = 2                    # clause-nesting budget d (the dataset is d<=2)
STEPS_PER_LEVEL = 5          # S -> DP -> CP_rel -> (DP|VP) -> ... per clause level
EXTRA_STEPS = 4


def _steps(depth: int) -> int:
    return STEPS_PER_LEVEL * (depth + 1) + EXTRA_STEPS


# ---------------------------------------------------------------------------
# Span oracle: right edge of a constituent, given its start pointer.
# Evaluated at every sequence position in parallel; the pointer (and result)
# values are only meaningful at post-<sep> positions, where every read lands in
# the past.  `budget` bounds clause nesting; the Python recursion below is the
# depth-d unrolling of the program family.
# ---------------------------------------------------------------------------

def _cls_at(cls: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    """Lexical class at pointer (width-1 causal retrieval; OTHER off the edges)."""
    return index_select(cls, ptr, default=OTHER)


def _end_dp(cls: np.ndarray, sp: np.ndarray, budget: int) -> np.ndarray:
    """End of a DP starting at sp: NPROP | D [Adj] N [CP_rel]."""
    c1 = _cls_at(cls, sp + 1)
    core_end = where(c1 == ADJ, sp + 2, sp + 1)
    end = core_end
    if budget >= 1:
        has_rc = _cls_at(cls, core_end + 1) == THAT
        end = where(has_rc, _end_cprel(cls, core_end + 1, budget), core_end)
    return where(_cls_at(cls, sp) == NPROP, sp, end)


def _end_cprel(cls: np.ndarray, q: np.ndarray, budget: int) -> np.ndarray:
    """End of a relative clause whose C_rel 'that' sits at q.

    Subject gap: that + VP.   Object gap: that + DP + V_dp (verb is last).
    """
    subj_gap = _cls_at(cls, q + 1) == V
    return where(
        subj_gap,
        _end_vp(cls, q + 1, budget - 1),
        _end_dp(cls, q + 1, budget - 1) + 1,
    )


def _end_cpsent(cls: np.ndarray, q: np.ndarray, budget: int) -> np.ndarray:
    """End of a sentential complement: that + S, S = DP VP (VP is last)."""
    dp_end = _end_dp(cls, q + 1, budget - 1)
    return _end_vp(cls, dp_end + 1, budget - 1)


def _end_vp(cls: np.ndarray, v: np.ndarray, budget: int) -> np.ndarray:
    """End of a VP headed at v: V [Adv] [DP | CP_sent]."""
    adv = _cls_at(cls, v + 1) == ADV
    nn_pos = where(adv, v + 2, v + 1)
    cn = _cls_at(cls, nn_pos)
    end = where(
        (cn == D) | (cn == NPROP),
        _end_dp(cls, nn_pos, budget),
        where(adv, v + 1, v),                       # bare/adv intransitive
    )
    if budget >= 1:
        end = where(cn == THAT, _end_cpsent(cls, nn_pos, budget), end)
    return end


# ---------------------------------------------------------------------------
# The program: one causal pass over the whole prefix.
# ---------------------------------------------------------------------------

def _snap(x) -> np.ndarray:
    """Trace-snapshot helper (also narrows the vendored ops' loose types)."""
    return np.asarray(x).copy()


def predict_tokens(ids: np.ndarray, depth: int = DEPTH, trace: list | None = None) -> np.ndarray:
    """Next-token prediction at every position (meaningful from <sep> onward).

    The value at position p >= sep_pos is the (p - sep_pos + 1)-th HF token,
    or <eos> once the permutation is exhausted.  Values before <sep> are
    unspecified filler — an autoregressive driver never reads them.

    Pass a list as `trace` to capture per-step state snapshots (introspection
    for teaching material only; it has no effect on the computation).
    """
    ids = np.asarray(ids, dtype=int)
    cls = tok_map(ids, _class_of_id)

    # --- segment geometry, as counts and one content landmark ---------------
    seen_sep = has_seen(ids, full(ids, SEP_ID))          # 1 at/after <sep>
    r = cumsum(seen_sep)                                 # 1-based output slot
    is_hi_word = seq_map(cls != OTHER, seen_sep, lambda w, sp: w and not sp)
    n = cumsum(is_hi_word)                               # |HI| once past <sep>
    sep_pos = firsts(ids, full(ids, SEP_ID), default=0)  # landmark position

    slot = r                                             # kept for the final emit

    # --- descent state: node kind, span [s, t], slot r within it ------------
    kind = full(ids, K_S)
    s = full(ids, 1)                                     # HI starts after <bos>
    t = sep_pos - 1
    done = full(ids, 0) == 1
    src = full(ids, 0)

    if trace is not None:
        trace.append({"phase": "init", "cls": _snap(cls), "seen_sep": _snap(seen_sep),
                      "slot": _snap(slot), "n": _snap(n), "sep_pos": _snap(sep_pos),
                      "kind": _snap(kind), "s": _snap(s), "t": _snap(t), "r": _snap(r),
                      "src": _snap(src), "done": _snap(done)})

    for _ in range(_steps(depth)):
        c_s1 = _cls_at(cls, s + 1)

        # K_S: S -> DP_subj VP; only place the span oracle is needed.
        dpe = _end_dp(cls, s, depth)
        ndp = dpe - s + 1
        in_dp = r <= ndp
        s_kind = where(in_dp, full(ids, K_DP), full(ids, K_VP))
        s_s = where(in_dp, s, dpe + 1)
        s_t = where(in_dp, dpe, t)
        s_r = where(in_dp, r, r - ndp)

        # K_DP: D [Adj] N [CP_rel]  ->  D  HF(CP_rel)  [Adj] N
        isprop = _cls_at(cls, s) == NPROP
        core_end = where(c_s1 == ADJ, s + 2, s + 1)
        rc_len = t - core_end                            # 0 when no relative
        dp_is_d = (~isprop) & (r == 1)
        dp_in_rc = (~isprop) & (r > 1) & (r <= 1 + rc_len)
        dp_done = isprop | dp_is_d | ~dp_in_rc
        dp_src = where(isprop | dp_is_d, s, s + (r - 1 - rc_len))
        dp_kind = full(ids, K_CPREL)
        dp_s = core_end + 1
        dp_t = t
        dp_r = r - 1

        # K_CPREL: that + S_gap  ->  HF(S_gap) + that
        size = t - s + 1
        cr_is_that = r == size
        subj_gap = c_s1 == V
        ndp_o = t - s - 1                                # object-gap subject DP
        cr_in_dp = (~cr_is_that) & (~subj_gap) & (r <= ndp_o)
        cr_is_v = (~cr_is_that) & (~subj_gap) & (r > ndp_o)
        cr_done = cr_is_that | cr_is_v
        cr_src = where(cr_is_that, s, t)
        cr_kind = where(subj_gap, full(ids, K_VP), full(ids, K_DP))
        cr_s = s + 1
        cr_t = where(subj_gap, t, t - 1)
        cr_r = r

        # K_CPSENT: that + S  ->  HF(S) + that
        cp_is_that = r == size
        cp_done = cp_is_that
        cp_src = s
        cp_s = s + 1

        # K_VP: V [Adv] [comp]  ->  HF(comp) [Adv] V
        adv = (s + 1 <= t) & (c_s1 == ADV)
        nn_pos = where(adv, s + 2, s + 1)
        has_comp = nn_pos <= t
        cn = _cls_at(cls, nn_pos)
        comp_len = t - nn_pos + 1
        vp_in_comp = has_comp & (r <= comp_len)
        vp_at_adv = adv & ((has_comp & (r == comp_len + 1)) | (~has_comp & (r == 1)))
        vp_done = ~vp_in_comp
        vp_src = where(vp_at_adv, s + 1, s)
        vp_kind = where(cn == THAT, full(ids, K_CPSENT), full(ids, K_DP))
        vp_s = nn_pos
        vp_r = r

        # ---- merge the five cases by current kind, freezing finished rows --
        def merge(vs, vdp, vcr, vcp, vvp):
            out = where(kind == K_S, vs,
                  where(kind == K_DP, vdp,
                  where(kind == K_CPREL, vcr,
                  where(kind == K_CPSENT, vcp, vvp))))
            return out

        new_done = merge(full(ids, 0), dp_done, cr_done, cp_done, vp_done) == 1
        new_src = merge(full(ids, 0), dp_src, cr_src, cp_src, vp_src)
        new_kind = merge(s_kind, dp_kind, cr_kind, full(ids, K_S), vp_kind)
        new_s = merge(s_s, dp_s, cr_s, cp_s, vp_s)
        new_t = merge(s_t, dp_t, cr_t, t, t)
        new_r = merge(s_r, dp_r, cr_r, r, vp_r)

        src = where(done, src, where(new_done, new_src, src))
        kind = where(done, kind, new_kind)
        s = where(done, s, new_s)
        t = where(done, t, new_t)
        r_next = where(done, r, new_r)
        done = done | new_done
        r = r_next

        if trace is not None:
            trace.append({"phase": "step", "dpe": _snap(dpe), "kind": _snap(kind),
                          "s": _snap(s), "t": _snap(t), "r": _snap(r),
                          "src": _snap(src), "done": _snap(done)})

    # --- emit ----------------------------------------------------------------
    out = index_select(ids, src, default=UNK_ID)
    result = where(slot == n + 1, full(ids, EOS_ID), out)
    if trace is not None:
        trace.append({"phase": "emit", "out": _snap(out), "final": _snap(result)})
    return result


# ---------------------------------------------------------------------------
# Autoregressive driver and word-level convenience wrappers.
# ---------------------------------------------------------------------------

def encode(words: list[str]) -> list[int]:
    return [W2I.get(w, UNK_ID) for w in words]


def decode(ids: list[int]) -> list[str]:
    return [I2W.get(int(i), UNK) for i in ids]


def next_token(prefix_ids: list[int], depth: int = DEPTH) -> int:
    """One decoder step: the model's prediction at the last position."""
    return int(predict_tokens(np.array(prefix_ids, dtype=int), depth=depth)[-1])


def translate(hi_words: list[str], depth: int = DEPTH) -> list[str]:
    """Greedy autoregressive rollout: <bos> hi <sep>  ->  hf ... <eos>."""
    seq = [BOS_ID] + encode(hi_words) + [SEP_ID]
    out: list[int] = []
    for _ in range(len(hi_words) + 1):
        tok = next_token(seq, depth=depth)
        if tok == EOS_ID:
            break
        out.append(tok)
        seq.append(tok)
    return decode(out)

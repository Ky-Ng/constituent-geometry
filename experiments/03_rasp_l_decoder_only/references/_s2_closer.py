# Note: This is a reference implementation for a bidirectional attention encoder-only, seq2seq algorithm using a modified RASP-L lib
"""Strategy A (closer-marking) RASP-L program for HI -> HF translation (v2 CFG, CP-depth <= 2).

Pipeline (all cross-position info flows through select/kqv/sel_width/shift only):

  hi_ids --(tok_map)--> lexical class + verb flags
         --(shift reads)--> fine POS + verb ROLE          [Stage 1, fully local]
         --(index_select relaxation)--> per-'that' clause-END e(q), phrase ENDs   [spans]
         --(sel_width stabbing)--> CP-depth[i]              [Strategy A: openers minus closes]
         --(local reads + stabbing)--> displacement disp[i] = hf_pos[i]-i
         --(kqv inverse-perm gather)--> hf_ids -> hf words  [Stage 3]

Core identities used (all verified 100% vs grammar_oracle on depth<=2):
  * HF is a permutation of HI:  out[i + disp[i]] = hi[i].
  * disp[i] = sum over enclosing FLIP nodes F of (+|comp(F)| in head-child, -|head(F)| in comp-child).
    We split that sum by flip-node TYPE into four additive pieces:
        d_adv   (verb<->adverb swaps, VP_x_adv)      -- purely local
        d_cp    (CP_sent + CP_rel: 'that' <-> clause)-- needs depth + clause span
        d_nprel (NP_singular -> [core, CP_rel])      -- needs RC span + a stabbing sum
        d_vp    (transitive/clausal VP head<->comp)  -- needs comp span + a stabbing sum
  * CP-depth[i] = #('that' at q<=i whose clause end e(q) >= i)   -- a stabbing count.
    Openers are the 'that' tokens; a clause "closes" at e(q); Strategy A recovers e(q) by
    matching (the pointer relaxation below), which also handles the multiplicity-2 closes.

Length-generalization honesty: this is a DEPTH-2-ONLY program.  The clause-END pointers are
found by a FIXED number (K) of index_select relaxation sweeps -- enough to resolve <=2 levels
of nesting, not unbounded Dyck.  Neighbour/forward reads over the HI block use causal=False
(justified: in the real 'hi <sep> hf' decoder the whole HI block is in the past).  Everything
else uses only content + relative-position predicates.
"""
from __future__ import annotations

import numpy as np

# Number of clause-END relaxation sweeps.  Each sweep == one transformer layer and
# propagates exactly one level of CP nesting, so this is the model's DEPTH BUDGET:
# sentences with CP-nesting deeper than ROUNDS are NOT resolved.  Set high (10) for
# the depth<=2 dataset; dial down to expose the length-generalization wall.
ROUNDS = 10

from grammar.v2.cfg_vocab import VOCAB, grammar_words
from rasp_l import (  # the ONLY cross-position machinery we are allowed to use
    tok_map, select, sel_width, kqv, index_select,
    indices, full, where, equals, geq,
)

# ---------------------------------------------------------------------------
# word <-> id tables (a fixed token embedding; legal as a tok_map lookup table)
# ---------------------------------------------------------------------------
_WORDS = list(dict.fromkeys(grammar_words() + ["that"]))
_W2I = {w: i for i, w in enumerate(_WORDS)}
_I2W = {i: w for w, i in _W2I.items()}

_D = set(VOCAB["D"]); _NSING = set(VOCAB["N_singular"]); _NPROP = set(VOCAB["N_proper"])
_ADJ = set(VOCAB["Adj"]); _ADV = set(VOCAB["Adv"])
_VINTR = set(VOCAB["V_intrans"]); _VDP = set(VOCAB["V_dp"]); _VCP = set(VOCAB["V_cp"])
_VERB = _VINTR | _VDP | _VCP

# lexical class codes
D, NSING, NPROP, ADJ, ADV, THAT, V, OTHER = 0, 1, 2, 3, 4, 5, 6, 7


def _lexclass(i: int) -> int:
    w = _I2W.get(int(i))
    if w in _D: return D
    if w in _NSING: return NSING
    if w in _NPROP: return NPROP
    if w in _ADJ: return ADJ
    if w in _ADV: return ADV
    if w == "that": return THAT
    if w in _VERB: return V
    return OTHER


def _is_intrans_lex(i: int) -> int:
    return 1 if _I2W.get(int(i)) in _VINTR else 0


def _read_off(x, off, default=-1):
    """out[i] = x[i+off]; a relative-position read (causal=False, HI block is in the past)."""
    return kqv(indices(x) - off, indices(x), x, equals, default=default, causal=False)


def _stab_ge(end_val):
    """Stabbing count: for query i, #keys j with j<=i (causal) and end_val[j] >= i.

    = #intervals [j, end_val[j]] that contain position i.  One counting attention head."""
    return sel_width(select(end_val, indices(end_val), geq, causal=True))


# ---------------------------------------------------------------------------
# main program
# ---------------------------------------------------------------------------
def hi_to_hf(hi_tokens: list[str]) -> list[str]:
    n = len(hi_tokens)
    if n == 0:
        return []
    hi_ids = np.array([_W2I.get(w, 0) for w in hi_tokens], dtype=int)
    idx = indices(hi_ids)
    ZERO = full(hi_ids, 0)

    # ===================== Stage 1: fine POS + verb ROLE (fully local) =====================
    lc = tok_map(hi_ids, _lexclass)                       # lexical class per token (embedding)
    is_intr = tok_map(hi_ids, _is_intrans_lex)            # lexical V_intrans membership

    prev_lc = _read_off(lc, -1)                           # lc[i-1]
    prev2_lc = _read_off(lc, -2)                          # lc[i-2]
    next_lc = _read_off(lc, +1)                           # lc[i+1]
    next2_lc = _read_off(lc, +2)                          # lc[i+2]

    isthat = (lc == THAT)                                 # opener tokens
    is_crel = isthat & (prev_lc == NSING)                 # 'that' after a common noun -> C_rel
    is_c = isthat & ~is_crel                              # else sentential C
    isverb = (lc == V)
    isadv = (lc == ADV)

    # next NON-adverb class (skip one optional adverb after a verb)
    nn_lc = where(next_lc == ADV, next2_lc, next_lc)

    # verb ROLE: 1 intrans, 2 transitive(+DP obj), 3 clausal(+CP), 4 object-gap, 0 non-verb
    role = where(
        isverb,
        where(nn_lc == THAT, full(hi_ids, 3),
        where((nn_lc == D) | (nn_lc == NPROP), full(hi_ids, 2),
        where(is_intr == 1, full(hi_ids, 1), full(hi_ids, 4)))),
        ZERO,
    )

    # ===================== spans: clause / phrase END pointers =====================
    # head-block end of a verb: verb+adverb -> i+1, else i
    hb_end = where(isverb & (next_lc == ADV), idx + 1, idx)

    # flat DP end (core-noun position), ignoring any relative clause
    dp_end_flat = where(lc == D, where(next_lc == ADJ, idx + 2, idx + 1), idx)
    rc_that_pos = dp_end_flat + 1                         # candidate C_rel 'that' after the core
    lc_at_rc = index_select(lc, rc_that_pos, default=-1, causal=False)
    dp_has_rc = (lc == D) & (lc_at_rc == THAT)            # common-noun DP with a relative clause

    # Relaxation: resolve the mutually-recursive END pointers.  Each sweep propagates one more
    # nesting level; K sweeps suffice for CP-depth<=2 (this is the bounded-depth, NON-length-
    # generalizing step).  dpfull=DP end (incl. RC), vpe=VP end, rce=CP_rel end, cpe=CP_sent end.
    dpfull = dp_end_flat
    vpe = hb_end
    rce = idx
    cpe = idx
    K = ROUNDS
    for _ in range(K):
        # DP end: jump over its relative clause when present
        rce_at_rc = index_select(rce, rc_that_pos, default=0, causal=False)
        dpfull = where(dp_has_rc, rce_at_rc, dp_end_flat)
        # VP end: intrans/objgap stop at head-block; trans -> object-DP end; clausal -> CP end
        hb1 = hb_end + 1
        dpfull_at_hb1 = index_select(dpfull, hb1, default=0, causal=False)
        cpe_at_hb1 = index_select(cpe, hb1, default=0, causal=False)
        vpe = where((role == 1) | (role == 4), hb_end,
              where(role == 2, dpfull_at_hb1,
              where(role == 3, cpe_at_hb1, idx)))
        # CP_rel end: subject-gap -> its VP end; object-gap -> the gap verb (after subject DP)
        q1 = idx + 1
        vpe_at_q1 = index_select(vpe, q1, default=0, causal=False)
        dpfull_at_q1 = index_select(dpfull, q1, default=0, causal=False)
        rce = where(is_crel, where(next_lc == V, vpe_at_q1, dpfull_at_q1 + 1), idx)
        # CP_sent end: 'that' DP VP -> end of the embedded clause's VP
        vpos = dpfull_at_q1 + 1
        vpe_at_vpos = index_select(vpe, vpos, default=0, causal=False)
        cpe = where(is_c, vpe_at_vpos, idx)

    # clause end e(q) for every 'that'
    e = where(is_crel, rce, where(is_c, cpe, idx))

    # ===================== Strategy A: CP-depth via opener/close stabbing =====================
    # depth[i] = #('that' q<=i with clause-end e(q) >= i).  Mask non-'that' ends to -1.
    end_that = where(isthat, e, full(hi_ids, -1))
    depth = _stab_ge(end_that)                            # one counting head

    # ===================== displacement: four additive flip-type contributions =============
    # --- d_adv: VP_x_adv swaps [V, Adv] -> [Adv, V]  (verb +1 if adverb follows; adverb -1) ---
    d_adv = where(isverb & (next_lc == ADV), full(hi_ids, 1), ZERO) \
          + where(isadv, full(hi_ids, -1), ZERO)

    # --- d_cp: CP_sent+CP_rel.  Every token in a clause body gets -1 per enclosing 'that'
    #     (= -depth, since 'that' is not in its own body); each 'that' also gets +|its clause|. ---
    span_clause = e - idx + 1                             # size of the clause a 'that' opens
    d_cp = -depth + where(isthat, span_clause, ZERO)

    # --- d_nprel: NP_singular -> [core noun(-phrase), CP_rel].  Core gets +|RC|; RC tokens get
    #     -|core| for each enclosing relative clause (a weighted stabbing sum, weight=|core|). ---
    e_at_p1 = index_select(e, idx + 1, default=0, causal=False)   # end of a C_rel 'that' at i+1
    e_at_p2 = index_select(e, idx + 2, default=0, causal=False)   # ... at i+2
    core_noun_bonus = where((lc == NSING) & (next_lc == THAT), e_at_p1 - (idx + 1) + 1, ZERO)
    core_adj_bonus = where((lc == ADJ) & (next_lc == NSING) & (next2_lc == THAT),
                           e_at_p2 - (idx + 2) + 1, ZERO)
    d_nprel_core = core_noun_bonus + core_adj_bonus
    # weight of a C_rel = |core| = 2 if the core noun carries an adjective (lc[q-2]==Adj) else 1
    has_adj_core = (prev2_lc == ADJ)
    crel_end_w1 = where(is_crel & ~has_adj_core, e, full(hi_ids, -1))
    crel_end_w2 = where(is_crel & has_adj_core, e, full(hi_ids, -1))
    d_nprel_inside = -(_stab_ge(crel_end_w1) + 2 * _stab_ge(crel_end_w2))
    d_nprel = d_nprel_core + d_nprel_inside

    # --- d_vp: transitive/clausal VP -> [head-block, complement].  Head-block (verb[+adverb])
    #     gets +|comp|; complement tokens get -|head-block| per enclosing such VP (weighted
    #     stabbing sum, weight=|head-block| in {1,2}). ---
    comp_len = vpe - hb_end                               # |complement| for a VP-flip verb
    is_vpflip = (role == 2) | (role == 3)
    prev_role = _read_off(role, -1, default=0)
    role_prev2 = _read_off(role, -2, default=0)
    comp_len_prev = _read_off(comp_len, -1, default=0)
    d_vp_head = where(is_vpflip, comp_len, ZERO) \
              + where(isadv & ((prev_role == 2) | (prev_role == 3)), comp_len_prev, ZERO)
    # complement-start markers, placed at the first complement token p:
    #   weight-1: verb at p-1 is a VP-flip with NO adverb (current token isn't the adverb)
    #   weight-2: verb at p-2 is a VP-flip WITH adverb at p-1
    vpe_prev = _read_off(vpe, -1, default=0)
    vpe_prev2 = _read_off(vpe, -2, default=0)
    w1_here = ((prev_role == 2) | (prev_role == 3)) & (lc != ADV)
    w2_here = ((role_prev2 == 2) | (role_prev2 == 3)) & (prev_lc == ADV)
    cs_end_w1 = where(w1_here, vpe_prev, full(hi_ids, -1))
    cs_end_w2 = where(w2_here, vpe_prev2, full(hi_ids, -1))
    d_vp_comp = -(_stab_ge(cs_end_w1) + 2 * _stab_ge(cs_end_w2))
    d_vp = d_vp_head + d_vp_comp

    # ===================== assemble hf_pos and gather (Stage 3) =====================
    disp = d_adv + d_cp + d_nprel + d_vp
    hf_pos = idx + disp                                   # target slot of each HI token
    # invert the permutation: for output slot j, find the unique i with hf_pos[i]==j ...
    src = kqv(hf_pos, idx, idx, equals, default=0, causal=False)
    # ... then gather the HI token id into that slot.
    out_ids = index_select(hi_ids, src, default=0, causal=False)
    return [_I2W.get(int(t), "<unk>") for t in out_ids]

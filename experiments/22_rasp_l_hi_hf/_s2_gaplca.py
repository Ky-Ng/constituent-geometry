"""Strategy C (pairwise flip-LCA counting) HI->HF reordering as a RASP-L program.

Interface:  hi_to_hf(hi_tokens: list[str]) -> list[str]

Verified facts we build on (see the task prompt):
  * HF is a permutation of HI, so it suffices to compute a target slot hf_pos[i]
    for every HI token i and then gather (inverse permutation).
  * hf_pos[i] = i + disp[i], where the displacement equals the pairwise flip-LCA
    count:   disp[i] = #{j>i : LCA(i,j) is a flip node}
                     - #{j<i : LCA(i,j) is a flip node}.
    (Empirically, for every token both flip-sets are a *contiguous* interval, and
    each pair's LCA is a flip node iff i and j sit in the two children of a
    reversing constituent.)

How we realize the two flip-LCA counts with s-ops only
------------------------------------------------------
For a pair i<j the LCA is a flip node iff i lies in the HEAD child and j in the
COMP child of some flip constituent F (head always precedes comp in HI).  In this
grammar every flip node's HEAD child is tiny (<=2 tokens: a 'that', a verb, a
verb+adverb, or a noun+adjective) while its COMP child is the large block.  Hence

  Term1[i] = #{j>i : LCA flip} = sum over flip nodes F that i *heads* of |comp(F)|.
    i heads at most two flip nodes, so Term1 is a small local sum of comp-sizes.

  Term2[i] = #{j<i : LCA flip} = sum over flip nodes F whose comp contains i of
    |head(F)|.  Because head sizes are 1 or 2, this is a sum of *interval-stabbing
    counts*, each of which we get as a difference of two `sel_width` counts of the
    form  #{flip nodes whose comp-start <= i} - #{flip nodes whose comp-end < i}.

disp = Term1 - Term2,  hf_pos = indices + disp,  then invert-and-gather.

The only structural quantity that is not local is, for each 'that', where its CP
closes (the crux: clause closings are unmarked in the surface string).  We obtain
it with a fixed number (K) of relaxation passes of a mutually-recursive
end-of-constituent computation (close / dpend / vpend), each pass being a few
`index_select`/`where` s-ops.  K is a constant for depth<=2 (bounded-depth trick,
NOT length-generalizing -- flagged in the writeup).  All cross-position reads use
`index_select`/`select` with causal=False, justified because in the real
`hi <sep> hf` decoder the whole HI block lies in the past.
"""
from __future__ import annotations
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rasp_l import (  # noqa: E402
    tok_map, seq_map, select, sel_width, kqv, where, index_select,
    full, indices, equals, leq, lt,
)
from grammar.v2.cfg_vocab import VOCAB  # noqa: E402

# ---------------------------------------------------------------------------
# token -> id and id -> local-class dictionaries (a fixed token embedding / MLP)
# ---------------------------------------------------------------------------
_D = set(VOCAB["D"]); _NS = set(VOCAB["N_singular"]); _NP = set(VOCAB["N_proper"])
_ADJ = set(VOCAB["Adj"]); _ADV = set(VOCAB["Adv"])
_VERB = set(VOCAB["V_intrans"]) | set(VOCAB["V_dp"]) | set(VOCAB["V_cp"])

_WORDS = sorted(set().union(_D, _NS, _NP, _ADJ, _ADV, _VERB, {"that"}))
_WORD2ID = {w: i + 1 for i, w in enumerate(_WORDS)}   # 0 reserved / unused
_ID2WORD = {i: w for w, i in _WORD2ID.items()}

BIG = 1000        # sentinel "infinity" for non-anchor entries (positions are < 28)


def _cls(word, s):
    return 1 if word in s else 0


# ---------------------------------------------------------------------------
# the RASP-L program
# ---------------------------------------------------------------------------
def hi_to_hf(hi_tokens: list[str]) -> list[str]:
    n = len(hi_tokens)
    if n == 0:
        return []

    # ---- encode words to integer ids (token embedding) --------------------
    ids = np.array([_WORD2ID[w] for w in hi_tokens], dtype=int)

    # ---- elementwise local classification (one MLP each) ------------------
    is_that  = tok_map(ids, lambda x: 1 if _ID2WORD[x] == "that" else 0)   # token == 'that'
    is_verb  = tok_map(ids, lambda x: _cls(_ID2WORD[x], _VERB))            # any verb
    is_N     = tok_map(ids, lambda x: _cls(_ID2WORD[x], _NS))              # singular noun
    is_prop  = tok_map(ids, lambda x: _cls(_ID2WORD[x], _NP))             # proper noun
    is_adj   = tok_map(ids, lambda x: _cls(_ID2WORD[x], _ADJ))            # adjective
    is_adv   = tok_map(ids, lambda x: _cls(_ID2WORD[x], _ADV))            # adverb
    is_Dst   = tok_map(ids, lambda x: 1 if (_ID2WORD[x] in _D or _ID2WORD[x] in _NP) else 0)  # DP start (D or proper)

    idx = indices(ids)  # position s-op [0..n-1]

    # neighbour readers (relative-position gathers; causal=False = read HI block)
    def rd(arr, target, default):
        # arr[target[i]] with out-of-range -> default (one attention head)
        return index_select(arr, target, default=default, causal=False)

    is_that_next = rd(is_that, idx + 1, 0)   # is position i+1 a 'that'
    is_N_next    = rd(is_N,    idx + 1, 0)   # is position i+1 a singular noun
    is_adv_next  = rd(is_adv,  idx + 1, 0)   # is position i+1 an adverb
    is_N_prev    = rd(is_N,    idx - 1, 0)   # is position i-1 a singular noun
    is_adj_prev  = rd(is_adj,  idx - 1, 0)   # is position i-1 an adjective

    # is this 'that' a relative complementizer? -> previous token is a sing. noun
    is_crel = seq_map(is_that, is_N_prev, lambda t, p: 1 if (t and p) else 0)  # C_rel vs C

    # CORE[s] = position of the noun of the DP that starts at s (local)
    core_if_D = where(is_N_next, idx + 1, idx + 2)             # D (Adj) N -> noun at s+1 or s+2
    CORE = where(is_prop, idx, core_if_D)                      # proper -> itself
    # AE[v] = end of the verb's head block: v, or v+1 if an adverb follows (local)
    AE = where(is_adv_next, idx + 1, idx)

    # ---- relaxation for close / dpend / vpend (K bounded passes) ----------
    # DPEND[s] : end position of the DP starting at s (>= its noun, +RC if any)
    # VPEND[v] : end position of the VP headed by verb v
    # CLOSE[t] : end position of the CP opened by 'that' at t
    DPEND = where(is_Dst,  CORE, full(ids, BIG))   # init: DP with no relative clause
    VPEND = where(is_verb, AE,   full(ids, BIG))   # init: intransitive verb
    CLOSE = where(is_that, idx,  full(ids, BIG))   # init: empty CP
    K = 6                                          # constant #passes (depth<=2)
    for _ in range(K):
        oCL, oDP, oVP = CLOSE, DPEND, VPEND
        # DPEND[s] = CLOSE[CORE[s]+1] if an RC follows the core noun, else CORE[s]
        rc_that   = rd(is_that, CORE + 1, 0)        # is there a 'that' right after the core noun
        close_rc  = rd(oCL,     CORE + 1, BIG)      # that RC's close
        nDP = where(is_Dst, where(rc_that, close_rc, CORE), full(ids, BIG))
        # VPEND[v] : complement chosen by the token after the head block
        nxt       = AE + 1
        that_nxt  = rd(is_that, nxt, 0)
        Dst_nxt   = rd(is_Dst,  nxt, 0)
        close_nxt = rd(oCL,     nxt, BIG)           # CP_sent complement end
        dpend_nxt = rd(oDP,     nxt, BIG)           # DP object complement end
        vp_val = where(that_nxt, close_nxt, where(Dst_nxt, dpend_nxt, AE))
        nVP = where(is_verb, vp_val, full(ids, BIG))
        # CLOSE[t] : CP_rel -> VPEND[t+1] (subj-gap) or DPEND[t+1]+1 (obj-gap);
        #            CP_sent -> VPEND[ DPEND[t+1]+1 ]
        bs        = idx + 1
        v_bs      = rd(is_verb, bs, 0)
        vpend_bs  = rd(oVP, bs, BIG)
        dpend_bs  = rd(oDP, bs, BIG)
        crel_val  = where(v_bs, vpend_bs, dpend_bs + 1)
        vpend_de1 = rd(oVP, dpend_bs + 1, BIG)      # CP_sent: verb after the subject DP
        cl_val = where(is_crel, crel_val, vpend_de1)
        nCL = where(is_that, cl_val, full(ids, BIG))
        DPEND, VPEND, CLOSE = nDP, nVP, nCL

    # ---- Term1[i] = sum of |comp(F)| over the <=2 flip nodes i heads -------
    # CP flip headed by a 'that': comp = body = [t+1 .. CLOSE[t]]
    close_here = rd(CLOSE, idx, BIG)                          # CLOSE[i]
    t1_cp = where(is_that, close_here - idx, full(ids, 0))    # |body| = CLOSE[i]-i

    # a verb heads VP_x_adv (comp = the adverb, size 1) when an adverb follows
    t1_vadv = where(seq_map(is_verb, is_adv_next, lambda v, a: 1 if (v and a) else 0),
                    full(ids, 1), full(ids, 0))

    # a verb (at its head-block end AE=i) heads main-VP when a complement follows;
    # comp end = CLOSE[nxt] (that) or DPEND[nxt] (DP object); size = end-nxt+1
    v_nxt      = AE + 1
    v_that     = rd(is_that, v_nxt, 0)
    v_Dst      = rd(is_Dst,  v_nxt, 0)
    v_closeN   = rd(CLOSE, v_nxt, BIG)
    v_dpendN   = rd(DPEND, v_nxt, BIG)
    v_compend  = where(v_that, v_closeN, where(v_Dst, v_dpendN, idx))   # comp end (or idx if none)
    v_hascomp  = seq_map(is_verb, seq_map(v_that, v_Dst, lambda a, b: 1 if (a or b) else 0),
                         lambda v, c: 1 if (v and c) else 0)
    t1_vmain   = where(v_hascomp, v_compend - v_nxt + 1, full(ids, 0))

    # an adverb is the 2nd head token of main-VP: comp starts at i+1 (verb at i-1)
    a_nxt      = idx + 1
    a_that     = rd(is_that, a_nxt, 0)
    a_Dst      = rd(is_Dst,  a_nxt, 0)
    a_closeN   = rd(CLOSE, a_nxt, BIG)
    a_dpendN   = rd(DPEND, a_nxt, BIG)
    a_compend  = where(a_that, a_closeN, where(a_Dst, a_dpendN, idx))
    a_hascomp  = seq_map(is_adv, seq_map(a_that, a_Dst, lambda a, b: 1 if (a or b) else 0),
                         lambda v, c: 1 if (v and c) else 0)
    t1_aadv    = where(a_hascomp, a_compend - a_nxt + 1, full(ids, 0))

    # a noun heads NP-rel when a relative clause follows: comp = [nn+1 .. CLOSE[nn+1]]
    n_close1   = rd(CLOSE, idx + 1, BIG)                      # CLOSE[i+1]
    n_hasrc    = seq_map(is_N, is_that_next, lambda a, b: 1 if (a and b) else 0)
    t1_noun    = where(n_hasrc, n_close1 - idx, full(ids, 0)) # (CLOSE[i+1]-(i+1)+1)=CLOSE[i+1]-i

    # an adjective is the 1st head token of NP-rel when its noun (i+1) has an RC:
    # comp = [nn+2 .. CLOSE[nn+2]]
    adj_N_next   = is_N_next
    adj_that_2   = rd(is_that, idx + 2, 0)                    # 'that' two positions ahead
    adj_close2   = rd(CLOSE, idx + 2, BIG)                    # CLOSE[i+2]
    adj_hasrc    = seq_map(is_adj, seq_map(adj_N_next, adj_that_2, lambda a, b: 1 if (a and b) else 0),
                           lambda a, b: 1 if (a and b) else 0)
    t1_adj       = where(adj_hasrc, adj_close2 - idx - 1, full(ids, 0))  # CLOSE[i+2]-(i+2)+1

    Term1 = t1_cp + t1_vadv + t1_vmain + t1_aadv + t1_noun + t1_adj

    # ---- Term2[i] = weighted stabbing counts over flip comp-spans ----------
    # helper counts: #{positions p : arr[p] < i}  and  #{p : arr[p] <= i}
    def cnt_lt(arr):   # sel_width of one attention head, causal=False
        return sel_width(select(arr, idx, lt, causal=False))
    def cnt_le(arr):
        return sel_width(select(arr, idx, leq, causal=False))

    # CP flips (head size 1): T2_CP = #{that t<i} - #{that close(t)<i}
    THAT_pos   = where(is_that, idx, full(ids, BIG))
    THAT_close = where(is_that, close_here, full(ids, BIG))
    t2_cp = cnt_lt(THAT_pos) - cnt_lt(THAT_close)

    # VP_x_adv flips (head size 1): comp = the adverb itself -> contributes 1 at adverbs
    t2_vadv = is_adv

    # main-VP flips: anchor at the verb; comp-start = v_nxt, comp-end = v_compend.
    #   a_mvp = #{start<=i} - #{end<i};  extra +1 for head size 2 (verb+adverb).
    MVP_start = where(v_hascomp, v_nxt,     full(ids, BIG))
    MVP_end   = where(v_hascomp, v_compend, full(ids, BIG))
    a_mvp = cnt_le(MVP_start) - cnt_lt(MVP_end)
    hs2_mvp = seq_map(v_hascomp, is_adv_next, lambda c, a: 1 if (c and a) else 0)  # verb+adverb head
    MVP_start2 = where(hs2_mvp, v_nxt,     full(ids, BIG))
    MVP_end2   = where(hs2_mvp, v_compend, full(ids, BIG))
    b_mvp = cnt_le(MVP_start2) - cnt_lt(MVP_end2)

    # NP-rel flips: anchor at the noun; comp = [nn+1 .. CLOSE[nn+1]].
    NPR_start = where(n_hasrc, idx + 1,   full(ids, BIG))
    NPR_end   = where(n_hasrc, n_close1,  full(ids, BIG))
    a_np = cnt_le(NPR_start) - cnt_lt(NPR_end)
    hs2_np = seq_map(n_hasrc, is_adj_prev, lambda c, a: 1 if (c and a) else 0)  # adj+noun core
    NPR_start2 = where(hs2_np, idx + 1,  full(ids, BIG))
    NPR_end2   = where(hs2_np, n_close1, full(ids, BIG))
    b_np = cnt_le(NPR_start2) - cnt_lt(NPR_end2)

    Term2 = t2_cp + t2_vadv + a_mvp + b_mvp + a_np + b_np

    # ---- displacement, target slot, invert, gather ------------------------
    disp   = Term1 - Term2                       # elementwise
    hf_pos = idx + disp                          # target slot of HI token i

    # inverse permutation: inv[j] = the i with hf_pos[i]==j  (one attention head)
    inv = kqv(hf_pos, idx, idx, equals, default=0, causal=False)
    out_ids = index_select(ids, inv, causal=False)   # out[j] = hi[inv[j]]

    return [_ID2WORD[int(x)] for x in out_ids]

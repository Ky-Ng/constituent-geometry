"""Strategy D (explicit bounded 2-level recursion) RASP-L program: HI -> HF.

Translates a head-initial sentence of the bounded v2 toy CFG (CP-nesting depth
<= 2, length <= 28) into its head-final reordering.  HF is a permutation of HI
(VERIFIED FACT 1): it suffices to compute, per HI position i, its target slot
    hf_pos[i] = i + disp[i]
and gather.  disp[i] is the nested-reversal displacement: a sum over enclosing
FLIP nodes F of ( +|comp(F)| if i in head(F) else -|head(F)| ) (VERIFIED FACT 2).

We split disp into four additive contribution families (each a distinct kind of
flip node; they simply add because a token's total displacement is the sum over
ALL its flip-ancestors):

  A) CP flips        (CP_sent [C S] and CP_rel [C_rel S_gap], |head|=1):
        the 'that' head gets +|clause|, every token in the clause gets -1.
        => non-'that' token: -depth[i]   ('depth' already = #enclosing CPs);
           'that'          : +|its clause| - (depth[i]-1).
  B) VP-main flips   (VP with an overt complement: V_dp+objDP or V_cp+CP_sent):
        head block = verb (+adverb), comp = object DP or CP_sent.
        verb(+adverb) get +|comp|; comp tokens get -|head block| (1 or 2).
  C) Adverb flips    (VP_intrans/dp/cp_adv = [V Adv], |head|=|comp|=1):
        the verb gets +1, the adverb gets -1  (purely local).
  D) NP rel-attach   (NP_singular -> [coreNoun CP_rel]):
        the core noun (noun, +adjective) gets +|CP_rel|; the whole relative
        clause gets -|core noun| (1 or 2).

The crux (VERIFIED FACT 4) is that clause closings are UNMARKED in the surface
string, so 'depth' and every span size must be inferred.  We do this with a
bounded fix-point that resolves, for each constituent-start position, the index
where that constituent ENDS (CP/DP/VP/clause end pointers).  The recursion is
mutually defined and bottom-up (Strategy D: innermost clauses resolve first,
then depth-1, then the matrix) and, because CP-depth <= 2 and length <= 28, it
converges in a bounded number of iterations.

RASP-L discipline: every cross-position read goes through select / kqv /
sel_width / index_select; all elementwise work is tok_map / seq_map (legal
token-embeddings / MLPs).  We use causal=False for reads over the HI block,
which is justified: in the real 'hi <sep> hf' decoder the HI block is entirely
in the past.  The fix-point uses a fixed iteration count (a bounded-depth trick)
so this program is depth<=2 / length<=28 specific and is NOT claimed to
length-generalize -- exactly what Strategy D permits.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from rasp_l import (  # noqa: E402
    full, indices, tok_map, seq_map, select, sel_width, index_select, where,
    kqv, equals, geq,
)
try:  # vocabulary (surface word -> id, id -> coarse POS)
    from grammar.v2.cfg_vocab import VOCAB, token_to_id
except ModuleNotFoundError:  # allow running with src/ not yet on path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from grammar.v2.cfg_vocab import VOCAB, token_to_id

# ---------------------------------------------------------------------------
# Static token tables (a fixed token-embedding: word -> id -> coarse category)
# ---------------------------------------------------------------------------
_D = set(VOCAB["D"]); _NSING = set(VOCAB["N_singular"]); _NPROP = set(VOCAB["N_proper"])
_ADJ = set(VOCAB["Adj"]); _ADV = set(VOCAB["Adv"])
_VERB = set(VOCAB["V_intrans"]) | set(VOCAB["V_dp"]) | set(VOCAB["V_cp"])

# coarse category ids (fine POS of 'that'/verbs is disambiguated by neighbours)
THAT, CD, CNPROP, CNSING, CADJ, CADV, CVERB, NONE = range(8)

def _cat_of_word(w: str) -> int:
    if w == "that": return THAT
    if w in _D: return CD
    if w in _NPROP: return CNPROP
    if w in _NSING: return CNSING
    if w in _ADJ: return CADJ
    if w in _ADV: return CADV
    if w in _VERB: return CVERB
    return NONE

_WORD2ID = token_to_id()
_ID2WORD = {i: w for w, i in _WORD2ID.items()}
_ID2CAT = {i: _cat_of_word(w) for w, i in _WORD2ID.items()}

INF = 1000        # out-of-range / unresolved pointer sentinel (> any length)
ITERS = 40        # bounded fix-point sweeps (depth<=2, length<=28 => converges)


# ---------------------------------------------------------------------------
# small helpers built only from the allowed s-ops
# ---------------------------------------------------------------------------

def _rd(x, offset, default):
    """Read x at position i+offset (causal=False: HI block is all in the past).

    shift_right can only read the past (its internal kqv is causal), so future
    neighbour reads (offset > 0) must go through an explicit non-causal gather.
    """
    return index_select(x, indices(x) + offset, default=default, causal=False)


def _gather(x, idx, default):
    """out[i] = x[idx[i]] (non-causal gather over the HI block)."""
    return index_select(x, idx, default=default, causal=False)


def _cover(endvals, mask):
    """# keys t with mask[t]==1, t<=i, endvals[t]>=i  (one attention head).

    Key value = span-end (or -1 when unmasked so the geq fails); causal makes
    t<=i automatic, so the head counts spans [t..endvals[t]] that cover i.
    """
    keyval = where(mask, endvals, full(mask, -1))               # mask off non-keys
    A = select(keyval, indices(mask), geq, causal=True)         # end[t]>=i & t<=i
    return sel_width(A)                                         # count covering spans


def _eq(x, c):
    """Elementwise (x == c) as a 0/1 s-op (a token-embedding / MLP)."""
    return tok_map(x, lambda v, c=c: 1 if v == c else 0)


def _and(a, b):
    return seq_map(a, b, lambda x, y: 1 if (x and y) else 0)


def _or(a, b):
    return seq_map(a, b, lambda x, y: 1 if (x or y) else 0)


# ---------------------------------------------------------------------------
# main program
# ---------------------------------------------------------------------------

def hi_to_hf(hi_tokens: list[str]) -> list[str]:
    if len(hi_tokens) == 0:
        return []

    # --- encode words -> ids -> coarse category (token-embedding) ----------
    ids = tok_map(hi_tokens, lambda w: _WORD2ID[w])             # surface word -> id
    cat = tok_map(ids, lambda i: _ID2CAT[i])                    # id -> coarse POS
    idx = indices(cat)                                          # [0,1,...,n-1]

    isThat = _eq(cat, THAT)
    isVerb = _eq(cat, CVERB)
    isAdv = _eq(cat, CADV)
    isAdj = _eq(cat, CADJ)
    isNsing = _eq(cat, CNSING)
    isD = _eq(cat, CD)
    isNprop = _eq(cat, CNPROP)
    isDPstart = _or(isD, isNprop)                              # a DP begins here

    # neighbour categories (causal=False reads over the past HI block)
    catP1 = _rd(cat, -1, NONE)                                 # cat[i-1]
    catP2 = _rd(cat, -2, NONE)                                 # cat[i-2]
    catN1 = _rd(cat, 1, NONE)                                  # cat[i+1]
    catN2 = _rd(cat, 2, NONE)                                  # cat[i+2]

    # 'that' is C_rel iff the previous token is a singular noun, else C (FACT 3)
    isCrel = _and(isThat, _eq(catP1, CNSING))

    # verb sub-classification by the next NON-adverb token (FACT 3)
    hasAdv = _and(isVerb, _eq(catN1, CADV))                    # verb followed by adverb
    after = seq_map(idx, hasAdv, lambda i, h: i + 2 if h else i + 1)  # 1st post-verb-block slot
    nnaCat = where(hasAdv, catN2, catN1)                      # category of that slot
    VCPv = _and(isVerb, _eq(nnaCat, THAT))                    # verb + CP_sent complement
    VDPobj = _and(isVerb, tok_map(nnaCat,
                                  lambda c: 1 if c in (CD, CNPROP) else 0))  # verb + overt obj DP
    VPflip = _or(VCPv, VDPobj)                                 # VP that actually reverses

    # noun slot inside a D-headed DP (skip an optional adjective)
    nz = seq_map(idx, catN1, lambda i, c: i + 2 if c == CADJ else i + 1)

    # ----- bounded fix-point: resolve constituent END pointers -------------
    # DPEND/VPEND/CLEND/CPEND[i] = last index of the DP/VP/clause/CP starting
    # at i.  All references point strictly right, so a bounded number of sweeps
    # (bottom-up: innermost clauses first) reaches the fix-point.
    DPEND = idx.copy(); VPEND = idx.copy(); CLEND = idx.copy(); CPEND = idx.copy()
    for _ in range(ITERS):
        # DPEND: proper -> self; D [Adj] N [rel] -> noun, or the rel-clause end
        thatAtNz1 = _gather(isThat, nz + 1, 0)                 # is there a rel clause on the noun?
        cpAtNz1 = _gather(CPEND, nz + 1, INF)                  # that rel clause's end
        dcaseD = where(thatAtNz1, cpAtNz1, nz)                 # D-headed DP end
        newDP = where(isNprop, idx, where(isD, dcaseD, DPEND))

        # VPEND: V_cp -> embedded CP end; V_dp+obj -> object DP end; else verb(+adv)
        cpAtAfter = _gather(CPEND, after, INF)                 # end of a CP_sent complement
        dpAtAfter = _gather(DPEND, after, INF)                 # end of an object DP
        intransEnd = where(hasAdv, idx + 1, idx)              # bare verb / verb+adverb end
        vpval = where(VCPv, cpAtAfter, where(VDPobj, dpAtAfter, intransEnd))
        newVP = where(isVerb, vpval, VPEND)

        # CLEND: verb-start (subj-gap) -> its VP end; DP-subject -> the VP after it
        vpAfterSubj = _gather(newVP, newDP + 1, INF)          # VP end past a DP subject
        newCL = where(isVerb, newVP, where(isDPstart, vpAfterSubj, CLEND))

        # CPEND: a 'that' opens a CP whose end is the end of the clause it heads
        clAfterThat = _gather(newCL, idx + 1, INF)
        newCP = where(isThat, clAfterThat, CPEND)

        DPEND, VPEND, CLEND, CPEND = newDP, newVP, newCL, newCP

    # ----- covering-depths (attention counts of enclosing spans) -----------
    depth = _cover(CPEND, isThat)                             # # enclosing CPs (FACT 4)
    relAdjMask = _and(isCrel, _eq(catP2, CADJ))              # rel clauses with an adj core
    relDepth = _cover(CPEND, isCrel)                          # # enclosing relative clauses
    relAdjDepth = _cover(CPEND, relAdjMask)                   # # ... whose core has an adjective

    # ----- (A) CP-flip contribution ----------------------------------------
    thatA = seq_map(seq_map(CPEND, idx, lambda c, i: c - i),  # +|clause| ...
                    depth, lambda comp, d: comp - (d - 1))    # ... minus outer CPs
    capA = where(isThat, thatA, tok_map(depth, lambda d: -d)) # non-that: -depth

    # ----- (C) adverb-flip contribution ------------------------------------
    verbAdvPlus = _and(isVerb, _eq(catN1, CADV))              # verb just before an adverb: +1
    advC = where(isAdv, full(cat, -1), verbAdvPlus)          # adverb: -1

    # ----- (D) NP rel-attachment contribution ------------------------------
    # + part: core noun (and its adjective) get +|CP_rel| of the rel clause to its right
    cpN1 = _rd(CPEND, 1, INF)                                 # CPEND[j+1]
    cpN2 = _rd(CPEND, 2, INF)                                 # CPEND[j+2]
    nounRel = _and(isNsing, _eq(catN1, THAT))                # noun immediately before a rel clause
    nounPlus = where(nounRel, seq_map(cpN1, idx, lambda c, i: c - i), full(cat, 0))
    adjRel = _and(_and(isAdj, _eq(catN1, CNSING)), _eq(catN2, THAT))  # adj of an adj+noun core w/ rel
    adjPlus = where(adjRel, seq_map(cpN2, idx, lambda c, i: c - i - 1), full(cat, 0))
    plusD = seq_map(nounPlus, adjPlus, lambda a, b: a + b)
    # - part: every token in a rel clause loses |core| (=1, +1 more if adj core)
    minusD = seq_map(relDepth, relAdjDepth, lambda a, b: -(a + b))

    # ----- (B) VP-main-flip contribution -----------------------------------
    # + part: verb (and its adverb) get +|comp|
    ceV = where(VCPv, cpAtAfter, dpAtAfter)                   # comp end for a flip verb
    compSize = where(VPflip, seq_map(ceV, after, lambda ce, cs: ce - cs + 1), full(cat, 0))
    compSizePrev = _rd(compSize, -1, 0)                       # its verb's |comp|, read by the adverb
    plusB = seq_map(compSize, where(isAdv, compSizePrev, full(cat, 0)),
                    lambda v, a: v + a)
    # - part: comp tokens lose |head block| (1, or 2 when the verb has an adverb).
    # Mark each comp-START position s and give it the comp end + head-size flag,
    # then count covering comps with the same attention trick as depth.
    VPfP1 = _rd(VPflip, -1, 0); VPfP2 = _rd(VPflip, -2, 0)    # is s-1 / s-2 a flip verb?
    VCPvP1 = _rd(VCPv, -1, 0); VCPvP2 = _rd(VCPv, -2, 0)
    caseA = _and(VPfP1, tok_map(cat, lambda c: 1 if c != CADV else 0))  # comp right after verb (no adv)
    caseB = _and(VPfP2, _eq(catP1, CADV))                    # comp after verb+adverb
    markB = _or(caseA, caseB)                                 # s is a flip comp start
    isVcpGv = _or(_and(caseA, VCPvP1), _and(caseB, VCPvP2))   # governing verb is V_cp?
    endB = where(isVcpGv, CPEND, DPEND)                       # comp end at position s
    C1 = _cover(endB, markB)                                  # # enclosing flip comps
    C2 = _cover(endB, caseB)                                  # # ... whose head block is size 2
    minusB = seq_map(C1, C2, lambda a, b: -(a + b))

    # ----- total displacement and the permutation --------------------------
    disp = capA
    for part in (advC, plusD, minusD, plusB, minusB):
        disp = seq_map(disp, part, lambda a, b: a + b)
    hf_pos = seq_map(idx, disp, lambda i, d: i + d)          # target slot of each HI token

    # inverse permutation: src[j] = the i with hf_pos[i]==j; then out = hi[src]
    src = kqv(hf_pos, idx, idx, equals, default=0, causal=False)
    outIds = index_select(ids, src, default=0, causal=False)  # gather permuted ids

    # decode ids back to surface words (inverse of the initial token-embedding)
    return [_ID2WORD[int(t)] for t in outIds]

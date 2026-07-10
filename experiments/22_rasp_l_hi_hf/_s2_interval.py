"""Strategy B (opener->closer interval matching) RASP-L program for HI -> HF.

Interface:  hi_to_hf(hi_tokens: list[str]) -> list[str]

The program is a straight-line composition of the s-op primitives in ``rasp_l.py``
(one attention head per select/kqv/sel_width, one MLP per tok_map/seq_map/where).
No Python control flow reads across sequence positions; every cross-position fact
is carried by select / kqv / sel_width / a relative-position read. ``causal=False``
is used only for reads over the HI block, which in the real ``hi <sep> hf`` decoder
lies entirely in the past.

Algorithm (all verified 100% on depth<=2, see LOG at bottom):
  1. Encode words -> ids; token-embed each id to its lexical class.
  2. Resolve fine POS locally (prev / next / next-skip-adverb reads).
  3. Balance  B = (#that so far) - (#verb so far).  Each clause is balanced
     (#that == #verb inside it), so a 'that' opener and its clause's MAIN verb
     are a matched open/close pair in this that/verb bracketing.
  4. Match every verb LEFTWARD to the 'that' it closes (moc), then invert to get
     mv[p] = main verb of the clause opened by 'that' p.
  5. The clause END (interval close) of a 'that' extends PAST its main verb by the
     verb's complement tail (object DP / embedded CP).  Compute vend(verb) and
     close(that)=vend(mv(that)) by bounded pointer-jumping (deeper complements
     resolve first; depth<=2 => a few rounds converge).
  6. depth[i] = (#that <= i) - (#that whose close < i)  = # intervals covering i.
  7. Displacement = sum of signed block-swaps at every flip node.  Split into a
     LOCAL head-part (a token in a flipped node's head child moves right by the
     comp size) and a comp-part computed by interval counting (a token in a flip's
     comp child moves left by the head size); comp sizes are all 1 or 2, so the
     comp-part is two indicator-interval counts.  hf_pos = i + disp; gather.
"""
from __future__ import annotations

import numpy as np

from rasp_l import (
    tok_map, seq_map, select, sel_width, aggr, kqv,
    cumsum_incl, where, index_select, full, indices,
    equals, leq, lt, geq, gt,
)
from grammar.v2.cfg_vocab import VOCAB, grammar_words

# --------------------------------------------------------------------------- #
# Fixed lexicons (used only inside tok_map / seq_map = token-embeddings/MLPs).
# --------------------------------------------------------------------------- #
_D = set(VOCAB["D"]); _NS = set(VOCAB["N_singular"]); _NP = set(VOCAB["N_proper"])
_ADJ = set(VOCAB["Adj"]); _ADV = set(VOCAB["Adv"])
_VIN = set(VOCAB["V_intrans"]); _VDP = set(VOCAB["V_dp"]); _VCP = set(VOCAB["V_cp"])
_VERB = _VIN | _VDP | _VCP
_THAT = "that"

_WORDS = grammar_words()
_W2ID = {w: i for i, w in enumerate(_WORDS)}
_ID2W = {i: w for w, i in _W2ID.items()}

BIG = 100000  # sentinel "position at infinity" (never satisfies a <= index test)

# lexical-class codes: 1 D, 2 Ns, 3 Np, 4 Adj, 5 Adv, 6 that, 7 V (0 = n/a)
def _lex_code(wid):
    w = _ID2W.get(wid)
    if w in _D:   return 1
    if w in _NS:  return 2
    if w in _NP:  return 3
    if w in _ADJ: return 4
    if w in _ADV: return 5
    if w == _THAT: return 6
    if w in _VERB: return 7
    return 0

def _is_vintr(wid):
    return 1 if _ID2W.get(wid) in _VIN else 0

# fine-POS codes: 1 D, 2 Ns, 3 Np, 4 Adj, 5 Adv, 6 C, 7 Crel, 8 Vin, 9 Vdp,
#                 10 Vgap(obj-gap V_dp), 11 Vcp
def _decode_pos(v):
    lc = v % 8; lcp = (v // 8) % 8; lcn = (v // 64) % 8
    lcnn = (v // 512) % 8; vintr = (v // 4096) % 2
    if lc == 6:                        # 'that': Crel iff previous token is N_singular
        return 7 if lcp == 2 else 6
    if lc == 7:                        # verb: look at next non-adverb token
        nn = lcnn if lcn == 5 else lcn
        if nn == 6:                    # ... that  -> V_cp
            return 11
        if nn in (1, 3):               # ... D / N_proper -> V_dp (has object)
            return 9
        return 8 if vintr else 10      # else intransitive, else obj-gap V_dp
    return lc                          # D/Ns/Np/Adj/Adv share codes 1..5


# per-verb clause-tail decoders, from packed local POS window p0..p5 (pos at v..v+5)
def _unpack6(v):
    return [(v >> (4 * k)) & 15 for k in range(6)]

def _verb_a(p):            # 1 if an adverb immediately follows the verb
    return 1 if p[1] == 5 else 0

def _jump_flag(v):
    """1 iff this verb's clause tail is a DEEPER clause (V_cp, or V_dp with a
    relative-clause object) whose close must be inherited; 0 = local tail."""
    p = _unpack6(v); pv = p[0]
    if pv not in (8, 9, 10, 11): return 0
    if pv in (8, 10): return 0                     # V_intrans / obj-gap: local
    if pv == 11: return 1                          # V_cp: jump to its CP_sent
    a = _verb_a(p); ps = p[1 + a]                  # POS at object start s=v+1+a
    if ps == 3: return 0                           # proper-name object: local
    if ps == 1:                                    # determiner-headed object DP
        adj = 1 if p[2 + a] == 4 else 0
        mp1 = p[(3 + a) + adj]                      # POS just after the core noun
        return 1 if mp1 == 7 else 0                # Crel there -> relative clause
    return 0

def _jump_off(v):
    """Offset (from the verb) of the 'that' whose close this verb's tail inherits."""
    p = _unpack6(v); pv = p[0]; a = _verb_a(p)
    if pv == 11: return 1 + a                       # the C 'that' at v+1+a
    if pv == 9:
        adj = 1 if p[2 + a] == 4 else 0
        return (3 + a) + adj                        # the Crel at m+1
    return 0

def _loc_off(v):
    """Offset (from the verb) of the clause-final token when the tail is LOCAL."""
    p = _unpack6(v); pv = p[0]; a = _verb_a(p)
    if pv in (8, 10): return a                       # verb (+ adverb)
    if pv == 9:
        ps = p[1 + a]
        if ps == 3: return 1 + a                      # proper-name object end
        adj = 1 if p[2 + a] == 4 else 0
        return (2 + a) + adj                          # core-noun end (no rel clause)
    return 0


# comp-start classifier: cls code at each position j (0 = not a comp-start)
#   1 F1 clause-start (after a 'that')       ce = close[j-1]  head 1
#   2 F2 V_cp complement (the C 'that')      ce = close[j]    head 1 or 2
#   3 F4 relative clause (the Crel 'that')   ce = close[j]    head 1 or 2
#   4 F2 V_dp object, no adverb              ce = vend[j-1]   head 1
#   5 F2 V_dp object, with adverb            ce = vend[j-2]   head 2
#   6 F3 adverb (VP_x_adv comp)              ce = j           head 1
def _decode_cls(v):
    pj = v % 16; prev = (v // 16) % 16; pp = (v // 256) % 16
    if prev in (6, 7): return 1
    if pj == 6:        return 2
    if pj == 7:        return 3
    if pj in (1, 3):
        if prev == 9:                       return 4
        if prev == 5 and pp == 9:           return 5
        return 0
    if pj == 5:        return 6
    return 0

def _decode_hsz(v):
    """Head size (1 or 2) of the flip whose comp starts here (0 if none)."""
    pj = v % 16; prev = (v // 16) % 16; pp = (v // 256) % 16
    cls = _decode_cls(v)
    if cls == 0: return 0
    if cls == 1 or cls == 4 or cls == 6: return 1
    if cls == 5: return 2
    if cls == 2: return 2 if prev == 5 else 1       # V_cp had an adverb -> 2
    if cls == 3: return 2 if pp == 4 else 1         # adj+noun core -> 2
    return 1


# --------------------------------------------------------------------------- #
# relative-position read (one attention head): out[i] = x[i + off], default dfl.
# causal=False: a read over the HI block, which is entirely in the decoder's past.
# --------------------------------------------------------------------------- #
def _shift(x, off, dfl=0):
    return kqv(indices(x), indices(x) + off, x, equals, default=dfl, causal=False)


def hi_to_hf(hi_tokens: list[str]) -> list[str]:
    n = len(hi_tokens)
    if n == 0:
        return []

    # 0. token-embedding: words -> integer ids -----------------------------------
    ids = np.array([_W2ID[w] for w in hi_tokens], dtype=int)   # input s-op

    # 1. lexical class of each token (token-embedding) ---------------------------
    lc   = tok_map(ids, _lex_code)                    # MLP: id -> lexical class
    vin  = tok_map(ids, _is_vintr)                    # MLP: id -> is lexically intrans

    # 2. fine POS by local resolution --------------------------------------------
    lcp  = _shift(lc, -1)                              # prev token's lexical class
    lcn  = _shift(lc, +1)                              # next token's lexical class
    lcnn = _shift(lc, +2)                              # next-next lexical class
    #    pack the 5 local features into one value, then one MLP decodes the POS
    pack = seq_map(lc, lcp, lambda a, b: a + 8 * b)             # lc + 8*prev
    pack = seq_map(pack, lcn, lambda a, b: a + 64 * b)          # + 64*next
    pack = seq_map(pack, lcnn, lambda a, b: a + 512 * b)        # + 512*nextnext
    pack = seq_map(pack, vin, lambda a, b: a + 4096 * b)        # + 4096*is_intrans
    pos  = tok_map(pack, _decode_pos)                  # MLP: -> fine POS code

    isthat = tok_map(pos, lambda p: 1 if p in (6, 7) else 0)    # 'that' opener?
    isverb = tok_map(pos, lambda p: 1 if p in (8, 9, 10, 11) else 0)

    # 3. that/verb balance  B[i] = (#that<=i) - (#verb<=i) ------------------------
    cumT = cumsum_incl(isthat)                         # prefix #that  (one head)
    cumV = cumsum_incl(isverb)                         # prefix #verb  (one head)
    B    = seq_map(cumT, cumV, lambda a, b: a - b)     # MLP: subtract

    # 4a. LEFTWARD match: each verb -> the 'that' it closes -----------------------
    #     moc[q] = last 'that' p<=q with B[p] == B[q]+1  (causal 'find latest open')
    keyB   = seq_map(isthat, B, lambda t, b: b if t else -999)  # B at thats, else sentinel
    Bplus1 = tok_map(B, lambda b: b + 1)               # query balance level
    moc0   = kqv(keyB, Bplus1, indices(pos), equals,
                 default=-1, reduction="max", causal=True)      # latest matching that
    moc    = seq_map(isverb, moc0, lambda v, m: m if v else -1) # keep only at verbs

    # 4b. invert: mv[p] = the verb q with moc[q] == p (unique) --------------------
    mv = kqv(moc, indices(pos), indices(pos), equals,
             default=-1, reduction="min", causal=False)         # main verb of clause p

    # 5. clause-end (interval close) by bounded pointer-jumping -------------------
    #    local tail geometry for every verb (packed 6-POS window -> MLP decoders)
    p1 = _shift(pos, 1); p2 = _shift(pos, 2); p3 = _shift(pos, 3)
    p4 = _shift(pos, 4); p5 = _shift(pos, 5)
    win = pos.copy()
    for k, pk in enumerate((p1, p2, p3, p4, p5), start=1):
        win = seq_map(win, pk, (lambda sh: (lambda a, b: a + (b << (4 * sh))))(k))
    jump = tok_map(win, _jump_flag)                    # tail is a deeper clause?
    joff = tok_map(win, _jump_off)                     # -> offset of that deeper 'that'
    loff = tok_map(win, _loc_off)                      # -> offset of local clause end
    jtar = seq_map(indices(pos), joff, lambda i, o: i + o)   # absolute jump target
    lloc = seq_map(indices(pos), loff, lambda i, o: i + o)   # absolute local end

    close_arr = full(pos, BIG)                         # close[that]  (unknown = BIG)
    vend_arr  = full(pos, BIG)                         # vend[verb]
    for _ in range(6):                                 # depth<=2 => converges fast
        g_close  = index_select(close_arr, jtar, default=BIG, causal=False)  # close@target
        vend_new = where(jump, g_close, lloc)          # jump -> inherit; else local
        vend_arr = where(isverb, vend_new, full(pos, BIG))                    # verbs only
        g_vend   = index_select(vend_arr, mv, default=BIG, causal=False)      # vend@mainverb
        close_arr = where(isthat, g_vend, full(pos, BIG))                     # thats only

    # 6. depth[i] = (#that<=i) - (#that with close < i) ---------------------------
    idx_m1     = tok_map(indices(pos), lambda i: i - 1)
    closed_lt  = sel_width(select(close_arr, idx_m1, leq, causal=False))  # #thats closed <i
    depth      = seq_map(cumT, closed_lt, lambda a, b: a - b)             # (kept for LOG)

    # 7. displacement = head-part (local) + comp-part (interval counting) ---------
    idx   = indices(pos)
    a_adv = tok_map(_shift(pos, 1), lambda p: 1 if p == 5 else 0)   # adverb right after?

    #    HEAD-part: a token in a flip node's HEAD child moves right by |comp|.
    #    Every such contribution is readable locally (<=2 per token).
    cnext  = _shift(close_arr, 1, dfl=BIG)             # close[i+1]
    cnext2 = _shift(close_arr, 2, dfl=BIG)             # close[i+2]
    vprev1 = _shift(vend_arr, -1, dfl=BIG)             # vend[i-1]
    posn1  = _shift(pos, 1); posn2 = _shift(pos, 2); posp1 = _shift(pos, -1)

    h_that = where(isthat, seq_map(close_arr, idx, lambda c, i: c - i), full(pos, 0))
    #    N_singular that heads a relative clause: +|CP_rel| = close[i+1]-i
    m_nrel = seq_map(pos, posn1, lambda p, q: 1 if (p == 2 and q == 7) else 0)
    h_nrel = where(m_nrel, seq_map(cnext, idx, lambda c, i: c - i), full(pos, 0))
    #    adjective of an adj+noun core that heads a relative clause: +|CP_rel|
    m_arel = seq_map(seq_map(pos, posn1, lambda p, q: 1 if (p == 4 and q == 2) else 0),
                     posn2, lambda ok, r: 1 if (ok and r == 7) else 0)
    h_arel = where(m_arel, seq_map(cnext2, idx, lambda c, i: c - i - 1), full(pos, 0))
    #    verb with a complement (V_dp obj / V_cp): +|complement| = vend[i]-i-a
    m_vcomp = tok_map(pos, lambda p: 1 if p in (9, 11) else 0)
    h_vcomp = where(m_vcomp,
                    seq_map(seq_map(vend_arr, idx, lambda v, i: v - i), a_adv,
                            lambda d, a: d - a),
                    full(pos, 0))
    #    verb that has an adverb: head of VP_x_adv, +|comp|=1
    m_vadv = seq_map(isverb, a_adv, lambda v, a: 1 if (v and a) else 0)
    h_vadv = where(m_vadv, full(pos, 1), full(pos, 0))
    #    adverb sitting in a V_dp/V_cp head-block: +|complement| = vend[i-1]-i
    m_advc = seq_map(pos, posp1, lambda p, q: 1 if (p == 5 and q in (9, 11)) else 0)
    h_advc = where(m_advc, seq_map(vprev1, idx, lambda v, i: v - i), full(pos, 0))

    head = h_that
    for term in (h_nrel, h_arel, h_vcomp, h_vadv, h_advc):
        head = seq_map(head, term, lambda a, b: a + b)

    #    COMP-part: a token in a flip's COMP child moves left by |head| (1 or 2).
    #    Each flip has a UNIQUE comp-start; identify it locally and read its comp-end.
    posp2 = _shift(pos, -2)
    clspk = seq_map(seq_map(pos, posp1, lambda a, b: a + 16 * b),
                    posp2, lambda a, b: a + 256 * b)         # (pos, prev, prevprev)
    cls   = tok_map(clspk, _decode_cls)                # comp-start class (0 = none)
    hsz   = tok_map(clspk, _decode_hsz)                # head size of that flip
    cs_b  = tok_map(cls, lambda c: 1 if c != 0 else 0) # is a comp-start
    hs2_b = seq_map(cs_b, hsz, lambda b, h: 1 if (b and h == 2) else 0)

    #    comp-end position for the flip that starts here (by class)
    cprev = _shift(close_arr, -1, dfl=BIG)             # close[j-1]
    vprv2 = _shift(vend_arr, -2, dfl=BIG)              # vend[j-2]
    ce = where(tok_map(cls, lambda c: 1 if c == 1 else 0), cprev,
         where(tok_map(cls, lambda c: 1 if c in (2, 3) else 0), close_arr,
         where(tok_map(cls, lambda c: 1 if c == 4 else 0), vprev1,
         where(tok_map(cls, lambda c: 1 if c == 5 else 0), vprv2,
         where(tok_map(cls, lambda c: 1 if c == 6 else 0), idx,
               full(pos, BIG))))))
    ceval  = where(cs_b, ce, full(pos, BIG))           # comp-end, else BIG
    ceval2 = where(hs2_b, ce, full(pos, BIG))          # same, head-size-2 flips only

    #    count of flips whose comp covers i  =  (#comp-starts<=i) - (#comp-ends<i)
    cumCS  = cumsum_incl(cs_b)                          # comp-starts up to i
    endsB  = sel_width(select(ceval, idx_m1, leq, causal=False))   # comp-ends < i
    count1 = seq_map(cumCS, endsB, lambda a, b: a - b)
    cumCS2 = cumsum_incl(hs2_b)
    endsB2 = sel_width(select(ceval2, idx_m1, leq, causal=False))
    count2 = seq_map(cumCS2, endsB2, lambda a, b: a - b)
    comp   = seq_map(count1, count2, lambda a, b: -(a + b))   # sum of -|head| over flips

    disp   = seq_map(head, comp, lambda a, b: a + b)
    hf_pos = seq_map(idx, disp, lambda i, d: i + d)    # target slot of HI token i

    # 8. gather: out[j] = hi[i] where hf_pos[i] == j  (inverse permutation) -------
    inv     = kqv(hf_pos, idx, idx, equals, default=0, causal=False)   # inverse perm
    out_ids = index_select(ids, inv, causal=False)     # place HI tokens into HF order
    return [_ID2W[int(w)] for w in out_ids]

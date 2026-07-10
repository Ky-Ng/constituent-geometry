"""Step-by-step derivations of EVERY variable in _s2_closer.hi_to_hf.

Emits (a) a text derivation you can diff against a hand-trace, and (b) a
self-contained HTML visualizer with Prev/Next buttons (a pdb-style stepper).
All values are asserted equal to the real module and to grammar_oracle.

    uv run python _closer_viz.py                                  # default sentence
    uv run python _closer_viz.py "John knows that Mary swims"
    uv run python _closer_viz.py "..." --max-rounds 10            # print every round
    uv run python _closer_viz.py "..." --quiet                    # html only

Then open closer_viz.html in a browser (scp it off the cluster, or
`python -m http.server` from this directory).
"""
from __future__ import annotations
import os, sys, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

# This script must run non-interactively, but _s2_closer.py may have a breakpoint()
# parked in it while you hand-trace. Neutralize breakpoint() for this process only.
os.environ.setdefault("PYTHONBREAKPOINT", "0")
sys.breakpointhook = lambda *a, **kw: None

import numpy as np

import _s2_closer as C
from _s2_closer import _lexclass, _is_intrans_lex, _read_off, _stab_ge, _W2I, _I2W
from rasp_l import tok_map, index_select, indices, where, full, kqv, equals
from grammar_oracle import oracle_features

LCN = {0: "D", 1: "N_sing", 2: "N_prop", 3: "Adj", 4: "Adv", 5: "THAT", 6: "V", 7: "OTHER", -1: "·"}
ROLEN = {0: "–", 1: "intr", 2: "+DP", 3: "+CP", 4: "objgap"}


def build(sentence: str):
    toks = sentence.split()
    n = len(toks)
    hi = np.array([_W2I[w] for w in toks], dtype=int)
    idx = indices(hi)
    ZERO = full(hi, 0)

    STEPS = []

    def rec(name, expr, vals, derivs, stage, fmt="int", note="", rnd=None):
        STEPS.append(dict(stage=stage, round=rnd, name=name, expr=expr, fmt=fmt, note=note,
                          vals=[int(v) for v in np.asarray(vals, dtype=int)], derivs=derivs))

    # ---------- derivation builders ----------
    def d_gather(name, xn, x, idxn, iv, default):
        out = []
        for i in range(n):
            t = int(iv[i])
            if 0 <= t < n:
                out.append(f"{name}[{i}] = {xn}[{idxn}[{i}]={t}] = {int(x[t])}")
            else:
                out.append(f"{name}[{i}] = {xn}[{t}] → out of range ⇒ default {default}")
        return out

    def d_read(name, xn, x, off, default):
        out = []
        for i in range(n):
            t = i + off
            if 0 <= t < n:
                out.append(f"{name}[{i}] = {xn}[{i}{off:+d}] = {xn}[{t}] = {int(x[t])}")
            else:
                out.append(f"{name}[{i}] = {xn}[{t}] → out of range ⇒ {default}")
        return out

    def d_where(name, cn, c, an, a, bn, b):
        out = []
        for i in range(n):
            if c[i]:
                out.append(f"{name}[{i}]: {cn}[{i}]=1 ⇒ take {an}[{i}] = {int(a[i])}")
            else:
                out.append(f"{name}[{i}]: {cn}[{i}]=0 ⇒ take {bn}[{i}] = {int(b[i])}")
        return out

    def d_stab(name, en, ev):
        out = []
        for i in range(n):
            js = [j for j in range(i + 1) if int(ev[j]) >= i]
            s = ", ".join(map(str, js)) if js else "∅"
            out.append(f"{name}[{i}] = #{{j ≤ {i} : {en}[j] ≥ {i}}} = |{{{s}}}| = {len(js)}")
        return out

    def d_each(fn):
        return [fn(i) for i in range(n)]

    S1, S2A, S2B, S2C, S3 = ("Stage 1 — local POS/role", "Stage 2 — scaffolding",
                             "Stage 2 — relaxation", "Stage 2 — depth & displacement",
                             "Stage 3 — assemble & gather")

    # ================= Stage 1 =================
    rec("hi_ids", "np.array([_W2I[w] for w in hi_tokens])", hi,
        d_each(lambda i: f"hi_ids[{i}] = _W2I['{toks[i]}'] = {int(hi[i])}"), S1,
        note="Fixed word→id lookup (a token embedding). Repeated words share an id.")
    rec("idx", "indices(hi_ids)", idx, d_each(lambda i: f"idx[{i}] = {i}"), S1,
        note="Position s-op. The final answer is idx + disp.")

    lc = tok_map(hi, _lexclass)
    rec("lc", "tok_map(hi_ids, _lexclass)", lc,
        d_each(lambda i: f"lc[{i}] = _lexclass(hi_ids[{i}]={int(hi[i])}) = {LCN[int(lc[i])]}({int(lc[i])})"),
        S1, fmt="lc", note="id → coarse POS class.")

    is_intr = tok_map(hi, _is_intrans_lex)
    rec("is_intr", "tok_map(hi_ids, _is_intrans_lex)", is_intr,
        d_each(lambda i: f"is_intr[{i}] = ('{toks[i]}' ∈ V_intrans) = {int(is_intr[i])}"), S1, fmt="bool")

    prev_lc = _read_off(lc, -1);  rec("prev_lc", "_read_off(lc, -1)  # lc[i-1]", prev_lc, d_read("prev_lc", "lc", lc, -1, -1), S1, fmt="lc")
    prev2_lc = _read_off(lc, -2); rec("prev2_lc", "_read_off(lc, -2)  # lc[i-2]", prev2_lc, d_read("prev2_lc", "lc", lc, -2, -1), S1, fmt="lc")
    next_lc = _read_off(lc, +1);  rec("next_lc", "_read_off(lc, +1)  # lc[i+1]", next_lc, d_read("next_lc", "lc", lc, +1, -1), S1, fmt="lc",
                                      note="Forward read — legal only because the whole HI block precedes the output region.")
    next2_lc = _read_off(lc, +2); rec("next2_lc", "_read_off(lc, +2)  # lc[i+2]", next2_lc, d_read("next2_lc", "lc", lc, +2, -1), S1, fmt="lc")

    isthat = (lc == C.THAT)
    rec("isthat", "(lc == THAT)", isthat, d_each(lambda i: f"isthat[{i}] = (lc[{i}]={LCN[int(lc[i])]} == THAT) = {int(isthat[i])}"), S1, fmt="bool",
        note="The clause OPENERS — the only visible bracket-starts.")
    is_crel = isthat & (prev_lc == C.NSING)
    rec("is_crel", "isthat & (prev_lc == NSING)", is_crel,
        d_each(lambda i: f"is_crel[{i}] = isthat[{i}]={int(isthat[i])} & (prev_lc[{i}]={LCN[int(prev_lc[i])]} == N_sing) = {int(is_crel[i])}"), S1, fmt="bool",
        note="'that' right after a common noun ⇒ RELATIVE complementizer.")
    is_c = isthat & ~is_crel
    rec("is_c", "isthat & ~is_crel", is_c,
        d_each(lambda i: f"is_c[{i}] = isthat[{i}]={int(isthat[i])} & not is_crel[{i}]={int(is_crel[i])} = {int(is_c[i])}"), S1, fmt="bool",
        note="Otherwise a SENTENTIAL complementizer.")
    isverb = (lc == C.V); rec("isverb", "(lc == V)", isverb, d_each(lambda i: f"isverb[{i}] = (lc[{i}]={LCN[int(lc[i])]} == V) = {int(isverb[i])}"), S1, fmt="bool")
    isadv = (lc == C.ADV); rec("isadv", "(lc == ADV)", isadv, d_each(lambda i: f"isadv[{i}] = (lc[{i}]={LCN[int(lc[i])]} == Adv) = {int(isadv[i])}"), S1, fmt="bool")

    nn_lc = where(next_lc == C.ADV, next2_lc, next_lc)
    rec("nn_lc", "where(next_lc == ADV, next2_lc, next_lc)", nn_lc,
        d_each(lambda i: (f"nn_lc[{i}]: next_lc[{i}]=Adv ⇒ skip adverb, take next2_lc[{i}] = {LCN[int(next2_lc[i])]}"
                          if next_lc[i] == C.ADV else
                          f"nn_lc[{i}]: next_lc[{i}]={LCN[int(next_lc[i])]} (not Adv) ⇒ keep it")), S1, fmt="lc",
        note="'next non-adverb class': lets a verb see its complement's first token across an adverb.")

    role = where(isverb, where(nn_lc == C.THAT, full(hi, 3),
                 where((nn_lc == C.D) | (nn_lc == C.NPROP), full(hi, 2),
                 where(is_intr == 1, full(hi, 1), full(hi, 4)))), ZERO)

    def d_role(i):
        if not isverb[i]:
            return f"role[{i}]: isverb=0 ⇒ 0 (non-verb)"
        if nn_lc[i] == C.THAT:
            return f"role[{i}]: verb, nn_lc=THAT ⇒ 3 (+clause / V_cp)"
        if nn_lc[i] in (C.D, C.NPROP):
            return f"role[{i}]: verb, nn_lc={LCN[int(nn_lc[i])]} (DP start) ⇒ 2 (+object DP / V_dp)"
        if is_intr[i] == 1:
            return f"role[{i}]: verb, no complement, lexically intransitive ⇒ 1 (intrans)"
        return f"role[{i}]: verb, no complement, NOT lexically intrans ⇒ 4 (object-gap)"
    rec("role", "where(isverb, where(nn_lc==THAT,3, where(nn_lc∈{D,NPROP},2, where(is_intr,1,4))), 0)",
        role, d_each(d_role), S1, fmt="role",
        note="Roles 2 & 3 head a FLIP VP (they have a complement); 1 & 4 never flip.")

    # ================= Stage 2 scaffolding =================
    hb_end = where(isverb & (next_lc == C.ADV), idx + 1, idx)
    rec("hb_end", "where(isverb & (next_lc==ADV), idx+1, idx)", hb_end,
        d_each(lambda i: (f"hb_end[{i}]: verb followed by adverb ⇒ i+1 = {int(hb_end[i])}"
                          if (isverb[i] and next_lc[i] == C.ADV) else
                          f"hb_end[{i}]: no verb+adverb block ⇒ idx[{i}] = {i}")), S2A,
        note="End of the verb's HEAD block (the head side of a VP flip).")

    dp_end_flat = where(lc == C.D, where(next_lc == C.ADJ, idx + 2, idx + 1), idx)
    def d_dpf(i):
        if lc[i] != C.D:
            return f"dp_end_flat[{i}]: not a determiner ⇒ idx[{i}] = {i}"
        if next_lc[i] == C.ADJ:
            return f"dp_end_flat[{i}]: D + Adj + N ⇒ i+2 = {i+2}"
        return f"dp_end_flat[{i}]: D + N ⇒ i+1 = {i+1}"
    rec("dp_end_flat", "where(lc==D, where(next_lc==ADJ, idx+2, idx+1), idx)", dp_end_flat,
        d_each(d_dpf), S2A, note="Where a DP ends IGNORING any relative clause.")

    rc_that_pos = dp_end_flat + 1
    rec("rc_that_pos", "dp_end_flat + 1", rc_that_pos,
        d_each(lambda i: f"rc_that_pos[{i}] = dp_end_flat[{i}]={int(dp_end_flat[i])} + 1 = {int(rc_that_pos[i])}  (slot where a relative 'that' would sit)"), S2A)

    lc_at_rc = index_select(lc, rc_that_pos, default=-1, causal=False)
    rec("lc_at_rc", "index_select(lc, rc_that_pos, default=-1)", lc_at_rc,
        d_gather("lc_at_rc", "lc", lc, "rc_that_pos", rc_that_pos, -1), S2A, fmt="lc",
        note="One cross-position lookahead: what class actually sits in that slot?")

    dp_has_rc = (lc == C.D) & (lc_at_rc == C.THAT)
    rec("dp_has_rc", "(lc == D) & (lc_at_rc == THAT)", dp_has_rc,
        d_each(lambda i: f"dp_has_rc[{i}] = (lc[{i}]={LCN[int(lc[i])]}==D) & (lc_at_rc[{i}]={LCN[int(lc_at_rc[i])]}==THAT) = {int(dp_has_rc[i])}"),
        S2A, fmt="bool", note="This DP carries a relative clause.")

    # ================= Stage 2 relaxation =================
    dpfull = dp_end_flat.copy(); vpe = hb_end.copy(); rce = idx.copy(); cpe = idx.copy()
    rec("dpfull (seed)", "dpfull = dp_end_flat", dpfull, d_each(lambda i: f"dpfull[{i}] ← dp_end_flat[{i}] = {int(dpfull[i])}"), S2B, rnd=0)
    rec("vpe (seed)", "vpe = hb_end", vpe, d_each(lambda i: f"vpe[{i}] ← hb_end[{i}] = {int(vpe[i])}"), S2B, rnd=0)
    rec("rce (seed)", "rce = idx", rce, d_each(lambda i: f"rce[{i}] ← idx[{i}] = {i}  ('ends at itself')"), S2B, rnd=0)
    rec("cpe (seed)", "cpe = idx", cpe, d_each(lambda i: f"cpe[{i}] ← idx[{i}] = {i}  ('ends at itself')"), S2B, rnd=0)

    for r in range(1, C.ROUNDS + 1):
        rce_at_rc = index_select(rce, rc_that_pos, default=0, causal=False)
        rec("rce_at_rc", "index_select(rce, rc_that_pos)", rce_at_rc,
            d_gather("rce_at_rc", "rce", rce, "rc_that_pos", rc_that_pos, 0), S2B, rnd=r,
            note="Fetch the relative clause's end FROM the 'that' slot, back to the DP's start.")
        prev_dpfull = dpfull.copy()
        dpfull = where(dp_has_rc, rce_at_rc, dp_end_flat)
        rec("dpfull", "where(dp_has_rc, rce_at_rc, dp_end_flat)", dpfull,
            d_each(lambda i: (f"dpfull[{i}]: dp_has_rc=1 ⇒ rce_at_rc[{i}] = {int(rce_at_rc[i])}  (= rce[{int(rc_that_pos[i])}])"
                              if dp_has_rc[i] else
                              f"dpfull[{i}]: no rel. clause ⇒ dp_end_flat[{i}] = {int(dp_end_flat[i])}")), S2B, rnd=r)

        hb1 = hb_end + 1
        rec("hb1", "hb_end + 1", hb1, d_each(lambda i: f"hb1[{i}] = hb_end[{i}]={int(hb_end[i])} + 1 = {int(hb1[i])}  (where the complement starts)"), S2B, rnd=r)
        dpfull_at_hb1 = index_select(dpfull, hb1, default=0, causal=False)
        rec("dpfull_at_hb1", "index_select(dpfull, hb1)", dpfull_at_hb1,
            d_gather("dpfull_at_hb1", "dpfull", dpfull, "hb1", hb1, 0), S2B, rnd=r,
            note="For a verb: the end of the object DP that starts right after its head block.")
        cpe_at_hb1 = index_select(cpe, hb1, default=0, causal=False)
        rec("cpe_at_hb1", "index_select(cpe, hb1)", cpe_at_hb1,
            d_gather("cpe_at_hb1", "cpe", cpe, "hb1", hb1, 0), S2B, rnd=r,
            note="For a clausal verb: the end of the CP_sent that starts right after it.")

        vpe = where((role == 1) | (role == 4), hb_end,
              where(role == 2, dpfull_at_hb1, where(role == 3, cpe_at_hb1, idx)))
        def d_vpe(i, _hb1=hb1, _d=dpfull_at_hb1, _c=cpe_at_hb1):
            if role[i] in (1, 4):
                return f"vpe[{i}]: role={int(role[i])} (no complement) ⇒ hb_end[{i}] = {int(hb_end[i])}"
            if role[i] == 2:
                return f"vpe[{i}]: role=2 (+DP) ⇒ dpfull_at_hb1[{i}] = {int(_d[i])}  (= dpfull[{int(_hb1[i])}])"
            if role[i] == 3:
                return f"vpe[{i}]: role=3 (+CP) ⇒ cpe_at_hb1[{i}] = {int(_c[i])}  (= cpe[{int(_hb1[i])}])"
            return f"vpe[{i}]: not a verb ⇒ idx[{i}] = {i}"
        rec("vpe", "where(role∈{1,4}, hb_end, where(role==2, dpfull_at_hb1, where(role==3, cpe_at_hb1, idx)))",
            vpe, d_each(d_vpe), S2B, rnd=r, note="A VP ends where its complement ends.")

        q1 = idx + 1
        vpe_at_q1 = index_select(vpe, q1, default=0, causal=False)
        rec("vpe_at_q1", "index_select(vpe, idx+1)", vpe_at_q1, d_gather("vpe_at_q1", "vpe", vpe, "q1", q1, 0), S2B, rnd=r)
        dpfull_at_q1 = index_select(dpfull, q1, default=0, causal=False)
        rec("dpfull_at_q1", "index_select(dpfull, idx+1)", dpfull_at_q1, d_gather("dpfull_at_q1", "dpfull", dpfull, "q1", q1, 0), S2B, rnd=r)

        rce = where(is_crel, where(next_lc == C.V, vpe_at_q1, dpfull_at_q1 + 1), idx)
        def d_rce(i, _v=vpe_at_q1, _d=dpfull_at_q1):
            if not is_crel[i]:
                return f"rce[{i}]: not a relative 'that' ⇒ idx[{i}] = {i}"
            if next_lc[i] == C.V:
                return f"rce[{i}]: SUBJECT-gap (next is a verb) ⇒ vpe_at_q1[{i}] = {int(_v[i])}  (= vpe[{i+1}])"
            return f"rce[{i}]: OBJECT-gap ⇒ dpfull_at_q1[{i}]+1 = {int(_d[i])}+1 = {int(_d[i])+1}  (the gap verb)"
        rec("rce", "where(is_crel, subj? vpe_at_q1 : dpfull_at_q1+1, idx)", rce, d_each(d_rce), S2B, rnd=r)

        vpos = dpfull_at_q1 + 1
        rec("vpos", "dpfull_at_q1 + 1", vpos,
            d_each(lambda i: f"vpos[{i}] = dpfull_at_q1[{i}]={int(dpfull_at_q1[i])} + 1 = {int(vpos[i])}  (embedded VP start, past the subject DP)"), S2B, rnd=r)
        vpe_at_vpos = index_select(vpe, vpos, default=0, causal=False)
        rec("vpe_at_vpos", "index_select(vpe, vpos)", vpe_at_vpos, d_gather("vpe_at_vpos", "vpe", vpe, "vpos", vpos, 0), S2B, rnd=r)

        cpe = where(is_c, vpe_at_vpos, idx)
        def d_cpe(i, _v=vpe_at_vpos, _p=vpos):
            if not is_c[i]:
                return f"cpe[{i}]: not a sentential 'that' ⇒ idx[{i}] = {i}"
            return f"cpe[{i}]: sentential ⇒ vpe_at_vpos[{i}] = {int(_v[i])}  (= vpe[{int(_p[i])}])"
        rec("cpe", "where(is_c, vpe_at_vpos, idx)", cpe, d_each(d_cpe), S2B, rnd=r)

    # ================= depth & displacement =================
    e = where(is_crel, rce, where(is_c, cpe, idx))
    rec("e", "where(is_crel, rce, where(is_c, cpe, idx))", e,
        d_each(lambda i: (f"e[{i}]: relative 'that' ⇒ rce[{i}] = {int(rce[i])}" if is_crel[i] else
                          (f"e[{i}]: sentential 'that' ⇒ cpe[{i}] = {int(cpe[i])}" if is_c[i] else
                           f"e[{i}]: not an opener ⇒ idx[{i}] = {i}"))), S2C)
    end_that = where(isthat, e, full(hi, -1))
    rec("end_that", "where(isthat, e, -1)", end_that,
        d_each(lambda i: (f"end_that[{i}]: opener ⇒ e[{i}] = {int(e[i])}" if isthat[i] else f"end_that[{i}]: not an opener ⇒ -1 (carries no interval)")), S2C)
    depth = _stab_ge(end_that)
    rec("depth", "sel_width(select(end_that, indices, geq, causal=True))", depth,
        d_stab("depth", "end_that", end_that), S2C, note="Stabbing count: how many opener-intervals [q, e(q)] cover position i.")

    d_adv = where(isverb & (next_lc == C.ADV), full(hi, 1), ZERO) + where(isadv, full(hi, -1), ZERO)
    rec("d_adv", "where(isverb & next==ADV, +1, 0) + where(isadv, -1, 0)", d_adv,
        d_each(lambda i: f"d_adv[{i}] = (verb before adverb? {int(bool(isverb[i] and next_lc[i]==C.ADV))}) + (is adverb? {-int(isadv[i])}) = {int(d_adv[i])}"), S2C)
    span_clause = e - idx + 1
    rec("span_clause", "e - idx + 1", span_clause,
        d_each(lambda i: f"span_clause[{i}] = e[{i}]={int(e[i])} - {i} + 1 = {int(span_clause[i])}" + ("  (only meaningful at an opener)" if not isthat[i] else "")), S2C)
    d_cp = -depth + where(isthat, span_clause, ZERO)
    rec("d_cp", "-depth + where(isthat, span_clause, 0)", d_cp,
        d_each(lambda i: f"d_cp[{i}] = -depth[{i}]={-int(depth[i])}" + (f" + span_clause[{i}]={int(span_clause[i])}" if isthat[i] else "") + f" = {int(d_cp[i])}"), S2C)

    e_at_p1 = index_select(e, idx + 1, default=0, causal=False)
    rec("e_at_p1", "index_select(e, idx+1)", e_at_p1, d_gather("e_at_p1", "e", e, "idx+1", idx + 1, 0), S2C)
    e_at_p2 = index_select(e, idx + 2, default=0, causal=False)
    rec("e_at_p2", "index_select(e, idx+2)", e_at_p2, d_gather("e_at_p2", "e", e, "idx+2", idx + 2, 0), S2C)
    core_noun_bonus = where((lc == C.NSING) & (next_lc == C.THAT), e_at_p1 - (idx + 1) + 1, ZERO)
    rec("core_noun_bonus", "where(lc==N & next==THAT, e_at_p1-(idx+1)+1, 0)", core_noun_bonus,
        d_each(lambda i: (f"core_noun_bonus[{i}]: noun before a relative 'that' ⇒ |RC| = e[{i+1}]={int(e_at_p1[i])} - {i+1} + 1 = {int(core_noun_bonus[i])}"
                          if (lc[i] == C.NSING and next_lc[i] == C.THAT) else f"core_noun_bonus[{i}] = 0")), S2C)
    core_adj_bonus = where((lc == C.ADJ) & (next_lc == C.NSING) & (next2_lc == C.THAT), e_at_p2 - (idx + 2) + 1, ZERO)
    rec("core_adj_bonus", "where(lc==Adj & next==N & next2==THAT, e_at_p2-(idx+2)+1, 0)", core_adj_bonus,
        d_each(lambda i: (f"core_adj_bonus[{i}]: adjective of an Adj-N core with a relative clause ⇒ {int(core_adj_bonus[i])}"
                          if (lc[i] == C.ADJ and next_lc[i] == C.NSING and next2_lc[i] == C.THAT) else f"core_adj_bonus[{i}] = 0")), S2C)
    d_nprel_core = core_noun_bonus + core_adj_bonus
    rec("d_nprel_core", "core_noun_bonus + core_adj_bonus", d_nprel_core,
        d_each(lambda i: f"d_nprel_core[{i}] = {int(core_noun_bonus[i])} + {int(core_adj_bonus[i])} = {int(d_nprel_core[i])}"), S2C)
    has_adj_core = (prev2_lc == C.ADJ)
    rec("has_adj_core", "(prev2_lc == ADJ)", has_adj_core,
        d_each(lambda i: f"has_adj_core[{i}] = (prev2_lc[{i}]={LCN[int(prev2_lc[i])]} == Adj) = {int(has_adj_core[i])}"), S2C, fmt="bool")
    crel_end_w1 = where(is_crel & ~has_adj_core, e, full(hi, -1))
    rec("crel_end_w1", "where(is_crel & ~has_adj_core, e, -1)", crel_end_w1,
        d_each(lambda i: (f"crel_end_w1[{i}]: relative 'that', |core|=1 ⇒ e[{i}] = {int(e[i])}" if (is_crel[i] and not has_adj_core[i]) else f"crel_end_w1[{i}] = -1 (masked)")), S2C)
    crel_end_w2 = where(is_crel & has_adj_core, e, full(hi, -1))
    rec("crel_end_w2", "where(is_crel & has_adj_core, e, -1)", crel_end_w2,
        d_each(lambda i: (f"crel_end_w2[{i}]: relative 'that', |core|=2 ⇒ e[{i}] = {int(e[i])}" if (is_crel[i] and has_adj_core[i]) else f"crel_end_w2[{i}] = -1 (masked)")), S2C)
    s_w1 = _stab_ge(crel_end_w1); s_w2 = _stab_ge(crel_end_w2)
    d_nprel_inside = -(s_w1 + 2 * s_w2)
    rec("d_nprel_inside", "-(stab(crel_end_w1) + 2*stab(crel_end_w2))", d_nprel_inside,
        d_each(lambda i: f"d_nprel_inside[{i}] = -({int(s_w1[i])} + 2·{int(s_w2[i])}) = {int(d_nprel_inside[i])}   (tokens inside a rel. clause lose |core|)"), S2C)
    d_nprel = d_nprel_core + d_nprel_inside
    rec("d_nprel", "d_nprel_core + d_nprel_inside", d_nprel,
        d_each(lambda i: f"d_nprel[{i}] = {int(d_nprel_core[i])} + {int(d_nprel_inside[i])} = {int(d_nprel[i])}"), S2C)

    comp_len = vpe - hb_end
    rec("comp_len", "vpe - hb_end", comp_len,
        d_each(lambda i: f"comp_len[{i}] = vpe[{i}]={int(vpe[i])} - hb_end[{i}]={int(hb_end[i])} = {int(comp_len[i])}"), S2C)
    is_vpflip = (role == 2) | (role == 3)
    rec("is_vpflip", "(role==2) | (role==3)", is_vpflip,
        d_each(lambda i: f"is_vpflip[{i}] = (role[{i}]={int(role[i])} ∈ {{2,3}}) = {int(is_vpflip[i])}"), S2C, fmt="bool")
    prev_role = _read_off(role, -1, default=0);   rec("prev_role", "_read_off(role, -1)", prev_role, d_read("prev_role", "role", role, -1, 0), S2C, fmt="role")
    role_prev2 = _read_off(role, -2, default=0);  rec("role_prev2", "_read_off(role, -2)", role_prev2, d_read("role_prev2", "role", role, -2, 0), S2C, fmt="role")
    comp_len_prev = _read_off(comp_len, -1, default=0); rec("comp_len_prev", "_read_off(comp_len, -1)", comp_len_prev, d_read("comp_len_prev", "comp_len", comp_len, -1, 0), S2C)
    d_vp_head = where(is_vpflip, comp_len, ZERO) + where(isadv & ((prev_role == 2) | (prev_role == 3)), comp_len_prev, ZERO)
    rec("d_vp_head", "where(is_vpflip, comp_len, 0) + where(isadv & prev_is_vpflip, comp_len_prev, 0)", d_vp_head,
        d_each(lambda i: f"d_vp_head[{i}] = (flip verb? {int(comp_len[i]) if is_vpflip[i] else 0}) + (adverb of a flip verb? {int(comp_len_prev[i]) if (isadv[i] and prev_role[i] in (2,3)) else 0}) = {int(d_vp_head[i])}"), S2C)
    vpe_prev = _read_off(vpe, -1, default=0);   rec("vpe_prev", "_read_off(vpe, -1)", vpe_prev, d_read("vpe_prev", "vpe", vpe, -1, 0), S2C)
    vpe_prev2 = _read_off(vpe, -2, default=0);  rec("vpe_prev2", "_read_off(vpe, -2)", vpe_prev2, d_read("vpe_prev2", "vpe", vpe, -2, 0), S2C)
    w1_here = ((prev_role == 2) | (prev_role == 3)) & (lc != C.ADV)
    rec("w1_here", "(prev_role ∈ {2,3}) & (lc != ADV)", w1_here,
        d_each(lambda i: f"w1_here[{i}] = (prev_role[{i}]={int(prev_role[i])} ∈ {{2,3}}) & (not adverb) = {int(w1_here[i])}   (complement start, head size 1)"), S2C, fmt="bool")
    w2_here = ((role_prev2 == 2) | (role_prev2 == 3)) & (prev_lc == C.ADV)
    rec("w2_here", "(role_prev2 ∈ {2,3}) & (prev_lc == ADV)", w2_here,
        d_each(lambda i: f"w2_here[{i}] = (role[{i}-2]={int(role_prev2[i])} ∈ {{2,3}}) & (prev is adverb) = {int(w2_here[i])}   (complement start, head size 2)"), S2C, fmt="bool")
    cs_end_w1 = where(w1_here, vpe_prev, full(hi, -1))
    rec("cs_end_w1", "where(w1_here, vpe_prev, -1)", cs_end_w1,
        d_each(lambda i: (f"cs_end_w1[{i}]: complement start ⇒ vpe[{i-1}] = {int(vpe_prev[i])}" if w1_here[i] else f"cs_end_w1[{i}] = -1 (masked)")), S2C)
    cs_end_w2 = where(w2_here, vpe_prev2, full(hi, -1))
    rec("cs_end_w2", "where(w2_here, vpe_prev2, -1)", cs_end_w2,
        d_each(lambda i: (f"cs_end_w2[{i}]: complement start (verb+adv) ⇒ vpe[{i-2}] = {int(vpe_prev2[i])}" if w2_here[i] else f"cs_end_w2[{i}] = -1 (masked)")), S2C)
    c1 = _stab_ge(cs_end_w1); c2 = _stab_ge(cs_end_w2)
    d_vp_comp = -(c1 + 2 * c2)
    rec("d_vp_comp", "-(stab(cs_end_w1) + 2*stab(cs_end_w2))", d_vp_comp,
        d_each(lambda i: f"d_vp_comp[{i}] = -({int(c1[i])} + 2·{int(c2[i])}) = {int(d_vp_comp[i])}   (comp tokens lose |head| per enclosing VP flip)"), S2C)
    d_vp = d_vp_head + d_vp_comp
    rec("d_vp", "d_vp_head + d_vp_comp", d_vp,
        d_each(lambda i: f"d_vp[{i}] = {int(d_vp_head[i])} + {int(d_vp_comp[i])} = {int(d_vp[i])}"), S2C)

    # ================= Stage 3 =================
    disp = d_adv + d_cp + d_nprel + d_vp
    rec("disp", "d_adv + d_cp + d_nprel + d_vp", disp,
        d_each(lambda i: f"disp[{i}] = {int(d_adv[i])} + {int(d_cp[i])} + {int(d_nprel[i])} + {int(d_vp[i])} = {int(disp[i])}"), S3)
    hf_pos = idx + disp
    rec("hf_pos", "idx + disp", hf_pos,
        d_each(lambda i: f"hf_pos[{i}] = {i} + {int(disp[i])} = {int(hf_pos[i])}   ('{toks[i]}' targets slot {int(hf_pos[i])})"), S3)
    src = kqv(hf_pos, idx, idx, equals, default=0, causal=False)
    rec("src", "kqv(hf_pos, idx, idx, equals)   # inverse permutation", src,
        d_each(lambda j: f"src[{j}] = the i with hf_pos[i] == {j}  ⇒  i = {int(src[j])}  ('{toks[int(src[j])]}')"), S3,
        note="Slot j asks: which token targets me? Exactly one does (hf_pos is a bijection).")
    out_ids = index_select(hi, src, default=0, causal=False)
    rec("out_ids", "index_select(hi_ids, src)", out_ids,
        d_each(lambda j: f"out_ids[{j}] = hi_ids[src[{j}]={int(src[j])}] = {int(out_ids[j])}  ('{_I2W[int(out_ids[j])]}')"), S3)

    out_words = [_I2W[int(t)] for t in out_ids]

    # ---------------- correctness guards ----------------
    assert out_words == C.hi_to_hf(toks), "trace diverged from _s2_closer!"
    orc = oracle_features(toks)
    assert out_words == orc["hf_tokens"], "module disagrees with oracle!"
    assert [int(x) for x in depth] == orc["depth"], "depth disagrees with oracle!"
    assert [int(x) for x in disp] == orc["disp"], "disp disagrees with oracle!"

    return toks, [int(x) for x in hi], STEPS, out_words


# ============================== text output ==============================
def print_text(toks, hi_ids, STEPS, out_words, max_rounds):
    n = len(toks)
    print("=" * 78)
    print("HI:", " ".join(toks))
    print("HF:", " ".join(out_words))
    print("positions:", "  ".join(f"{i}:{t}" for i, t in enumerate(toks)))
    print("=" * 78)
    stage = None
    shown_round = None
    for s in STEPS:
        if s["round"] and s["round"] > max_rounds:
            if shown_round != "skip":
                print(f"\n  … rounds {max_rounds+1}–{C.ROUNDS} omitted (use --max-rounds to show; "
                      f"they are the fixed point) …")
                shown_round = "skip"
            continue
        if s["stage"] != stage:
            stage = s["stage"]; print(f"\n{'═'*4} {stage} {'═'*4}")
        tag = f"  [round {s['round']}]" if s["round"] else ""
        print(f"\n{s['name']}{tag}\n    {s['expr']}")
        if s["note"]:
            print(f"    note: {s['note']}")
        print(f"    values: {s['vals']}")
        for line in s["derivs"]:
            print(f"      {line}")
    print("\n" + "=" * 78)
    print("✓ all values verified against _s2_closer.hi_to_hf and grammar_oracle.py")


# ============================== html output ==============================
HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>_s2_closer step-through</title><style>
:root{--bg:#fff;--fg:#1c1c1c;--mut:#666;--line:#ddd;--hl:#fff6d6;--chg:#c0392b;--acc:#1f5fbf;--pan:#f7f7f4}
@media(prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e6e6e6;--mut:#9aa0a6;--line:#333;--hl:#3a3520;--chg:#ff7b6b;--acc:#7aa7ff;--pan:#1e2126}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:20px 24px;max-width:1200px}
h1{font-size:20px;margin:0 0 4px} .sub{color:var(--mut);font-size:13px;margin-bottom:14px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.bar{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--line);z-index:5;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button{font:inherit;padding:5px 11px;border:1px solid var(--line);background:var(--pan);color:var(--fg);border-radius:6px;cursor:pointer}
button:hover{border-color:var(--acc)} button:disabled{opacity:.4;cursor:default}
#slider{flex:1;min-width:200px} #counter{color:var(--mut);font-size:13px;min-width:110px}
.wrap{overflow-x:auto;margin-top:14px}
table{border-collapse:collapse;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}
th,td{border:1px solid var(--line);padding:3px 9px;text-align:center;white-space:nowrap}
th.lbl,td.lbl{text-align:left;font-weight:600;position:sticky;left:0;background:var(--bg);z-index:2}
thead th{background:var(--pan)} .pos{color:var(--mut);font-weight:400;font-size:11px}
tr.cur td{background:var(--hl)} tr.cur td.lbl{background:var(--hl)}
td.chg{color:var(--chg);font-weight:700}
td.cell{cursor:pointer} td.cell:hover{outline:2px solid var(--acc)}
td.sel{outline:2px solid var(--acc)}
.panel{margin-top:16px;background:var(--pan);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.panel h2{margin:0 0 2px;font-size:15px} .stage{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.expr{margin:8px 0;padding:8px 10px;background:var(--bg);border:1px solid var(--line);border-radius:6px;overflow-x:auto}
.note{color:var(--mut);font-style:italic;margin:6px 0}
.derivs{margin-top:8px} .derivs div{padding:2px 6px;border-radius:4px;font-family:ui-monospace,Menlo,monospace;font-size:12.5px}
.derivs div.on{background:var(--hl);font-weight:600}
kbd{background:var(--pan);border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:11px}
</style></head><body>
<h1>Stepping through <span class="mono">_s2_closer.hi_to_hf</span></h1>
<div class="sub"><b>HI:</b> <span class="mono" id="hi"></span> &nbsp;→&nbsp; <b>HF:</b> <span class="mono" id="hf"></span>
&nbsp;·&nbsp; every value verified against the module and <span class="mono">grammar_oracle.py</span></div>
<div class="bar">
  <button id="first">⏮ First</button><button id="prev">◀ Prev</button>
  <span id="counter"></span>
  <button id="next">Next ▶</button><button id="last">Last ⏭</button>
  <input type="range" id="slider" min="0" value="0">
  <span class="sub" style="margin:0">keys <kbd>←</kbd> <kbd>→</kbd></span>
</div>
<div class="wrap"><table id="trace"></table></div>
<div class="panel">
  <div class="stage" id="stage"></div><h2 class="mono" id="name"></h2>
  <div class="expr mono" id="expr"></div><div class="note" id="note"></div>
  <div class="sub" style="margin:0">Per-position derivation — click a cell in the highlighted row to focus one.</div>
  <div class="derivs" id="derivs"></div>
</div>
<script>
const DATA = __DATA__;
const {toks, hi_ids, steps, lcn, rolen} = DATA;
const n = toks.length;
let cur = 0, sel = -1;

function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
function fmtVal(v, fmt){
  if(fmt==="lc")   return lcn[String(v)] ?? String(v);
  if(fmt==="role") return rolen[String(v)] ?? String(v);
  if(fmt==="bool") return String(v);
  return String(v);
}
function prevSame(k){ for(let j=k-1;j>=0;j--) if(steps[j].name===steps[k].name) return j; return -1; }

function render(){
  const s = steps[cur];
  // ---- trace table ----
  let h = "<thead><tr><th class='lbl'>s-op</th>";
  for(let i=0;i<n;i++) h += `<th><div class='pos'>${i}</div>${esc(toks[i])}</th>`;
  h += "</tr></thead><tbody>";
  for(let k=0;k<=cur;k++){
    const st = steps[k];
    const tag = st.round ? ` <span class='pos'>r${st.round}</span>` : "";
    const p = prevSame(k);
    h += `<tr class="${k===cur?'cur':''}"><td class='lbl'>${esc(st.name)}${tag}</td>`;
    for(let i=0;i<n;i++){
      const changed = p>=0 && steps[p].vals[i]!==st.vals[i];
      const cls = [k===cur?"cell":"", changed?"chg":"", (k===cur&&i===sel)?"sel":""].join(" ");
      const oc = k===cur ? `onclick="pick(${i})"` : "";
      h += `<td class="${cls}" ${oc}>${fmtVal(st.vals[i], st.fmt)}</td>`;
    }
    h += "</tr>";
  }
  document.getElementById("trace").innerHTML = h + "</tbody>";
  // ---- panel ----
  document.getElementById("stage").textContent = s.stage + (s.round ? ` · relaxation round ${s.round}` : "");
  document.getElementById("name").textContent = s.name;
  document.getElementById("expr").textContent = s.expr;
  document.getElementById("note").textContent = s.note || "";
  document.getElementById("derivs").innerHTML =
    s.derivs.map((d,i)=>`<div class="${i===sel?'on':''}" onclick="pick(${i})">${esc(d)}</div>`).join("");
  document.getElementById("counter").textContent = `step ${cur+1} / ${steps.length}`;
  document.getElementById("slider").value = cur;
  document.getElementById("prev").disabled = document.getElementById("first").disabled = (cur===0);
  document.getElementById("next").disabled = document.getElementById("last").disabled = (cur===steps.length-1);
  const tb = document.querySelector("tr.cur"); if(tb) tb.scrollIntoView({block:"nearest"});
}
function pick(i){ sel = (sel===i ? -1 : i); render(); }
function go(k){ cur = Math.max(0, Math.min(steps.length-1, k)); sel = -1; render(); }
document.getElementById("next").onclick  = ()=>go(cur+1);
document.getElementById("prev").onclick  = ()=>go(cur-1);
document.getElementById("first").onclick = ()=>go(0);
document.getElementById("last").onclick  = ()=>go(steps.length-1);
document.getElementById("slider").max = steps.length-1;
document.getElementById("slider").oninput = e=>go(+e.target.value);
document.addEventListener("keydown", e=>{
  if(e.key==="ArrowRight") {e.preventDefault(); go(cur+1);}
  if(e.key==="ArrowLeft")  {e.preventDefault(); go(cur-1);}
});
document.getElementById("hi").textContent = toks.join(" ");
document.getElementById("hf").textContent = DATA.hf.join(" ");
render();
</script></body></html>"""


def write_html(path, toks, hi_ids, STEPS, out_words):
    data = dict(toks=toks, hi_ids=hi_ids, steps=STEPS, hf=out_words,
                lcn={str(k): v for k, v in LCN.items()},
                rolen={str(k): v for k, v in ROLEN.items()})
    with open(path, "w") as f:
        payload = json.dumps(data).replace("</", "<\\/")
        f.write(HTML.replace("__DATA__", payload))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sentence", nargs="?", default="the boy that chases the dog swims")
    ap.add_argument("--out", default=os.path.join(HERE, "closer_viz.html"))
    ap.add_argument("--max-rounds", type=int, default=3, help="relaxation rounds to PRINT (html always has all)")
    ap.add_argument("--quiet", action="store_true", help="skip the text derivation")
    a = ap.parse_args()

    toks, hi_ids, STEPS, out_words = build(a.sentence)
    if not a.quiet:
        print_text(toks, hi_ids, STEPS, out_words, a.max_rounds)
    write_html(a.out, toks, hi_ids, STEPS, out_words)
    print(f"\n{len(STEPS)} steps · wrote {a.out}")


if __name__ == "__main__":
    main()

"""Faithful trace of _s2_closer.hi_to_hf: reproduces the pipeline verbatim with
prints of every intermediate, then asserts the final output equals the real
module (and the oracle).  Used to source the numbers in the closer handout.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np

import _s2_closer as C
from _s2_closer import (_lexclass, _is_intrans_lex, _read_off, _stab_ge,
                         _W2I, _I2W, D, NSING, NPROP, ADJ, ADV, THAT, V, OTHER)
from rasp_l import tok_map, kqv, index_select, indices, full, where, equals
from grammar_oracle import oracle_features


def trace(hi_tokens, verbose=True):
    hi_ids = np.array([_W2I.get(w, 0) for w in hi_tokens], dtype=int)
    idx = indices(hi_ids); ZERO = full(hi_ids, 0)

    lc = tok_map(hi_ids, _lexclass)
    is_intr = tok_map(hi_ids, _is_intrans_lex)
    prev_lc = _read_off(lc, -1); prev2_lc = _read_off(lc, -2)
    next_lc = _read_off(lc, +1); next2_lc = _read_off(lc, +2)
    isthat = (lc == THAT); is_crel = isthat & (prev_lc == NSING); is_c = isthat & ~is_crel
    isverb = (lc == V); isadv = (lc == ADV)
    nn_lc = where(next_lc == ADV, next2_lc, next_lc)
    role = where(isverb,
        where(nn_lc == THAT, full(hi_ids, 3),
        where((nn_lc == D) | (nn_lc == NPROP), full(hi_ids, 2),
        where(is_intr == 1, full(hi_ids, 1), full(hi_ids, 4)))), ZERO)

    hb_end = where(isverb & (next_lc == ADV), idx + 1, idx)
    dp_end_flat = where(lc == D, where(next_lc == ADJ, idx + 2, idx + 1), idx)
    rc_that_pos = dp_end_flat + 1
    lc_at_rc = index_select(lc, rc_that_pos, default=-1, causal=False)
    dp_has_rc = (lc == D) & (lc_at_rc == THAT)

    dpfull = dp_end_flat; vpe = hb_end; rce = idx.copy(); cpe = idx.copy()
    snapshots = []
    for r in range(C.ROUNDS):
        rce_at_rc = index_select(rce, rc_that_pos, default=0, causal=False)
        dpfull = where(dp_has_rc, rce_at_rc, dp_end_flat)
        hb1 = hb_end + 1
        dpfull_at_hb1 = index_select(dpfull, hb1, default=0, causal=False)
        cpe_at_hb1 = index_select(cpe, hb1, default=0, causal=False)
        vpe = where((role == 1) | (role == 4), hb_end,
              where(role == 2, dpfull_at_hb1,
              where(role == 3, cpe_at_hb1, idx)))
        q1 = idx + 1
        vpe_at_q1 = index_select(vpe, q1, default=0, causal=False)
        dpfull_at_q1 = index_select(dpfull, q1, default=0, causal=False)
        rce = where(is_crel, where(next_lc == V, vpe_at_q1, dpfull_at_q1 + 1), idx)
        vpos = dpfull_at_q1 + 1
        vpe_at_vpos = index_select(vpe, vpos, default=0, causal=False)
        cpe = where(is_c, vpe_at_vpos, idx)
        e_r = where(is_crel, rce, where(is_c, cpe, idx))
        snapshots.append(e_r.copy())

    e = where(is_crel, rce, where(is_c, cpe, idx))
    end_that = where(isthat, e, full(hi_ids, -1))
    depth = _stab_ge(end_that)

    d_adv = where(isverb & (next_lc == ADV), full(hi_ids, 1), ZERO) \
          + where(isadv, full(hi_ids, -1), ZERO)
    span_clause = e - idx + 1
    d_cp = -depth + where(isthat, span_clause, ZERO)
    e_at_p1 = index_select(e, idx + 1, default=0, causal=False)
    e_at_p2 = index_select(e, idx + 2, default=0, causal=False)
    core_noun_bonus = where((lc == NSING) & (next_lc == THAT), e_at_p1 - (idx + 1) + 1, ZERO)
    core_adj_bonus = where((lc == ADJ) & (next_lc == NSING) & (next2_lc == THAT),
                           e_at_p2 - (idx + 2) + 1, ZERO)
    has_adj_core = (prev2_lc == ADJ)
    crel_end_w1 = where(is_crel & ~has_adj_core, e, full(hi_ids, -1))
    crel_end_w2 = where(is_crel & has_adj_core, e, full(hi_ids, -1))
    d_nprel = core_noun_bonus + core_adj_bonus - (_stab_ge(crel_end_w1) + 2 * _stab_ge(crel_end_w2))

    comp_len = vpe - hb_end
    is_vpflip = (role == 2) | (role == 3)
    prev_role = _read_off(role, -1, default=0); role_prev2 = _read_off(role, -2, default=0)
    comp_len_prev = _read_off(comp_len, -1, default=0)
    d_vp_head = where(is_vpflip, comp_len, ZERO) \
              + where(isadv & ((prev_role == 2) | (prev_role == 3)), comp_len_prev, ZERO)
    vpe_prev = _read_off(vpe, -1, default=0); vpe_prev2 = _read_off(vpe, -2, default=0)
    w1_here = ((prev_role == 2) | (prev_role == 3)) & (lc != ADV)
    w2_here = ((role_prev2 == 2) | (role_prev2 == 3)) & (prev_lc == ADV)
    cs_end_w1 = where(w1_here, vpe_prev, full(hi_ids, -1))
    cs_end_w2 = where(w2_here, vpe_prev2, full(hi_ids, -1))
    d_vp = d_vp_head - (_stab_ge(cs_end_w1) + 2 * _stab_ge(cs_end_w2))

    disp = d_adv + d_cp + d_nprel + d_vp
    hf_pos = idx + disp
    src = kqv(hf_pos, idx, idx, equals, default=0, causal=False)
    out_ids = index_select(hi_ids, src, default=0, causal=False)
    out = [_I2W.get(int(t), "<unk>") for t in out_ids]

    # ---- faithfulness guards ----
    assert out == C.hi_to_hf(hi_tokens), "trace diverged from module!"
    orc = oracle_features(hi_tokens)
    assert out == orc["hf_tokens"], "module disagrees with oracle!"
    assert list(depth) == orc["depth"], "depth disagrees with oracle!"
    assert list(disp) == orc["disp"], "disp disagrees with oracle!"

    if verbose:
        role_name = {0: ".", 1: "intr", 2: "trans", 3: "cp", 4: "objgap"}
        print("=" * 78)
        print("HI:", " ".join(hi_tokens))
        print("HF:", " ".join(out))
        hdr = f"{'i':>2} {'tok':<9} {'role':<6} {'e(that)':>7} {'depth':>5} " \
              f"{'d_adv':>5} {'d_cp':>5} {'d_npr':>5} {'d_vp':>5} {'disp':>5} {'hf':>3}"
        print(hdr)
        for i, t in enumerate(hi_tokens):
            et = int(e[i]) if isthat[i] else -1
            print(f"{i:>2} {t:<9} {role_name[int(role[i])]:<6} {et:>7} {int(depth[i]):>5} "
                  f"{int(d_adv[i]):>+5} {int(d_cp[i]):>+5} {int(d_nprel[i]):>+5} "
                  f"{int(d_vp[i]):>+5} {int(disp[i]):>+5} {int(hf_pos[i]):>3}")
        print("clause-end e(q) per sweep (rows=sweeps 1..K), only 'that' cols matter:")
        for r in range(min(4, len(snapshots))):
            print(f"  sweep {r+1}: {list(map(int, snapshots[r]))}")
    return out


for s in ["John knows that Mary likes the dog",
          "the boy that chases the dog swims",
          "John knows that Mary knows that the dog swims"]:
    trace(s.split())
print("\nALL TRACES MATCH MODULE + ORACLE ✓")

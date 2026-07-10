"""Generate closer_explorer.html — an interactive viewer for _s2_closer with:

  (a) PRIMITIVES tab: every high-level primitive (Map, Where, Read, Gather,
      Stab, Invert) explained from first principles and ANIMATED on real,
      machine-verified arrays from the running sentence.
  (b) QUIZ tab: a "Generate quiz" button that builds fresh randomized tests
      (pure-primitive drills graded by a JS mirror of the primitives, plus
      questions drawn from the embedded verified trace, plus concept MCQs).
  (c) STEPPER tab: step through the pseudocode line by line; the matching
      line highlights while the s-op table grows row by row (all 10 rounds).

All embedded values come from _closer_viz.build(), which asserts equality
with _s2_closer.hi_to_hf and grammar_oracle.py.  Regenerate with:

    uv run python _closer_explorer.py
"""
from __future__ import annotations
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from _closer_viz import build, LCN, ROLEN  # noqa: E402  (neutralizes breakpoint(), verifies vs oracle)

SENTS = [
    ("depth-1 (relative clause)", "the boy that chases the dog swims"),
    ("depth-2 (nested CP_sent)", "John knows that Mary knows that the dog swims"),
]

sent_data = []
for label, s in SENTS:
    toks, ids, steps, hf = build(s)
    sent_data.append(dict(label=label, sent=s, toks=toks, hi_ids=ids, hf=hf, steps=steps))


def last_vals(sd, name, rnd=None):
    out = None
    for st in sd["steps"]:
        if st["name"] == name and (rnd is None or st["round"] == rnd):
            out = st["vals"]
    assert out is not None, f"missing step {name}"
    return out


d0 = sent_data[0]
ANIM = dict(
    toks=d0["toks"],
    ids=last_vals(d0, "hi_ids"),
    lc=last_vals(d0, "lc"),
    next_lc=last_vals(d0, "next_lc"),
    hb1=last_vals(d0, "hb1"),
    dpfull=last_vals(d0, "dpfull"),
    dpfull_at_hb1=last_vals(d0, "dpfull_at_hb1"),
    dp_has_rc=last_vals(d0, "dp_has_rc"),
    rce_at_rc=last_vals(d0, "rce_at_rc"),
    dp_end_flat=last_vals(d0, "dp_end_flat"),
    end_that=last_vals(d0, "end_that"),
    depth=last_vals(d0, "depth"),
    hf_pos=last_vals(d0, "hf_pos"),
    src=last_vals(d0, "src"),
)
# sanity: these exact arrays were verified earlier against the oracle
assert ANIM["dpfull"] == [5, 1, 2, 3, 5, 5, 6]
assert ANIM["depth"] == [0, 0, 1, 1, 1, 1, 0]
assert ANIM["hf_pos"] == [0, 5, 4, 3, 1, 2, 6]
assert ANIM["src"] == [0, 4, 5, 3, 2, 1, 6]
assert ANIM["dpfull_at_hb1"] == [1, 2, 3, 5, 5, 6, 0]
assert ANIM["end_that"] == [-1, -1, 5, -1, -1, -1, -1]

# --------------------------- pseudocode (data) ---------------------------
# One block per stage; each line lists the step-names it corresponds to.
S1 = "Stage 1 — local POS/role"
S2A = "Stage 2 — scaffolding"
S2B = "Stage 2 — relaxation"
S2C = "Stage 2 — depth & displacement"
S3 = "Stage 3 — assemble & gather"

PSEUDO = [
    dict(stage=S1, title="Stage1 — everything from a token and two neighbours", lines=[
        dict(names=["hi_ids"], t="ids ← Map(words, WORD2ID)", c="token embedding"),
        dict(names=["idx"], t="i ← [0, 1, …, n−1]", c="position s-op"),
        dict(names=["lc"], t="lc ← Map(ids, LexClass)", c="coarse POS"),
        dict(names=["is_intr"], t="isIntr ← Map(ids, ∈ V_intrans)", c="lexical flag"),
        dict(names=["prev_lc"], t="prev ← Read(lc, −1)", c=""),
        dict(names=["prev2_lc"], t="prev2 ← Read(lc, −2)", c=""),
        dict(names=["next_lc"], t="next ← Read(lc, +1)", c="forward read: HI block is in the past"),
        dict(names=["next2_lc"], t="next2 ← Read(lc, +2)", c=""),
        dict(names=["isthat"], t="isThat ← [lc = THAT]", c="the clause openers"),
        dict(names=["is_crel"], t="isCrel ← isThat ∧ [prev = N]", c="relative complementizer"),
        dict(names=["is_c"], t="isC ← isThat ∧ ¬isCrel", c="sentential complementizer"),
        dict(names=["isverb"], t="isVerb ← [lc = V]", c=""),
        dict(names=["isadv"], t="isAdv ← [lc = Adv]", c=""),
        dict(names=["nn_lc"], t="nn ← Where([next = Adv], next2, next)", c="next NON-adverb class"),
        dict(names=["role"], t="role ← 3 if nn=THAT · 2 if nn∈{D,Nprop} · 1 if isIntr · 4 else  (0 for non-verbs)",
             c="2 & 3 head a flip VP"),
    ]),
    dict(stage=S2A, title="Scaffold — one-shot local span pieces", lines=[
        dict(names=["hb_end"], t="hbEnd ← Where(isVerb ∧ [next = Adv], i+1, i)", c="verb(+adverb) head block"),
        dict(names=["dp_end_flat"], t="dpFlat ← Where([lc = D], Where([next = Adj], i+2, i+1), i)",
             c="DP end ignoring a rel. clause"),
        dict(names=["rc_that_pos"], t="rcPos ← dpFlat + 1", c="slot where a relative 'that' would sit"),
        dict(names=["lc_at_rc"], t="lc_at_rc ← Gather(lc, rcPos)", c="one lookahead"),
        dict(names=["dp_has_rc"], t="hasRC ← [lc = D] ∧ [lc_at_rc = THAT]", c="DP carries a rel. clause"),
    ]),
    dict(stage=S2B, title="Relax(K) — mutually recursive END pointers", lines=[
        dict(names=["dpfull (seed)"], t="dpfull ← dpFlat        (seed)", c=""),
        dict(names=["vpe (seed)"], t="vpe ← hbEnd            (seed)", c=""),
        dict(names=["rce (seed)"], t="rce ← i                (seed: 'ends at itself')", c=""),
        dict(names=["cpe (seed)"], t="cpe ← i                (seed)", c=""),
        dict(names=[], t="for r = 1 … K:", c="one sweep = one attention hop = one nesting level"),
        dict(names=["rce_at_rc"], t="  rce_at_rc ← Gather(rce, rcPos)", c="rel-clause end, fetched to the DP start"),
        dict(names=["dpfull"], t="  dpfull ← Where(hasRC, rce_at_rc, dpFlat)", c="DP skips its rel. clause"),
        dict(names=["hb1"], t="  hb1 ← hbEnd + 1", c="where a verb's complement starts"),
        dict(names=["dpfull_at_hb1"], t="  dpfull_at_hb1 ← Gather(dpfull, hb1)", c="object DP's end"),
        dict(names=["cpe_at_hb1"], t="  cpe_at_hb1 ← Gather(cpe, hb1)", c="embedded CP_sent's end"),
        dict(names=["vpe"], t="  vpe ← hbEnd if role∈{1,4} · dpfull_at_hb1 if role=2 · cpe_at_hb1 if role=3",
             c="a VP ends where its complement ends"),
        dict(names=["vpe_at_q1"], t="  vpe_at_q1 ← Gather(vpe, i+1)", c=""),
        dict(names=["dpfull_at_q1"], t="  dpfull_at_q1 ← Gather(dpfull, i+1)", c=""),
        dict(names=["rce"], t="  rce ← Where(isCrel, vpe_at_q1 if [next=V] else dpfull_at_q1+1, i)",
             c="subject gap: a VP · object gap: the gap verb"),
        dict(names=["vpos"], t="  vpos ← dpfull_at_q1 + 1", c="embedded VP starts past the subject DP"),
        dict(names=["vpe_at_vpos"], t="  vpe_at_vpos ← Gather(vpe, vpos)", c=""),
        dict(names=["cpe"], t="  cpe ← Where(isC, vpe_at_vpos, i)", c="CP_sent end = its inner VP's end"),
    ]),
    dict(stage=S2C, title="Depth, then displacement (four additive families)", lines=[
        dict(names=["e"], t="e ← Where(isCrel, rce, Where(isC, cpe, i))", c="clause end per opener"),
        dict(names=["end_that"], t="endThat ← Where(isThat, e, −1)", c="mask non-openers"),
        dict(names=["depth"], t="depth ← Stab(endThat)", c="# intervals [q, e(q)] covering i"),
        dict(names=["d_adv"], t="d_adv ← [verb before Adv]·(+1) + [isAdv]·(−1)", c="local verb↔adverb swap"),
        dict(names=["span_clause"], t="span ← e − i + 1", c="clause size (meaningful at openers)"),
        dict(names=["d_cp"], t="d_cp ← −depth + Where(isThat, span, 0)", c="body −1 each; 'that' +|clause|"),
        dict(names=["e_at_p1"], t="e_at_p1 ← Gather(e, i+1)", c=""),
        dict(names=["e_at_p2"], t="e_at_p2 ← Gather(e, i+2)", c=""),
        dict(names=["core_noun_bonus"], t="nounBonus ← Where([lc=N] ∧ [next=THAT], e_at_p1−(i+1)+1, 0)",
             c="noun jumps over its rel. clause"),
        dict(names=["core_adj_bonus"], t="adjBonus ← Where([lc=Adj] ∧ [next=N] ∧ [next2=THAT], e_at_p2−(i+2)+1, 0)", c=""),
        dict(names=["d_nprel_core"], t="d_nprel_core ← nounBonus + adjBonus", c=""),
        dict(names=["has_adj_core"], t="hasAdjCore ← [prev2 = Adj]", c="|core| = 2?"),
        dict(names=["crel_end_w1"], t="relEnd₁ ← Where(isCrel ∧ ¬hasAdjCore, e, −1)", c=""),
        dict(names=["crel_end_w2"], t="relEnd₂ ← Where(isCrel ∧ hasAdjCore, e, −1)", c=""),
        dict(names=["d_nprel_inside"], t="d_nprel_in ← −(Stab(relEnd₁) + 2·Stab(relEnd₂))",
             c="tokens inside a rel. clause lose |core|"),
        dict(names=["d_nprel"], t="d_nprel ← d_nprel_core + d_nprel_in", c=""),
        dict(names=["comp_len"], t="compLen ← vpe − hbEnd", c="a flip verb's complement size"),
        dict(names=["is_vpflip"], t="isFlip ← [role=2] ∨ [role=3]", c=""),
        dict(names=["prev_role"], t="prevRole ← Read(role, −1)", c=""),
        dict(names=["role_prev2"], t="prevRole2 ← Read(role, −2)", c=""),
        dict(names=["comp_len_prev"], t="compLenPrev ← Read(compLen, −1)", c=""),
        dict(names=["d_vp_head"], t="d_vp_head ← isFlip·compLen + [adv of flip verb]·compLenPrev", c="head gets +|comp|"),
        dict(names=["vpe_prev"], t="vpePrev ← Read(vpe, −1)", c=""),
        dict(names=["vpe_prev2"], t="vpePrev2 ← Read(vpe, −2)", c=""),
        dict(names=["w1_here"], t="w1 ← [prevRole ∈ {2,3}] ∧ [lc ≠ Adv]", c="comp start, |head|=1"),
        dict(names=["w2_here"], t="w2 ← [prevRole2 ∈ {2,3}] ∧ [prev = Adv]", c="comp start, |head|=2"),
        dict(names=["cs_end_w1"], t="csEnd₁ ← Where(w1, vpePrev, −1)", c=""),
        dict(names=["cs_end_w2"], t="csEnd₂ ← Where(w2, vpePrev2, −1)", c=""),
        dict(names=["d_vp_comp"], t="d_vp_comp ← −(Stab(csEnd₁) + 2·Stab(csEnd₂))",
             c="comp tokens lose |head| per enclosing flip VP"),
        dict(names=["d_vp"], t="d_vp ← d_vp_head + d_vp_comp", c=""),
    ]),
    dict(stage=S3, title="Assemble and gather", lines=[
        dict(names=["disp"], t="disp ← d_adv + d_cp + d_nprel + d_vp", c="equals the oracle displacement"),
        dict(names=["hf_pos"], t="hfPos ← i + disp", c="target slot per token (a bijection)"),
        dict(names=["src"], t="src ← Invert(hfPos)", c="src[j] = the i with hfPos[i] = j"),
        dict(names=["out_ids"], t="out ← Gather(ids, src);  return Decode(out)", c="one gather"),
    ]),
]

# every recorded step must map to a pseudocode line
cover = {}
for b in PSEUDO:
    cover.setdefault(b["stage"], set())
    for ln in b["lines"]:
        cover[b["stage"]].update(ln["names"])
for sd in sent_data:
    for st in sd["steps"]:
        assert st["name"] in cover[st["stage"]], f"unmapped step: {st['name']} ({st['stage']})"

DATA = dict(
    sents=sent_data, anim=ANIM, pseudo=PSEUDO,
    lcn={str(k): v for k, v in LCN.items()},
    rolen={str(k): v for k, v in ROLEN.items()},
)

# ============================== HTML template ==============================
HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>_s2_closer explorer</title><style>
:root{--bg:#ffffff;--fg:#1c1c1c;--mut:#666;--line:#d9d9d9;--pan:#f6f5f1;--hl:#fff3c4;
 --blue:#1f5fbf;--green:#1b7f3b;--red:#c0392b;--dim:#bbb;--codebg:#f2f1ec}
@media(prefers-color-scheme:dark){:root{--bg:#15171b;--fg:#e8e8e8;--mut:#9aa0a6;--line:#34383f;
 --pan:#1d2025;--hl:#3a3416;--blue:#7aa7ff;--green:#63c581;--red:#ff8073;--dim:#4a4f57;--codebg:#20242a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14.5px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:18px 26px 0}
h1{font-size:21px;margin:0}
.sub{color:var(--mut);font-size:13px;margin:4px 0 10px}
nav{display:flex;gap:6px;border-bottom:1px solid var(--line);padding:0 26px}
nav button{font:inherit;padding:9px 16px;border:1px solid var(--line);border-bottom:none;
 border-radius:8px 8px 0 0;background:var(--pan);color:var(--mut);cursor:pointer}
nav button.on{background:var(--bg);color:var(--fg);font-weight:600;position:relative;top:1px}
main{padding:18px 26px 60px;max-width:1200px}
section.tab{display:none} section.tab.on{display:block}
h2{font-size:17px;margin:26px 0 6px} h3{font-size:15px;margin:18px 0 4px}
.card{background:var(--pan);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}
.note{color:var(--mut);font-style:italic}
code,pre{font-family:ui-monospace,Menlo,monospace;font-size:12.5px}
pre{background:var(--codebg);border:1px solid var(--line);border-radius:8px;padding:9px 12px;overflow-x:auto}
button.ctl{font:inherit;font-size:13px;padding:4px 11px;border:1px solid var(--line);
 background:var(--bg);color:var(--fg);border-radius:6px;cursor:pointer}
button.ctl:hover{border-color:var(--blue)} button.ctl:disabled{opacity:.35;cursor:default}
button.big{font-weight:600;padding:8px 18px;border-color:var(--blue);color:var(--blue)}
/* ---- cell grids / animations ---- */
.anim{position:relative;margin:10px 0;overflow-x:auto;padding-bottom:4px}
.agrid{position:relative;display:inline-block;min-width:100%}
.arow{display:flex;align-items:center;gap:6px;margin:7px 0}
.albl{width:128px;min-width:128px;text-align:right;padding-right:8px;color:var(--mut);
 font-family:ui-monospace,Menlo,monospace;font-size:12px}
.acell{width:46px;height:34px;min-width:46px;display:flex;align-items:center;justify-content:center;
 border:1.5px solid var(--line);border-radius:5px;background:var(--bg);
 font-family:ui-monospace,Menlo,monospace;font-size:13px}
.tokhdr .acell{border:none;background:none;flex-direction:column;height:40px;font-style:italic;font-size:12.5px}
.tokhdr .idx{color:var(--mut);font-size:10px;font-style:normal}
.acell.q{border-color:var(--blue);box-shadow:0 0 0 1.5px var(--blue)}
.acell.s{border-color:var(--green);box-shadow:0 0 0 1.5px var(--green)}
.acell.w{border-color:var(--red);box-shadow:0 0 0 1.5px var(--red)}
.acell.dimk{opacity:.38}
.acell.done{background:var(--pan)}
svg.arrows{position:absolute;left:0;top:0;pointer-events:none;overflow:visible}
.cap{min-height:22px;font-size:13px;color:var(--fg);margin:2px 0 4px;font-family:ui-monospace,Menlo,monospace}
.controls{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:4px 0}
.controls .prog{color:var(--mut);font-size:12px;min-width:80px}
/* ---- stepper ---- */
.split{display:grid;grid-template-columns:minmax(340px,44%) 1fr;gap:16px}
@media(max-width:900px){.split{grid-template-columns:1fr}}
.pseudo{background:var(--pan);border:1px solid var(--line);border-radius:10px;padding:10px 6px;
 max-height:72vh;overflow-y:auto;font-family:ui-monospace,Menlo,monospace;font-size:12.3px}
.pseudo .blk{margin:6px 0 12px}
.pseudo .bt{font-weight:700;padding:2px 10px;color:var(--fg);font-family:inherit}
.pseudo .ln{padding:2px 10px;white-space:pre-wrap;border-left:3px solid transparent;color:var(--fg)}
.pseudo .ln .cm{color:var(--mut)}
.pseudo .ln.on{background:var(--hl);border-left-color:var(--red);font-weight:700}
.pseudo .ln.dimmed{opacity:.55}
.stwrap{max-height:46vh;overflow:auto;border:1px solid var(--line);border-radius:8px}
table.st{border-collapse:collapse;font-family:ui-monospace,Menlo,monospace;font-size:12.5px;width:max-content}
.st th,.st td{border:1px solid var(--line);padding:3px 9px;text-align:center;white-space:nowrap}
.st th.lbl,.st td.lbl{text-align:left;position:sticky;left:0;background:var(--bg);z-index:2}
.st thead th{position:sticky;top:0;background:var(--pan);z-index:3}
.st tr.cur td{background:var(--hl)}
.st td.chg{color:var(--red);font-weight:700}
.st td.cell{cursor:pointer} .st td.sel{outline:2px solid var(--blue)}
.dpanel{background:var(--pan);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin-top:12px}
.dpanel .derivs div{padding:2px 6px;border-radius:4px;font-family:ui-monospace,Menlo,monospace;font-size:12.3px}
.dpanel .derivs div.on{background:var(--hl);font-weight:600}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}
.chip{font-size:12px;padding:3px 10px;border:1px solid var(--line);border-radius:999px;
 background:var(--bg);cursor:pointer;color:var(--mut)} .chip:hover{border-color:var(--blue);color:var(--fg)}
/* ---- quiz ---- */
.q{border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:12px 0;background:var(--pan)}
.q .tag{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut)}
.q .prompt{margin:6px 0}
.q input[type=text]{font:inherit;font-family:ui-monospace,Menlo,monospace;width:110px;padding:4px 8px;
 border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
.q label{display:block;margin:3px 0;cursor:pointer}
.q .verdict{font-weight:700;margin-top:8px} .q .verdict.ok{color:var(--green)} .q .verdict.no{color:var(--red)}
.q .explain{margin-top:6px;padding:8px 10px;background:var(--bg);border:1px solid var(--line);
 border-radius:6px;font-family:ui-monospace,Menlo,monospace;font-size:12.3px;white-space:pre-wrap}
.qtable{display:inline-block;margin:4px 0;border:1px solid var(--line);border-radius:6px;padding:4px 8px;
 font-family:ui-monospace,Menlo,monospace;font-size:12.5px;white-space:pre}
.score{font-size:16px;font-weight:700;margin:10px 0}
kbd{background:var(--pan);border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:11px}
.legend{font-size:12px;color:var(--mut);margin:6px 0}
.legend .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 3px -1px 8px}
</style></head><body>
<header>
  <h1>Exploring <span class="mono">_s2_closer</span> from first principles</h1>
  <div class="sub">Every value on this page was generated by the real implementation and asserted equal to
  <span class="mono">grammar_oracle.py</span>. Sentence:
  <span class="mono" id="hdr-sent"></span></div>
</header>
<nav>
  <button data-tab="prim" class="on">1 · Primitives</button>
  <button data-tab="code">2 · Pseudocode stepper</button>
  <button data-tab="quiz">3 · Quiz me</button>
</nav>
<main>

<!-- ============================ PRIMITIVES ============================ -->
<section class="tab on" id="tab-prim">
<div class="card">
<b>First principles.</b> The program manipulates <b>s-ops</b>: length-<i>n</i> integer arrays, one value
per token position — that is the <i>only</i> data structure (no tree, no stack). A transformer layer can
do exactly two things: compute <i>elementwise</i> at each position (an MLP), or move information
<i>between</i> positions through an attention head. Everything in <span class="mono">_s2_closer</span>
is built from six primitives; the animations below run them on real arrays from the running sentence.
Press ▶ or step through each one.
<div class="legend">Colors: <span class="sw" style="background:var(--blue)"></span>query / where-to-look
<span class="sw" style="background:var(--green)"></span>matched source / value
<span class="sw" style="background:var(--red)"></span>value being written</div>
</div>
<div id="prim-panels"></div>
</section>

<!-- ============================ STEPPER ============================ -->
<section class="tab" id="tab-code">
<div class="card">
Step through the pseudocode line by line. The highlighted line is the one that just executed; its s-op
appears as a new row in the table (<span style="color:var(--red);font-weight:700">red</span> cells changed
since that s-op's previous value — watch the relaxation ripple). Click a table cell (or a derivation line)
to focus one position's derivation. Keys: <kbd>←</kbd> <kbd>→</kbd>.
</div>
<div class="controls">
  sentence:&nbsp;<select id="sentsel" class="ctl" style="font:inherit;padding:4px 8px;border-radius:6px;
   background:var(--bg);color:var(--fg);border:1px solid var(--line)"></select>
  <button class="ctl" id="st-first">⏮ first</button>
  <button class="ctl" id="st-prev">◀ prev</button>
  <span class="prog" id="st-counter"></span>
  <button class="ctl" id="st-next">next ▶</button>
  <button class="ctl" id="st-last">last ⏭</button>
  <input type="range" id="st-slider" min="0" value="0" style="flex:1;min-width:160px">
</div>
<div class="chips" id="st-chips"></div>
<div class="split">
  <div class="pseudo" id="pseudo"></div>
  <div>
    <div class="sub mono" id="st-hihf"></div>
    <div class="stwrap"><table class="st" id="st-table"></table></div>
    <div class="dpanel">
      <div class="tag" id="st-stage" style="font-size:11px;text-transform:uppercase;color:var(--mut)"></div>
      <h3 class="mono" id="st-name" style="margin:2px 0 4px"></h3>
      <pre id="st-expr" style="margin:6px 0"></pre>
      <div class="note" id="st-note"></div>
      <div class="derivs" id="st-derivs" style="margin-top:8px"></div>
    </div>
  </div>
</div>
</section>

<!-- ============================ QUIZ ============================ -->
<section class="tab" id="tab-quiz">
<div class="card">
<b>Test yourself.</b> Each quiz mixes: <b>primitive drills</b> (random arrays — compute Read/Gather/Stab/Invert
by hand), <b>trace questions</b> (real values from the verified run of <span class="mono">_s2_closer</span>),
and <b>concept checks</b>. Answers are graded against a JS mirror of the primitives and the embedded
machine-verified trace; every question shows a full derivation after grading.
</div>
<div class="controls">
  <button class="ctl big" id="qz-new">⟳ Generate new quiz</button>
  <button class="ctl" id="qz-grade">Grade all</button>
  <span class="score" id="qz-score"></span>
</div>
<div id="qz-list"></div>
</section>
</main>

<script>
const DATA = __DATA__;
const LCN = DATA.lcn, ROLEN = DATA.rolen, ANIM = DATA.anim;
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
function fmtVal(v, fmt){
  if(fmt==="lc")   return LCN[String(v)] ?? String(v);
  if(fmt==="role") return ROLEN[String(v)] ?? String(v);
  return String(v);
}
/*PURE-BEGIN*/
// Exact JS mirrors of the rasp_l primitives (integer semantics).
const P = {
  read:   (x,d,def=-1)=> x.map((_,i)=> (i+d>=0 && i+d<x.length) ? x[i+d] : def),
  gather: (x,p,def=0) => p.map(t => (t>=0 && t<x.length) ? x[t] : def),
  stab:   (ends)=> ends.map((_,i)=>{let c=0;for(let j=0;j<=i;j++) if(ends[j]>=i)c++;return c;}),
  invert: (hf)  => hf.map((_,j)=>{for(let i=0;i<hf.length;i++) if(hf[i]===j) return i; return 0;}),
};
function parseIntStrict(s){
  s = String(s).trim().replace(/^\+/, "");
  if(!/^-?\d+$/.test(s)) return null;
  return parseInt(s,10);
}
/*PURE-END*/

/* ============================ tabs ============================ */
document.querySelectorAll("nav button").forEach(b=>{
  b.onclick = ()=>{
    document.querySelectorAll("nav button").forEach(x=>x.classList.remove("on"));
    document.querySelectorAll("section.tab").forEach(x=>x.classList.remove("on"));
    b.classList.add("on");
    document.getElementById("tab-"+b.dataset.tab).classList.add("on");
  };
});
document.getElementById("hdr-sent").textContent =
  DATA.sents[0].sent + "   →   " + DATA.sents[0].hf.join(" ");

/* ============================ animation engine ============================ */
class Anim {
  constructor(host, rows, frames, tokens){
    this.rows = rows; this.frames = frames; this.k = -1; this.timer = null;
    const grid = document.createElement("div"); grid.className = "agrid";
    if(tokens){
      const tr = document.createElement("div"); tr.className = "arow tokhdr";
      tr.innerHTML = `<div class="albl"></div>` + tokens.map((t,i)=>
        `<div class="acell"><span>${esc(t)}</span><span class="idx">${i}</span></div>`).join("");
      grid.appendChild(tr);
    }
    this.cells = [];
    rows.forEach((r,ri)=>{
      const div = document.createElement("div"); div.className = "arow";
      div.innerHTML = `<div class="albl">${esc(r.label)}</div>` +
        r.vals.map(v=>`<div class="acell"></div>`).join("");
      grid.appendChild(div);
      this.cells.push([...div.querySelectorAll(".acell")]);
    });
    this.svg = document.createElementNS("http://www.w3.org/2000/svg","svg");
    this.svg.classList.add("arrows"); grid.appendChild(this.svg);
    const animDiv = document.createElement("div"); animDiv.className = "anim";
    animDiv.appendChild(grid); host.appendChild(animDiv);
    this.grid = grid;
    const cap = document.createElement("div"); cap.className = "cap"; host.appendChild(cap);
    this.capEl = cap;
    const ctl = document.createElement("div"); ctl.className = "controls";
    ctl.innerHTML = `<button class="ctl bprev">◀</button><button class="ctl bplay">▶ play</button>
      <button class="ctl bnext">step ▶</button><button class="ctl breset">reset</button>
      <span class="prog"></span>`;
    host.appendChild(ctl);
    this.prog = ctl.querySelector(".prog");
    ctl.querySelector(".bnext").onclick = ()=>{this.pause(); this.go(this.k+1);};
    ctl.querySelector(".bprev").onclick = ()=>{this.pause(); this.go(this.k-1);};
    ctl.querySelector(".breset").onclick = ()=>{this.pause(); this.go(-1);};
    this.playBtn = ctl.querySelector(".bplay");
    this.playBtn.onclick = ()=>{ this.timer ? this.pause() : this.play(); };
    this.go(-1);
  }
  play(){
    if(this.k >= this.frames.length-1) this.go(-1);
    this.playBtn.textContent = "⏸ pause";
    this.timer = setInterval(()=>{
      if(this.k >= this.frames.length-1){ this.pause(); return; }
      this.go(this.k+1);
    }, 950);
  }
  pause(){ clearInterval(this.timer); this.timer = null; this.playBtn.textContent = "▶ play"; }
  go(k){
    k = Math.max(-1, Math.min(this.frames.length-1, k)); this.k = k;
    // reset visuals
    this.rows.forEach((r,ri)=> r.vals.forEach((v,ci)=>{
      const el = this.cells[ri][ci];
      el.className = "acell";
      el.textContent = r.dynamic ? "?" : fmtVal(v, r.fmt);
    }));
    this.svg.innerHTML = "";
    // replay permanent writes up to k
    for(let f=0; f<=k; f++){
      const fr = this.frames[f];
      (fr.set||[]).forEach(([ri,ci,val])=>{
        this.cells[ri][ci].textContent = val;
        this.cells[ri][ci].classList.add("done");
      });
    }
    if(k >= 0){
      const fr = this.frames[k];
      (fr.hl||[]).forEach(([ri,ci,cls])=> this.cells[ri][ci].classList.add(cls));
      (fr.arrows||[]).forEach(a=> this.drawArrow(...a));
      this.capEl.textContent = fr.cap || "";
    } else {
      this.capEl.textContent = "Press ▶ play, or step through frame by frame.";
    }
    this.prog.textContent = `frame ${k+1} / ${this.frames.length}`;
  }
  drawArrow(r1,c1,r2,c2,color){
    const g = this.grid.getBoundingClientRect();
    const a = this.cells[r1][c1].getBoundingClientRect();
    const b = this.cells[r2][c2].getBoundingClientRect();
    const x1 = a.left - g.left + a.width/2, y1 = a.top - g.top + (r2>=r1 ? a.height : 0);
    const x2 = b.left - g.left + b.width/2, y2 = b.top - g.top + (r2>=r1 ? 0 : b.height);
    this.svg.setAttribute("width", this.grid.scrollWidth);
    this.svg.setAttribute("height", this.grid.scrollHeight);
    const mid = (y1+y2)/2;
    const col = getComputedStyle(document.documentElement).getPropertyValue(color==="g"?"--green":color==="r"?"--red":"--blue");
    const path = document.createElementNS("http://www.w3.org/2000/svg","path");
    path.setAttribute("d", `M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2}`);
    path.setAttribute("fill","none"); path.setAttribute("stroke",col);
    path.setAttribute("stroke-width","2");
    const tip = document.createElementNS("http://www.w3.org/2000/svg","path");
    const dy = y2>y1 ? -7 : 7;
    tip.setAttribute("d", `M ${x2-5} ${y2+dy} L ${x2} ${y2} L ${x2+5} ${y2+dy}`);
    tip.setAttribute("fill","none"); tip.setAttribute("stroke",col); tip.setAttribute("stroke-width","2");
    this.svg.appendChild(path); this.svg.appendChild(tip);
  }
}

/* ---------------- primitive panel definitions ---------------- */
const T = ANIM.toks, N = T.length;
const PRIMS = [
{
 id:"map", title:"Map(x, f)  /  Where(cond, a, b) — elementwise (an MLP)",
 intro:`The cheapest primitive: apply a fixed function independently at every position. No information
moves between positions — in a transformer this is the MLP (or the token embedding itself). Here:
the word-id at each position is looked up in a fixed table to get its coarse part of speech.`,
 code:`lc = tok_map(hi_ids, _lexclass)        # rasp_l.tok_map: out[i] = f(x[i])`,
 build(){
   const rows = [
     {label:"ids", vals:ANIM.ids},
     {label:"lc = Map(ids)", vals:ANIM.lc, fmt:"lc", dynamic:true},
   ];
   const frames = ANIM.ids.map((v,i)=>({
     hl:[[0,i,"q"],[1,i,"w"]], arrows:[[0,i,1,i,"r"]],
     set:[[1,i,fmtVal(ANIM.lc[i],"lc")]],
     cap:`lc[${i}] = LexClass(ids[${i}] = ${v}) = ${fmtVal(ANIM.lc[i],"lc")}   ('${T[i]}')`,
   }));
   return {rows, frames, tokens:T};
 }
},
{
 id:"read", title:"Read(x, δ) — relative-offset neighbour read (one attention head)",
 intro:`out[i] = x[i+δ]. One attention head whose query at position i matches the key at relative
offset δ — never an absolute index, which is what makes it length-generalizing. Out-of-range reads
return a default (−1 here). Below: next = Read(lc, +1), the forward read each token uses to see its
right neighbour's class. (Forward is causally legal because the whole HI block is in the decoder's past.)`,
 code:`next_lc = _read_off(lc, +1)            # out[i] = lc[i+1]; default -1 at the boundary`,
 build(){
   const rows = [
     {label:"lc", vals:ANIM.lc, fmt:"lc"},
     {label:"next = Read(lc,+1)", vals:ANIM.next_lc, fmt:"lc", dynamic:true},
   ];
   const frames = ANIM.lc.map((_,i)=>{
     const t = i+1, ok = t < N;
     return {
       hl: ok ? [[1,i,"w"],[0,t,"s"]] : [[1,i,"w"]],
       arrows: ok ? [[0,t,1,i,"g"]] : [],
       set:[[1,i,fmtVal(ANIM.next_lc[i],"lc")]],
       cap: ok ? `next[${i}] = lc[${i}+1] = lc[${t}] = ${fmtVal(ANIM.next_lc[i],"lc")}`
               : `next[${i}] = lc[${t}] → out of range ⇒ default −1`,
     };
   });
   return {rows, frames, tokens:T};
 }
},
{
 id:"where", title:"Where(cond, a, b) — the elementwise switch every rule is built on",
 intro:`out[i] = a[i] if cond[i] else b[i]. Still elementwise (an MLP), but it is how every update rule
selects between cases. Below is the real dpfull rule: a DP that carries a relative clause takes the
clause's end (rce_at_rc); every other position keeps the flat end. Only position 0 ('the boy that…')
has hasRC = 1.`,
 code:`dpfull = where(dp_has_rc, rce_at_rc, dp_end_flat)`,
 build(){
   const rows = [
     {label:"hasRC (cond)", vals:ANIM.dp_has_rc},
     {label:"rce_at_rc (a)", vals:ANIM.rce_at_rc},
     {label:"dpFlat (b)", vals:ANIM.dp_end_flat},
     {label:"dpfull = Where", vals:ANIM.dpfull, dynamic:true},
   ];
   const frames = ANIM.dp_has_rc.map((c,i)=>({
     hl:[[0,i,"q"],[c?1:2,i,"s"],[3,i,"w"]],
     arrows:[[c?1:2,i,3,i,"g"]],
     set:[[3,i,String(ANIM.dpfull[i])]],
     cap: c ? `dpfull[${i}]: hasRC=1 ⇒ take a = rce_at_rc[${i}] = ${ANIM.rce_at_rc[i]}  (the DP absorbs its relative clause)`
            : `dpfull[${i}]: hasRC=0 ⇒ take b = dpFlat[${i}] = ${ANIM.dp_end_flat[i]}`,
   }));
   return {rows, frames, tokens:T};
 }
},
{
 id:"gather", title:"Gather(x, p) — follow a pointer (one attention head)",
 intro:`out[i] = x[p[i]]: position i holds a pointer p[i]; go to that position, copy what's there, bring
it back to i. This is the A_at_B naming convention — dpfull_at_hb1 means "dpfull, fetched from position
hb1". It is how a verb learns where its object DP ends: the verb knows only where the object STARTS
(hb1 = one past its head block); the gather brings back that DP's end-pointer. Watch position 3
('chases'): p[3]=4, dpfull[4]=5 ⇒ its VP ends at 5. Position 6 points out of range ⇒ default 0.`,
 code:`dpfull_at_hb1 = index_select(dpfull, hb1, default=0)   # out[i] = dpfull[hb1[i]]`,
 build(){
   const rows = [
     {label:"hb1 (pointers p)", vals:ANIM.hb1},
     {label:"dpfull (x)", vals:ANIM.dpfull},
     {label:"out = Gather(x,p)", vals:ANIM.dpfull_at_hb1, dynamic:true},
   ];
   const frames = [];
   ANIM.hb1.forEach((t,i)=>{
     const ok = t>=0 && t<N;
     frames.push({hl:[[0,i,"q"]], cap:`i=${i}: read the pointer — p[${i}] = ${t}${ok?"":" (out of range!)"}`});
     if(ok){
       frames.push({hl:[[0,i,"q"],[1,t,"s"]], arrows:[[0,i,1,t,"b"]],
         cap:`follow it: x[${t}] = ${ANIM.dpfull[t]}`});
       frames.push({hl:[[1,t,"s"],[2,i,"w"]], arrows:[[1,t,2,i,"g"]],
         set:[[2,i,String(ANIM.dpfull_at_hb1[i])]],
         cap:`deliver to i: out[${i}] = x[p[${i}]=${t}] = ${ANIM.dpfull_at_hb1[i]}`});
     } else {
       frames.push({hl:[[0,i,"q"],[2,i,"w"]], set:[[2,i,"0"]],
         cap:`out[${i}]: pointer ${t} is out of range ⇒ default 0`});
     }
   });
   return {rows, frames, tokens:T};
 }
},
{
 id:"stab", title:"Stab(ends) — count the intervals covering me (one counting head)",
 intro:`out[i] = #{ j ≤ i : ends[j] ≥ i } — each position j owning a real end value defines an interval
[j, ends[j]]; position i counts how many such intervals cover it. One attention head: select keys with
ends[j] ≥ i (causally, j ≤ i), then count the selected keys (sel_width). This is exactly how depth is
computed: openers ('that') carry their clause end, everyone else is masked to −1 so the ≥ test always
fails. Below, only position 2 carries an interval, [2,5] — so positions 2–5 count 1.`,
 code:`depth = sel_width(select(end_that, indices, geq, causal=True))`,
 build(){
   const rows = [
     {label:"endThat", vals:ANIM.end_that},
     {label:"depth = Stab", vals:ANIM.depth, dynamic:true},
   ];
   const frames = [];
   for(let i=0;i<N;i++){
     const matched = [];
     frames.push({hl:[[1,i,"w"]], cap:`query i=${i}: scan keys j ≤ ${i}, keep those with endThat[j] ≥ ${i}`});
     for(let j=0;j<=i;j++){
       const hit = ANIM.end_that[j] >= i;
       if(hit) matched.push(j);
       frames.push({
         hl:[[1,i,"w"], ...matched.map(m=>[0,m,"s"]), ...(hit?[]:[[0,j,"dimk"]])],
         cap:`  j=${j}: endThat[${j}] = ${ANIM.end_that[j]} ${hit?`≥ ${i} ✓ (interval [${j},${ANIM.end_that[j]}] covers ${i})`:`< ${i} ✗`}`,
       });
     }
     frames.push({hl:[[1,i,"w"], ...matched.map(m=>[0,m,"s"])],
       set:[[1,i,String(ANIM.depth[i])]],
       cap:`depth[${i}] = ${ANIM.depth[i]}   (matched: ${matched.length? "{"+matched.join(",")+"}" : "∅"})`});
   }
   return {rows, frames, tokens:T};
 }
},
{
 id:"invert", title:"Invert(hfPos) — undo the permutation (one attention head)",
 intro:`Stage 3. Every token knows its TARGET slot (hfPos[i] = i + disp[i]), but to build the output we
need the opposite: for each output slot j, WHICH token wants it? One attention head matches slot number
j (query) against hfPos (keys): src[j] = the i with hfPos[i] = j. Because the displacement construction
guarantees hfPos is a bijection, exactly one key matches — that is the built-in correctness check
(mean-aggregation over one match is just that value).`,
 code:`src = kqv(hf_pos, indices, indices, equals, causal=False)`,
 build(){
   const rows = [
     {label:"hfPos (keys)", vals:ANIM.hf_pos},
     {label:"src = Invert", vals:ANIM.src, dynamic:true},
   ];
   const frames = [];
   for(let j=0;j<N;j++){
     frames.push({hl:[[1,j,"w"]], cap:`slot j=${j} asks: who targets me? scan hfPos for the value ${j}`});
     for(let i=0;i<N;i++){
       const hit = ANIM.hf_pos[i]===j;
       frames.push({
         hl:[[1,j,"w"], [0,i, hit?"s":"dimk"]],
         ...(hit? {arrows:[[0,i,1,j,"g"]], set:[[1,j,String(i)]]} : {}),
         cap: hit? `  hfPos[${i}] = ${j} ✓ ⇒ src[${j}] = ${i}   ('${T[i]}' fills slot ${j})`
                 : `  hfPos[${i}] = ${ANIM.hf_pos[i]} ✗`,
       });
       if(hit) break;
     }
   }
   return {rows, frames, tokens:T};
 }
},
];

const primHost = document.getElementById("prim-panels");
PRIMS.forEach(p=>{
  const sec = document.createElement("div"); sec.className = "card";
  sec.innerHTML = `<h3 style="margin-top:2px">${esc(p.title)}</h3>
    <p style="margin:6px 0">${esc(p.intro)}</p><pre>${esc(p.code)}</pre>`;
  primHost.appendChild(sec);
  const spec = p.build();
  new Anim(sec, spec.rows, spec.frames, spec.tokens);
});

/* ============================ STEPPER ============================ */
let SI = 0;               // sentence index
let cur = 0, sel = -1;    // step index, focused position
const sentsel = document.getElementById("sentsel");
DATA.sents.forEach((s,i)=>{
  const o = document.createElement("option"); o.value = i;
  o.textContent = s.label + " — " + s.sent;
  sentsel.appendChild(o);
});
sentsel.onchange = ()=>{ SI = +sentsel.value; cur = 0; sel = -1; buildPseudo(); renderStep(); };

function steps(){ return DATA.sents[SI].steps; }
function prevSame(k){
  const s = steps();
  for(let j=k-1;j>=0;j--) if(s[j].name===s[k].name) return j;
  return -1;
}
function buildPseudo(){
  const host = document.getElementById("pseudo"); host.innerHTML = "";
  DATA.pseudo.forEach((blk,bi)=>{
    const d = document.createElement("div"); d.className = "blk";
    d.innerHTML = `<div class="bt">${esc(blk.title)}</div>` + blk.lines.map((ln,li)=>
      `<div class="ln" id="pl-${bi}-${li}">${esc(ln.t)}${ln.c?`   <span class="cm">▷ ${esc(ln.c)}</span>`:""}</div>`
    ).join("");
    host.appendChild(d);
  });
}
function lineFor(step){
  for(let bi=0; bi<DATA.pseudo.length; bi++){
    const blk = DATA.pseudo[bi];
    if(blk.stage !== step.stage) continue;
    for(let li=0; li<blk.lines.length; li++)
      if(blk.lines[li].names.includes(step.name)) return [bi,li];
  }
  return null;
}
function renderStep(){
  const S = steps(), st = S[cur], toks = DATA.sents[SI].toks, n = toks.length;
  document.getElementById("st-hihf").textContent =
    "HI: " + DATA.sents[SI].sent + "    →    HF: " + DATA.sents[SI].hf.join(" ");
  // pseudocode highlight
  document.querySelectorAll(".pseudo .ln").forEach(el=>el.classList.remove("on"));
  const lf = lineFor(st);
  if(lf){
    const el = document.getElementById(`pl-${lf[0]}-${lf[1]}`);
    el.classList.add("on"); el.scrollIntoView({block:"nearest"});
  }
  // table
  let h = "<thead><tr><th class='lbl'>s-op</th>" +
    toks.map((t,i)=>`<th><div style="color:var(--mut);font-size:10px">${i}</div>${esc(t)}</th>`).join("") +
    "</tr></thead><tbody>";
  for(let k=0;k<=cur;k++){
    const s = S[k], p = prevSame(k);
    const tag = s.round ? ` <span style="color:var(--mut);font-size:10px">r${s.round}</span>` : "";
    h += `<tr class="${k===cur?'cur':''}"><td class='lbl'>${esc(s.name)}${tag}</td>`;
    for(let i=0;i<n;i++){
      const changed = p>=0 && S[p].vals[i]!==s.vals[i];
      const cls = [k===cur?"cell":"", changed?"chg":"", (k===cur&&i===sel)?"sel":""].join(" ");
      h += `<td class="${cls}" ${k===cur?`onclick="pick(${i})"`:""}>${fmtVal(s.vals[i], s.fmt)}</td>`;
    }
    h += "</tr>";
  }
  document.getElementById("st-table").innerHTML = h + "</tbody>";
  const wrap = document.querySelector(".stwrap");
  wrap.scrollTop = wrap.scrollHeight;
  // detail panel
  document.getElementById("st-stage").textContent = st.stage + (st.round?` · sweep ${st.round}`:"");
  document.getElementById("st-name").textContent = st.name;
  document.getElementById("st-expr").textContent = st.expr;
  document.getElementById("st-note").textContent = st.note || "";
  document.getElementById("st-derivs").innerHTML =
    st.derivs.map((d,i)=>`<div class="${i===sel?'on':''}" onclick="pick(${i})">${esc(d)}</div>`).join("");
  document.getElementById("st-counter").textContent = `step ${cur+1} / ${S.length}`;
  const sl = document.getElementById("st-slider"); sl.max = S.length-1; sl.value = cur;
  // chips: stages + rounds
  const chips = [];
  let seen = new Set();
  S.forEach((s,k)=>{
    if(!seen.has(s.stage)){ seen.add(s.stage); chips.push([s.stage.replace(/Stage \d — /,""), k]); }
  });
  const roundStarts = {};
  S.forEach((s,k)=>{ if(s.round && !(s.round in roundStarts)) roundStarts[s.round] = k; });
  [1,2,3,10].forEach(r=>{ if(r in roundStarts) chips.push([`sweep ${r}`, roundStarts[r]]); });
  chips.push(["fixed point → end", S.length-1]);
  document.getElementById("st-chips").innerHTML =
    chips.map(([t,k])=>`<span class="chip" onclick="goStep(${k})">${esc(t)}</span>`).join("");
}
function pick(i){ sel = (sel===i? -1 : i); renderStep(); }
function goStep(k){ cur = Math.max(0, Math.min(steps().length-1, k)); sel = -1; renderStep(); }
document.getElementById("st-next").onclick  = ()=>goStep(cur+1);
document.getElementById("st-prev").onclick  = ()=>goStep(cur-1);
document.getElementById("st-first").onclick = ()=>goStep(0);
document.getElementById("st-last").onclick  = ()=>goStep(steps().length-1);
document.getElementById("st-slider").oninput = e=>goStep(+e.target.value);
document.addEventListener("keydown", e=>{
  if(!document.getElementById("tab-code").classList.contains("on")) return;
  if(e.key==="ArrowRight"){e.preventDefault(); goStep(cur+1);}
  if(e.key==="ArrowLeft") {e.preventDefault(); goStep(cur-1);}
});
buildPseudo(); renderStep();

/* ============================ QUIZ ============================ */
function rint(n){ return Math.floor(Math.random()*n); }
function choice(a){ return a[rint(a.length)]; }
function shuffle(a){ a=[...a]; for(let i=a.length-1;i>0;i--){const j=rint(i+1);[a[i],a[j]]=[a[j],a[i]];} return a; }
function randArr(n,max=9){ return Array.from({length:n},()=>rint(max+1)); }
function arrHTML(label, a){
  return `<div class="qtable">${esc(label)}: idx ${a.map((_,i)=>String(i).padStart(2)).join(" ")}<br>` +
         `${" ".repeat(label.length)}  val ${a.map(v=>String(v).padStart(2)).join(" ")}</div>`;
}

const CONCEPTS = [
 {q:"Which verb roles head a FLIP VP (reverse with a complement)?",
  ok:"roles 2 (+object DP) and 3 (+clause)",
  wrong:["roles 1 and 4","all four roles","only role 3 (+clause)"],
  ex:"Roles 2 and 3 have an overt complement to swap with. Role 1 (intransitive) and role 4 (object-gap) have no complement, so their VPs never flip."},
 {q:"What is the seed (initial value) of the clause-end pointers rce and cpe?",
  ok:"idx — each clause 'ends at itself'",
  wrong:["0 everywhere","−1 everywhere","the sentence length n−1"],
  ex:"The relaxation seeds every pointer to a safe lower bound: 'I end where I start'. Each sweep then extends resolved ends outward by one nesting level."},
 {q:"What does ONE relaxation sweep accomplish?",
  ok:"it resolves exactly one more level of nesting (one attention hop per pointer)",
  wrong:["it resolves the whole sentence if it is grammatical","it fixes one token per sweep, left to right","it only checks convergence"],
  ex:"Each update reads a pointer at ONE other position (one Gather = one attention hop). So depth-d nesting needs ~d sweeps — the layers-scale-with-depth wall."},
 {q:"With ROUNDS = R sweeps, what CP-nesting depth can the program resolve?",
  ok:"depth ≤ R − 1 (it fails completely one level deeper)",
  wrong:["depth ≤ R + 1","depth ≤ 2R","any depth — sweeps only affect speed"],
  ex:"Verified by sweeping ROUNDS: R rounds solve depth ≤ R−1 and fail completely at depth R. ROUNDS is the model's depth budget."},
 {q:"Why is the forward read next_lc = Read(lc, +1) causally legal in a decoder?",
  ok:"in the 'hi <sep> hf' layout, the whole HI block is already in the past when outputs are produced",
  wrong:["attention is bidirectional in decoders","it is not legal — it is a known bug","because δ=+1 is small enough"],
  ex:"The abstract forward read over the HI block is realized as a backward read from the output region: every HI position precedes the <sep>, so it is in the past of any output query."},
 {q:"In the naming convention, A_at_B[i] equals…",
  ok:"A[B[i]] — the value of A fetched from the position stored in B",
  wrong:["B[A[i]]","A[i] + B[i]","A[i] if B[i] else 0"],
  ex:"A_at_B = Gather(A, B). The suffix names WHERE the value was fetched from; the result is delivered back to position i."},
 {q:"Why is mean-aggregation safe when inverting the permutation (src = kqv(hf_pos, idx, idx, equals))?",
  ok:"hf_pos is a bijection, so exactly one key matches each slot — the mean of one value is that value",
  wrong:["mean is rounded to the nearest integer anyway","several matches are averaged deliberately","the default value fixes any collision"],
  ex:"The displacement construction guarantees a true permutation. If two tokens ever claimed one slot, the mean would produce garbage — a built-in correctness check."},
 {q:"dp_end_flat for the fragment 'the nice boy' is…",
  ok:"[2, 1, 2]",
  wrong:["[1, 1, 2]","[2, 2, 2]","[0, 1, 2]"],
  ex:"Position 0 is a determiner whose next token is an adjective ⇒ i+2 = 2 (skip the adjective to the noun). Positions 1 and 2 are not DP-starts ⇒ harmless idx filler."},
 {q:"In a sentence with NO sentential 'that', which pointer stays at its seed through all sweeps?",
  ok:"cpe (CP_sent end) — its Where condition isC never fires",
  wrong:["vpe","dpfull","rce"],
  ex:"cpe = Where(isC, …, idx). With no sentential complementizer, isC is all zeros, so cpe = idx forever — a useful negative example in the trace."},
 {q:"depth[i] is defined as…",
  ok:"the number of openers q ≤ i whose clause end e(q) ≥ i (intervals covering i)",
  wrong:["the number of 'that' tokens anywhere in the sentence","i minus the position of the nearest 'that'","the tree height at token i"],
  ex:"A stabbing count: each 'that' owns the interval [q, e(q)]; depth[i] counts how many such intervals contain position i. One counting attention head."},
 {q:"What breaks FIRST on a sentence nested deeper than ROUNDS − 1?",
  ok:"the END pointers of the outer clauses stop short (stale), so e(q), depth, and disp are wrong",
  wrong:["the token embedding overflows","the gather in stage 3 crashes","the POS tagging becomes ambiguous"],
  ex:"Each sweep resolves one level. Past the budget, an outer clause still holds a stale end (like cpe[2]=5 after sweep 1 in the depth-2 trace) — displacement built on it is wrong."},
 {q:"Which operations are allowed to move information ACROSS positions?",
  ok:"only attention: Read / Gather / Stab (select, kqv, sel_width, index_select)",
  wrong:["any numpy indexing","Python for-loops over positions","Map and Where"],
  ex:"RASP-L discipline: elementwise ops (Map/Where) are MLPs; every cross-position read must be an attention head. The relaxation loop only re-stacks these layers."},
];

const STEPQ_NAMES = ["lc","next_lc","nn_lc","role","hb_end","dp_end_flat","rc_that_pos","dp_has_rc",
 "dpfull","vpe","rce","cpe","e","depth","d_cp","d_nprel","d_vp","disp","hf_pos","src"];

function qRead(){
  const x = randArr(7), d = choice([-2,-1,1,2]), i = rint(7);
  const out = P.read(x,d,-1);
  return {tag:"primitive drill — Read", numeric:true, answer:out[i],
    prompt:`${arrHTML("x",x)}<br>Compute <code>Read(x, ${d>0?"+"+d:d})[${i}]</code> (default −1 if out of range).`,
    explain:`Read(x, δ)[i] = x[i+δ].  Here i+δ = ${i}+${d} = ${i+d}` +
      ((i+d>=0 && i+d<7) ? `, in range ⇒ x[${i+d}] = ${out[i]}.` : ` — out of range ⇒ default −1.`)};
}
function qGather(){
  const x = randArr(7), p = Array.from({length:7},()=>rint(8)), i = rint(7);
  const out = P.gather(x,p,0);
  return {tag:"primitive drill — Gather", numeric:true, answer:out[i],
    prompt:`${arrHTML("x",x)}<br>${arrHTML("p",p)}<br>Compute <code>Gather(x, p)[${i}]</code> = x[p[${i}]] (default 0 if out of range).`,
    explain:`p[${i}] = ${p[i]}` + ((p[i]<7) ? ` ⇒ x[${p[i]}] = ${out[i]}.` : ` — out of range (n=7) ⇒ default 0.`)};
}
function qStab(){
  const ends = Array.from({length:7},()=> Math.random()<0.55 ? -1 : rint(7));
  if(ends.every(v=>v===-1)) ends[rint(7)] = rint(7);
  const i = 2 + rint(5);
  const out = P.stab(ends);
  const js = []; for(let j=0;j<=i;j++) if(ends[j]>=i) js.push(j);
  return {tag:"primitive drill — Stab", numeric:true, answer:out[i],
    prompt:`${arrHTML("ends",ends)}<br>Compute <code>Stab(ends)[${i}]</code> = #{ j ≤ ${i} : ends[j] ≥ ${i} }.`,
    explain:`Scan j = 0…${i}: matches are {${js.join(", ")||"∅"}} ⇒ count = ${out[i]}.`};
}
function qInvert(){
  const hf = shuffle([0,1,2,3,4,5,6]), j = rint(7);
  const out = P.invert(hf);
  return {tag:"primitive drill — Invert", numeric:true, answer:out[j],
    prompt:`${arrHTML("hfPos",hf)}<br>Compute <code>src[${j}]</code> — the i with hfPos[i] = ${j}.`,
    explain:`Scan hfPos for the value ${j}: hfPos[${out[j]}] = ${j} ⇒ src[${j}] = ${out[j]}.`};
}
function qStep(){
  const sd = choice(DATA.sents);
  const cands = sd.steps.filter(s => STEPQ_NAMES.includes(s.name) && (s.round===null || s.round<=2));
  const st = choice(cands);
  const n = sd.toks.length;
  let poss = [...Array(n).keys()];
  const interesting = poss.filter(i =>
    (st.name.startsWith("d_")||st.name==="disp") ? st.vals[i]!==0 :
    (["dpfull","vpe","rce","cpe","e","hf_pos","src"].includes(st.name)) ? st.vals[i]!==i :
    true);
  const i = choice(interesting.length? interesting : poss);
  const strip = sd.toks.map((t,k)=>`${k}:${t}`).join("  ");
  const rtag = st.round ? ` after sweep ${st.round}` : "";
  return {tag:`trace question — ${sd.label}`, numeric:true, answer:st.vals[i],
    prompt:`Sentence: <span class="mono">${esc(strip)}</span><br>` +
      `Computed as <code>${esc(st.expr)}</code>.<br>` +
      `What is <code>${esc(st.name)}[${i}]</code>${rtag}? (token '${esc(sd.toks[i])}')`,
    explain: st.derivs[i]};
}
function qConcept(){
  const c = choice(CONCEPTS);
  const opts = shuffle([{t:c.ok, ok:true}, ...c.wrong.map(t=>({t, ok:false}))]);
  return {tag:"concept check", numeric:false, options:opts, answer:opts.findIndex(o=>o.ok),
    prompt: esc(c.q), explain: c.ex};
}

let QUIZ = [];
function newQuiz(){
  const picks = shuffle([qRead, qGather, qStab, qInvert, qStep, qStep, qStep, qConcept, qConcept, qConcept]);
  QUIZ = picks.map(f=>f());
  const host = document.getElementById("qz-list"); host.innerHTML = "";
  document.getElementById("qz-score").textContent = "";
  QUIZ.forEach((q,qi)=>{
    const d = document.createElement("div"); d.className = "q"; d.id = "q-"+qi;
    let body;
    if(q.numeric){
      body = `<input type="text" id="qa-${qi}" placeholder="integer">`;
    } else {
      body = q.options.map((o,oi)=>
        `<label><input type="radio" name="qa-${qi}" value="${oi}"> ${esc(o.t)}</label>`).join("");
    }
    d.innerHTML = `<div class="tag">Q${qi+1} · ${esc(q.tag)}</div>
      <div class="prompt">${q.prompt}</div>${body}
      <div class="verdict" id="qv-${qi}"></div><div class="explain" id="qe-${qi}" style="display:none"></div>`;
    host.appendChild(d);
  });
}
function gradeQuiz(){
  let right = 0, answered = 0;
  QUIZ.forEach((q,qi)=>{
    let got = null;
    if(q.numeric){
      got = parseIntStrict(document.getElementById("qa-"+qi).value);
    } else {
      const sel = document.querySelector(`input[name="qa-${qi}"]:checked`);
      got = sel ? parseInt(sel.value,10) : null;
    }
    const v = document.getElementById("qv-"+qi), e = document.getElementById("qe-"+qi);
    if(got===null){ v.textContent = "— no answer"; v.className = "verdict"; }
    else {
      answered++;
      const ok = got === q.answer;
      if(ok) right++;
      v.textContent = ok ? "✓ correct" : `✗ — correct answer: ${q.numeric? q.answer : esc(q.options[q.answer].t)}`;
      v.className = "verdict " + (ok?"ok":"no");
    }
    e.style.display = "block";
    e.textContent = q.explain;
  });
  document.getElementById("qz-score").textContent =
    `score: ${right} / ${QUIZ.length}` + (answered<QUIZ.length? ` (${QUIZ.length-answered} unanswered)`:"");
}
document.getElementById("qz-new").onclick = newQuiz;
document.getElementById("qz-grade").onclick = gradeQuiz;
newQuiz();
</script></body></html>"""

payload = json.dumps(DATA).replace("</", "<\\/")
out_path = os.path.join(HERE, "closer_explorer.html")
with open(out_path, "w") as f:
    f.write(HTML.replace("__DATA__", payload))

total_steps = sum(len(sd["steps"]) for sd in sent_data)
print(f"wrote {out_path}")
print(f"  sentences: {[sd['sent'] for sd in sent_data]}")
print(f"  steps embedded: {total_steps}   html: {os.path.getsize(out_path)//1024} KB")
print("  all values verified against _s2_closer.hi_to_hf and grammar_oracle.py")

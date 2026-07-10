"""Generate closer_trace_handout.tex: a full per-token trace of _s2_closer.hi_to_hf.

Every s-op becomes a table row (columns = token positions); the relaxation loop is
shown as four per-round evolution tables (init + all 10 rounds), with cells that
changed from the row above highlighted.  All values are computed by mirroring
hi_to_hf verbatim and are asserted equal to the module and to grammar_oracle.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import numpy as np

import _s2_closer as C
from _s2_closer import (_lexclass, _is_intrans_lex, _read_off, _stab_ge, _W2I, _I2W,
                        D, NSING, NPROP, ADJ, ADV, THAT, V, OTHER)
from rasp_l import tok_map, index_select, indices, where, full, kqv, equals
from grammar_oracle import oracle_features

SENT = "the boy that chases the dog swims"
toks = SENT.split()
hi = np.array([_W2I[w] for w in toks]); idx = indices(hi); ZERO = full(hi, 0)
n = len(toks)

# ------- mirror hi_to_hf, storing every variable in TR, round snapshots in RS -------
TR = {}
def put(name, arr): TR[name] = [int(x) for x in np.asarray(arr, dtype=int)]

put("hi_ids", hi); put("idx", idx)
lc = tok_map(hi, _lexclass); put("lc", lc)
is_intr = tok_map(hi, _is_intrans_lex); put("is_intr", is_intr)
prev_lc = _read_off(lc, -1); prev2_lc = _read_off(lc, -2)
next_lc = _read_off(lc, +1); next2_lc = _read_off(lc, +2)
for nm, a in [("prev_lc",prev_lc),("prev2_lc",prev2_lc),("next_lc",next_lc),("next2_lc",next2_lc)]:
    put(nm, a)
isthat = (lc == THAT); is_crel = isthat & (prev_lc == NSING); is_c = isthat & ~is_crel
isverb = (lc == V); isadv = (lc == ADV)
for nm, a in [("isthat",isthat),("is_crel",is_crel),("is_c",is_c),("isverb",isverb),("isadv",isadv)]:
    put(nm, a)
nn_lc = where(next_lc == ADV, next2_lc, next_lc); put("nn_lc", nn_lc)
role = where(isverb, where(nn_lc == THAT, full(hi,3),
        where((nn_lc==D)|(nn_lc==NPROP), full(hi,2),
        where(is_intr==1, full(hi,1), full(hi,4)))), ZERO); put("role", role)

hb_end = where(isverb & (next_lc==ADV), idx+1, idx); put("hb_end", hb_end)
dp_end_flat = where(lc==D, where(next_lc==ADJ, idx+2, idx+1), idx); put("dp_end_flat", dp_end_flat)
rc_that_pos = dp_end_flat + 1; put("rc_that_pos", rc_that_pos)
lc_at_rc = index_select(lc, rc_that_pos, default=-1, causal=False); put("lc_at_rc", lc_at_rc)
dp_has_rc = (lc==D) & (lc_at_rc==THAT); put("dp_has_rc", dp_has_rc)

dpfull = dp_end_flat.copy(); vpe = hb_end.copy(); rce = idx.copy(); cpe = idx.copy()
RS = {"dpfull":[], "vpe":[], "rce":[], "cpe":[]}
def snap():
    for k, a in [("dpfull",dpfull),("vpe",vpe),("rce",rce),("cpe",cpe)]:
        RS[k].append([int(x) for x in a])
snap()  # init (row 0)
for _ in range(C.ROUNDS):
    rce_at_rc = index_select(rce, rc_that_pos, default=0, causal=False)
    dpfull = where(dp_has_rc, rce_at_rc, dp_end_flat)
    hb1 = hb_end + 1
    dpfull_at_hb1 = index_select(dpfull, hb1, default=0, causal=False)
    cpe_at_hb1 = index_select(cpe, hb1, default=0, causal=False)
    vpe = where((role==1)|(role==4), hb_end, where(role==2, dpfull_at_hb1, where(role==3, cpe_at_hb1, idx)))
    q1 = idx + 1
    vpe_at_q1 = index_select(vpe, q1, default=0, causal=False)
    dpfull_at_q1 = index_select(dpfull, q1, default=0, causal=False)
    rce = where(is_crel, where(next_lc==V, vpe_at_q1, dpfull_at_q1+1), idx)
    vpos = dpfull_at_q1 + 1
    vpe_at_vpos = index_select(vpe, vpos, default=0, causal=False)
    cpe = where(is_c, vpe_at_vpos, idx)
    snap()

e = where(is_crel, rce, where(is_c, cpe, idx)); put("e", e)
end_that = where(isthat, e, full(hi,-1)); put("end_that", end_that)
depth = _stab_ge(end_that); put("depth", depth)

d_adv = where(isverb & (next_lc==ADV), full(hi,1), ZERO) + where(isadv, full(hi,-1), ZERO); put("d_adv", d_adv)
span_clause = e - idx + 1; put("span_clause", span_clause)
d_cp = -depth + where(isthat, span_clause, ZERO); put("d_cp", d_cp)
e_at_p1 = index_select(e, idx+1, default=0, causal=False); put("e_at_p1", e_at_p1)
e_at_p2 = index_select(e, idx+2, default=0, causal=False); put("e_at_p2", e_at_p2)
core_noun_bonus = where((lc==NSING)&(next_lc==THAT), e_at_p1-(idx+1)+1, ZERO); put("core_noun_bonus", core_noun_bonus)
core_adj_bonus = where((lc==ADJ)&(next_lc==NSING)&(next2_lc==THAT), e_at_p2-(idx+2)+1, ZERO); put("core_adj_bonus", core_adj_bonus)
d_nprel_core = core_noun_bonus + core_adj_bonus; put("d_nprel_core", d_nprel_core)
has_adj_core = (prev2_lc==ADJ); put("has_adj_core", has_adj_core)
crel_end_w1 = where(is_crel & ~has_adj_core, e, full(hi,-1)); put("crel_end_w1", crel_end_w1)
crel_end_w2 = where(is_crel & has_adj_core, e, full(hi,-1)); put("crel_end_w2", crel_end_w2)
d_nprel_inside = -(_stab_ge(crel_end_w1) + 2*_stab_ge(crel_end_w2)); put("d_nprel_inside", d_nprel_inside)
d_nprel = d_nprel_core + d_nprel_inside; put("d_nprel", d_nprel)

comp_len = vpe - hb_end; put("comp_len", comp_len)
is_vpflip = (role==2)|(role==3); put("is_vpflip", is_vpflip)
prev_role = _read_off(role, -1, default=0); put("prev_role", prev_role)
role_prev2 = _read_off(role, -2, default=0); put("role_prev2", role_prev2)
comp_len_prev = _read_off(comp_len, -1, default=0); put("comp_len_prev", comp_len_prev)
d_vp_head = where(is_vpflip, comp_len, ZERO) + where(isadv & ((prev_role==2)|(prev_role==3)), comp_len_prev, ZERO); put("d_vp_head", d_vp_head)
vpe_prev = _read_off(vpe, -1, default=0); put("vpe_prev", vpe_prev)
vpe_prev2 = _read_off(vpe, -2, default=0); put("vpe_prev2", vpe_prev2)
w1_here = ((prev_role==2)|(prev_role==3)) & (lc!=ADV); put("w1_here", w1_here)
w2_here = ((role_prev2==2)|(role_prev2==3)) & (prev_lc==ADV); put("w2_here", w2_here)
cs_end_w1 = where(w1_here, vpe_prev, full(hi,-1)); put("cs_end_w1", cs_end_w1)
cs_end_w2 = where(w2_here, vpe_prev2, full(hi,-1)); put("cs_end_w2", cs_end_w2)
d_vp_comp = -(_stab_ge(cs_end_w1) + 2*_stab_ge(cs_end_w2)); put("d_vp_comp", d_vp_comp)
d_vp = d_vp_head + d_vp_comp; put("d_vp", d_vp)

disp = d_adv + d_cp + d_nprel + d_vp; put("disp", disp)
hf_pos = idx + disp; put("hf_pos", hf_pos)
src = kqv(hf_pos, idx, idx, equals, default=0, causal=False); put("src", src)
out_ids = index_select(hi_ids := hi, src, default=0, causal=False); put("out_ids", out_ids)
out_words = [_I2W.get(int(t), "?") for t in out_ids]

# ------------------------- correctness guards -------------------------
assert out_words == C.hi_to_hf(toks), "trace diverged from module"
orc = oracle_features(toks)
assert out_words == orc["hf_tokens"] and TR["depth"] == orc["depth"] and TR["disp"] == orc["disp"]

# =========================== LaTeX emission ===========================
LCNAME = {0:"D",1:"N",2:"Nprop",3:"Adj",4:"Adv",5:"THAT",6:"V",7:"OTHER",-1:"--"}
def esc(s): return s.replace("_", r"\_").replace("&", r"\&").replace("#", r"\#")
def tt(s): return r"\texttt{" + esc(s) + "}"
def cell(x):  # generic int cell
    return "--" if x == -1 else str(x)

hdr = " & ".join([r"\textbf{s-op}"] + [f"{esc(t)}$_{{{i}}}$" for i, t in enumerate(toks)]) + r" \\"

def valrow(name, fmt=cell):
    vals = " & ".join(fmt(x) for x in TR[name])
    return f"{tt(name)} & {vals} " + r"\\"

def lcfmt(x): return LCNAME.get(x, str(x))

L = []  # document body lines
def w(s): L.append(s)

# --- a section rendered as a longtable of (desc, valrow) pairs ---
def section(title, intro, entries):
    if title: w(r"\section{" + title + "}")
    if intro: w(intro)
    w(r"\begin{center}\footnotesize")
    w(r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.15}")
    w(r"\begin{longtable}{@{\extracolsep{\fill}}l|" + "c"*n + r"@{}}")
    w(r"\toprule " + hdr + r" \midrule \endfirsthead")
    w(r"\toprule " + hdr + r" \midrule \endhead \bottomrule \endfoot")
    for name, desc, fmt in entries:
        w(r"\multicolumn{" + str(n+1) + r"}{@{}p{0.98\textwidth}@{}}{\itshape " + desc + r"} \\[1pt]")
        w(valrow(name, fmt))
        w(r"\addlinespace[3pt]")
    w(r"\end{longtable}\end{center}")

# --- an evolution table for one relaxation pointer, all rounds, changes highlighted ---
def evotable(name, desc):
    w(r"\paragraph{" + tt(name) + r".} " + desc)
    w(r"\begin{center}\footnotesize\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.1}")
    w(r"\begin{tabular}{@{}l|" + "c"*n + r"@{}}")
    w(r"\toprule \textbf{round} & " + " & ".join(f"{esc(t)}$_{{{i}}}$" for i,t in enumerate(toks)) + r" \\ \midrule")
    rows = RS[name]
    for r, row in enumerate(rows):
        lbl = "init" if r == 0 else f"R{r}"
        cells = []
        for j, x in enumerate(row):
            s = cell(x)
            if r > 0 and rows[r-1][j] != x:
                s = r"\textcolor{chg}{\textbf{" + s + "}}"
            cells.append(s)
        w(lbl + " & " + " & ".join(cells) + r" \\")
    w(r"\bottomrule\end{tabular}\end{center}")

# ============================ build body ============================
w(r"\section{Stage 1 --- encoding and part of speech}")
w(r"Each column is a token \emph{position} (note \emph{the} appears at both 0 and 4). "
  r"Every row is one s-op: a length-7 array holding one value per position. "
  r"Lexical-class codes: " + ", ".join(f"{v}={k}" for k,v in
    [("D",0),("N\\_sing",1),("N\\_prop",2),("Adj",3),("Adv",4),("THAT",5),("V",6),("OTHER",7)]) + r". "
  r"Verb-role codes: 0=non-verb, 1=intransitive, 2=+object DP, 3=+clause, 4=object-gap.")
section("", "", [
 ("hi_ids", "The fixed word$\\to$id lookup $\\texttt{\\_W2I}$ (a token embedding). Each surface word becomes its unique integer id; repeated words (\\emph{the}) get the \\emph{same} id.", cell),
 ("idx", "The position s-op $[0,1,\\dots,n{-}1]$ from $\\texttt{indices}$. The whole program's output is $\\texttt{idx}+\\text{disp}$, so this is the identity we perturb.", cell),
 ("lc", "A second embedding $\\texttt{tok\\_map(hi\\_ids,\\,\\_lexclass)}$: id$\\to$coarse POS class. Shown by name; the underlying values are the integer codes above.", lcfmt),
 ("is_intr", "Pure lexical flag: 1 iff the word is in $\\texttt{V\\_intrans}$. Only \\emph{swims} qualifies. Used later to tell an intransitive verb from an object-gap verb.", cell),
 ("prev_lc", "Backward neighbour read $\\texttt{\\_read\\_off(lc,-1)}=lc[i{-}1]$ (one attention head keyed on relative offset). $-1$ = fell off the left edge.", lcfmt),
 ("prev2_lc", "$lc[i{-}2]$. Feeds the `adjective-core' relative-clause test far below ($\\texttt{has\\_adj\\_core}$).", lcfmt),
 ("next_lc", "Forward read $lc[i{+}1]$. This is the read that relies on the `HI block is in the past' convention (it looks at a later position, legal only because the whole HI block precedes the output region).", lcfmt),
 ("next2_lc", "$lc[i{+}2]$, used to skip one optional post-verb adverb.", lcfmt),
 ("isthat", "$(lc=\\text{THAT})$. Marks the clause \\textbf{openers} --- the only visible bracket-starts in the string. Fires only at position 2.", cell),
 ("is_crel", "$\\texttt{isthat}\\ \\&\\ (prev\\_lc=\\text{N})$: a \\emph{that} right after a common noun is a \\textbf{relative} complementizer. Here \\emph{boy}(1) precedes \\emph{that}(2), so this fires.", cell),
 ("is_c", "$\\texttt{isthat}\\ \\&\\ \\lnot\\texttt{is\\_crel}$: the \\emph{sentential} \\emph{that}. Empty here --- our only \\emph{that} is relative.", cell),
 ("isverb", "$(lc=\\text{V})$. True at \\emph{chases}(3) and \\emph{swims}(6).", cell),
 ("isadv", "$(lc=\\text{Adv})$. Empty here (no adverb).", cell),
 ("nn_lc", "`next non-adverb class': $\\texttt{where(next\\_lc=Adv,\\,next2\\_lc,\\,next\\_lc)}$. Lets a verb see its complement's first token even across an adverb. With no adverb it equals $\\texttt{next\\_lc}$.", lcfmt),
 ("role", "The verb role, from $\\texttt{nn\\_lc}$: \\emph{chases}(3) sees \\emph{the}=D $\\Rightarrow$ 2 (transitive); \\emph{swims}(6) is lexically intransitive $\\Rightarrow$ 1. Roles 2/3 head a flip VP; 1/4 do not.", cell),
])

section("Stage 2 scaffolding --- one-shot local span pieces",
 "These are computed once, \\emph{before} the loop; each is a fixed-offset lookahead, no iteration.", [
 ("hb_end", "Verb head-block end: $i{+}1$ for a verb\\,+\\,adverb, else $i$. No adverbs here, so it equals $\\texttt{idx}$. This is the `head' side of a VP flip.", cell),
 ("dp_end_flat", "Where a DP ends \\emph{ignoring} relative clauses: $D\\,(Adj)\\,N$. \\emph{the}(0)$\\to$\\emph{boy}(1); \\emph{the}(4)$\\to$\\emph{dog}(5). Note it does \\emph{not} yet know about the relative clause on \\emph{boy}.", cell),
 ("rc_that_pos", "$\\texttt{dp\\_end\\_flat}+1$: the slot just past a DP's noun, where a relative \\emph{that} would sit. For \\emph{the}(0) this is 2.", cell),
 ("lc_at_rc", "$\\texttt{index\\_select(lc,\\,rc\\_that\\_pos)}$: the class actually sitting at that slot (one cross-position lookahead). At position 0 it reads $lc[2]=\\text{THAT}$.", lcfmt),
 ("dp_has_rc", "$(lc=D)\\ \\&\\ (\\texttt{lc\\_at\\_rc}=\\text{THAT})$: this DP carries a relative clause. True at \\emph{the}(0) --- `the boy \\textbf{that}\\,\\dots'.", cell),
])

w(r"\section{Stage 2 relaxation --- the loop, all " + str(C.ROUNDS) + r" rounds}")
w(r"Four END pointers are seeded to a lower bound (`each ends at itself') and refined by "
  + str(C.ROUNDS) + r" sweeps. A \textcolor{chg}{\textbf{red}} cell changed from the row above; "
  r"once a table stops turning red it has reached its fixed point. Only the openers'/verbs' "
  r"columns carry meaning --- other columns hold the harmless $\texttt{idx}$ filler.")
evotable("dpfull", "DP end including a trailing relative clause. Watch column \\emph{the}$_0$: at $\\texttt{init}$ it is the flat 1; it cannot finish until the relative clause's end is known, so it lags one sweep behind $\\texttt{rce}$ and only settles at \\textbf{R2}.")
evotable("vpe", "VP end (verb\\,+\\,complement). \\emph{chases}$_3$ jumps to its object DP's end (5) in R1; \\emph{swims}$_6$ is intransitive so it never moves off 6.")
evotable("rce", "$\\texttt{CP\\_rel}$ end. \\emph{that}$_2$ is a subject-gap relative, so its end = the VP right after it ($\\texttt{vpe}$ at position 3), reaching 5 in R1.")
evotable("cpe", "$\\texttt{CP\\_sent}$ end. Never active here (no sentential \\emph{that}), so it stays at the $\\texttt{idx}$ seed every round --- a useful negative example.")
w(r"After R2 nothing changes: the depth-1 sentence has reached the fixed point with rounds 3--"
  + str(C.ROUNDS) + r" merely confirming it. A depth-$d$ sentence would keep turning red through about round $d{+}1$.")

section("Stage 2 output --- clause ends and depth",
 "Collapse the pointers into one clause-end per \\emph{that}, then count.", [
 ("e", "Clause end $e(q)$: $\\texttt{rce}$ for a relative \\emph{that}, $\\texttt{cpe}$ for a sentential one, else $\\texttt{idx}$. Here $e(2)=5$: `that chases the dog' closes on \\emph{dog}(5).", cell),
 ("end_that", "$\\texttt{where(isthat,\\,e,\\,-1)}$: mask non-openers to $-1$ so they carry no interval into the count.", cell),
 ("depth", "The stabbing count $\\#\\{\\text{\\emph{that}}\\ q\\le i:\\ e(q)\\ge i\\}$ (one attention head). Positions 2--5 are inside the one relative clause $\\Rightarrow$ depth 1; the rest 0.", cell),
])

section("Stage 2 output --- displacement, four flip-typed pieces",
 "Each piece is a (weighted) stabbing sum over enclosing flips of one type, plus a local head bonus. They add to $\\texttt{disp}$.", [
 ("d_adv", "Adverb flip (verb\\,$+1$, adverb\\,$-1$). All zero --- no adverb.", cell),
 ("span_clause", "$e-\\texttt{idx}+1$: the size of the clause an opener spans (only meaningful at \\emph{that}$_2$: $5-2+1=4$).", cell),
 ("d_cp", "$-\\texttt{depth}+(\\text{\\emph{that}}?\\ \\texttt{span\\_clause}:0)$. Every token in a clause body gets $-1$ per enclosing \\emph{that}; the \\emph{that} itself gets $+|\\text{clause}|$. (Here no \\emph{sentential} that, so the clause body $-1$'s come via the relative-clause piece below instead --- $\\texttt{d\\_cp}$ is 0 because $\\texttt{is\\_c}$ is empty and depth's sign is carried by $\\texttt{d\\_nprel}$.)", cell),
 ("e_at_p1", "$e[i{+}1]$: the clause end of a \\emph{that} sitting at $i{+}1$. Read by a noun to size the relative clause it is about to be modified by.", cell),
 ("core_noun_bonus", "A common noun immediately before a relative \\emph{that} gets $+|\\text{CP\\_rel}|$. \\emph{boy}(1): $e[2]-2+1=4$. This is the noun jumping over its relative clause.", cell),
 ("core_adj_bonus", "Same bonus for the adjective of an $Adj\\,N$ core carrying a relative clause. Zero here (no adjective).", cell),
 ("d_nprel_core", "$\\texttt{core\\_noun\\_bonus}+\\texttt{core\\_adj\\_bonus}$: the $+|\\text{RC}|$ for the modified noun. \\emph{boy}(1)$=+4$.", cell),
 ("has_adj_core", "$(prev2\\_lc=Adj)$: does this relative clause's core noun carry an adjective (so $|\\text{core}|=2$)? No.", cell),
 ("crel_end_w1", "Relative-clause ends with weight-1 core (mask others to $-1$), fed to a stabbing head.", cell),
 ("d_nprel_inside", "$-(\\text{stab}(w1)+2\\cdot\\text{stab}(w2))$: every token \\emph{inside} a relative clause loses $|\\text{core}|$. Positions 2--5 each $-1$.", cell),
 ("d_nprel", "$\\texttt{d\\_nprel\\_core}+\\texttt{d\\_nprel\\_inside}$: the full NP-relative contribution. \\emph{boy}$+4$; the clause body $-1$ each.", cell),
 ("comp_len", "$\\texttt{vpe}-\\texttt{hb\\_end}$: a VP-flip verb's complement size. \\emph{chases}(3): $5-3=2$ (`the dog').", cell),
 ("is_vpflip", "$(\\text{role}=2)\\lor(\\text{role}=3)$: verbs that actually reverse with a complement. \\emph{chases}(3) only.", cell),
 ("d_vp_head", "VP-flip head (verb, and its adverb) gets $+|\\text{comp}|$. \\emph{chases}(3)$=+2$.", cell),
 ("w1_here", "Marks a complement-start: the token whose previous token is a no-adverb VP-flip verb. \\emph{the}(4), right after \\emph{chases}.", cell),
 ("cs_end_w1", "That complement's end (from $\\texttt{vpe}[i{-}1]$), fed to a stabbing head.", cell),
 ("d_vp_comp", "$-(\\text{stab}(w1)+2\\cdot\\text{stab}(w2))$: complement tokens lose $|\\text{head}|$ per enclosing VP flip. \\emph{the}(4),\\emph{dog}(5) each $-1$.", cell),
 ("d_vp", "$\\texttt{d\\_vp\\_head}+\\texttt{d\\_vp\\_comp}$: full transitive/clausal-VP contribution.", cell),
])

section("Stage 3 --- assemble and gather",
 "Sum the pieces to a target slot per token, invert the permutation, gather.", [
 ("disp", "$\\texttt{d\\_adv}+\\texttt{d\\_cp}+\\texttt{d\\_nprel}+\\texttt{d\\_vp}$. This equals the oracle's displacement exactly.", cell),
 ("hf_pos", "$\\texttt{idx}+\\texttt{disp}$: the HF slot each HI token targets. It is a genuine permutation of $0\\dots6$.", cell),
 ("src", "$\\texttt{kqv(hf\\_pos,\\,idx,\\,idx,\\,equals)}$: the \\emph{inverse} --- for slot $j$, the source index $i$ with $\\texttt{hf\\_pos}[i]=j$.", cell),
 ("out_ids", "$\\texttt{index\\_select(hi\\_ids,\\,src)}$: gather the token id sitting at each source index into its slot.", cell),
])
w(r"Decoding $\texttt{out\_ids}$ back through $\texttt{\_I2W}$ gives the head-final string:")
w(r"\begin{center}\ttfamily HF = " + " ".join(out_words) + r"\end{center}")

# ---------------------- wrap with preamble, write ----------------------
body = "\n".join(L)
doc = r"""\documentclass[10pt]{article}
\usepackage[margin=1.6cm]{geometry}
\usepackage{amsmath,amssymb,xcolor,booktabs,longtable,array,parskip,microtype}
\definecolor{chg}{HTML}{C0392B}
\setcounter{secnumdepth}{1}
\begin{document}
\begin{center}{\LARGE\bfseries A Full Per-Token Trace of \texttt{\_s2\_closer}}\\[3pt]
{\large every s-op as a row, every token as a column ---}\\[1pt]
{\large\ttfamily HI = """ + SENT + r"""}\\[3pt]
{\small Generated by \texttt{\_gen\_closer\_handout.py}; all values asserted equal to the module and \texttt{grammar\_oracle.py}.}\end{center}
\hrule\vspace{4pt}
""" + body + "\n\\end{document}\n"

out_path = os.path.join(HERE, "closer_trace_handout.tex")
with open(out_path, "w") as f:
    f.write(doc)
print("wrote", out_path)
print("HF =", " ".join(out_words))
print("all asserts passed (module == oracle == trace)")

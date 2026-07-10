"""Ground-truth oracle for the v2 head-initial -> head-final map.

This is NOT the RASP-L program. It is an ordinary recursive-descent parser used
only to (a) establish that the flat HI string determines HF, and (b) produce
per-token ground-truth features (POS, CP-depth, displacement, hf_pos) that the
RASP-L program (hi_hf_rasp.py) is verified against, token by token.

Key facts it encodes (all read off src/grammar/v2/generate_with_frames.py):
  * HF is a permutation of HI. A node is a *flip* node (children reverse HI->HF)
    iff it is one of {CP_sent, CP_rel, VP with a complement, VP_x_adv, NP_singular
    -> NPsRel}. Everything else (S, DP, S_obj_gap, NP_singular_adj, unary) keeps
    child order.
  * Nested reversals compose additively:
        hf_pos(x) = hi_pos(x) + sum over flip-ancestors F of
                    ( +|comp(F)| if x in head-child(F) else -|head(F)| )
    (verified 100% on depth<=2; see LOG.md).
"""
from __future__ import annotations
from grammar.v2.cfg_vocab import VOCAB
from grammar.v2.generate_with_frames import Node

D = set(VOCAB["D"]); NSING = set(VOCAB["N_singular"]); NPROP = set(VOCAB["N_proper"])
ADJ = set(VOCAB["Adj"]); ADV = set(VOCAB["Adv"])
V_INTRANS = set(VOCAB["V_intrans"]); V_DP = set(VOCAB["V_dp"]); V_CP = set(VOCAB["V_cp"])
VERB = V_INTRANS | V_DP | V_CP
THAT = "that"

def _dp_start(t):
    return t in D or t in NPROP

class Parser:
    """One-token-lookahead recursive descent for the head-initial grammar."""

    def __init__(self, toks):
        self.t = toks
        self.i = 0

    def _peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def _eat(self):
        x = self.t[self.i]; self.i += 1; return x

    def parse(self):
        tree = self.S()
        assert self.i == len(self.t), f"unconsumed tail at {self.i}/{len(self.t)}"
        return tree

    def S(self):
        return Node("S", [self.DP(), self.VP()])

    def DP(self):
        if self._peek() in NPROP:
            return Node("DP", [Node("N_proper", word=self._eat())])
        d = self._eat(); assert d in D
        return Node("DP", [Node("D", word=d), self.NPs()])

    def NPs(self):
        if self._peek() in ADJ:
            a = self._eat(); n = self._eat(); assert n in NSING
            core = Node("NP_singular", [Node("NP_singular_adj",
                        [Node("AdjP", [Node("Adj", word=a)]), Node("N_singular", word=n)])])
        else:
            n = self._eat(); assert n in NSING
            core = Node("NP_singular", [Node("N_singular", word=n)])
        while self._peek() == THAT:                        # relative clause on this noun
            core = Node("NP_singular", [core, self.CPrel()])
        return core

    def CPrel(self):
        self._eat()                                        # C_rel "that"
        c = Node("C_rel", word=THAT)
        if self._peek() in VERB:                           # S_subj_gap -> VP
            gap = Node("S_subj_gap", [self.VP()])
        else:                                              # S_obj_gap -> DP VP_obj_gap
            dp = self.DP(); v = self._eat(); assert v in V_DP
            gap = Node("S_obj_gap", [dp, Node("VP_obj_gap", [Node("V_dp", word=v)])])
        return Node("CP_rel", [c, gap])

    def VP(self):
        v = self._eat(); assert v in VERB
        adv = self._eat() if self._peek() in ADV else None
        nxt = self._peek()
        if nxt == THAT:                                    # V_cp CP_sent
            comp = self.CPsent(); pos, kind = "V_cp", "cp"
        elif _dp_start(nxt):                               # V_dp DP
            comp = self.DP(); pos, kind = "V_dp", "dp"
        else:                                              # V_intrans / V_dp(obj-gap)
            comp = None; pos, kind = "V_intrans", "in"
        vn = Node(pos, word=v)
        an = Node("AdvP", [Node("Adv", word=adv)]) if adv else None
        if kind == "in":
            return Node("VP", [Node("VP_intrans_adv", [vn, an])]) if an else Node("VP", [vn])
        lbl = "VP_dp_adv" if kind == "dp" else "VP_cp_adv"
        head = Node(lbl, [vn, an]) if an else vn
        return Node("VP", [head, comp])

    def CPsent(self):
        self._eat()                                        # C "that"
        return Node("CP_sent", [Node("C", word=THAT), self.S()])

# --- tree utilities ---------------------------------------------------------

_FLIP2 = {"CP_sent", "CP_rel", "VP_intrans_adv", "VP_dp_adv", "VP_cp_adv"}
_CP = {"CP_sent", "CP_rel"}

def _is_flip(n):
    return (n.label in _FLIP2) or (n.label in ("VP", "NP_singular")
                                   and n.children and len(n.children) == 2)

def _leaves(n):
    if n.word is not None and n.children is None:
        return [n]
    out = []
    for c in n.children:
        out += _leaves(c)
    return out

def oracle_features(hi_tokens):
    """Return per-token ground truth: pos, depth, disp, hf_pos, plus hf_tokens.

    pos    : fine POS label (leaf label from the parse)
    depth  : number of enclosing CP nodes (CP_sent or CP_rel)
    disp   : hf_pos - hi_pos  (the nested-reversal displacement)
    hf_pos : target slot of this HI token in the HF string
    """
    tree = Parser(hi_tokens).parse()
    lv = _leaves(tree)
    for i, l in enumerate(lv):
        l.idx = i; l.pos = l.label; l.disp = 0; l.depth = 0

    def walk(n, depth):
        nd = depth + 1 if n.label in _CP else depth
        if n.word is not None and n.children is None:
            n.depth = nd
            return
        if _is_flip(n) and len(n.children) == 2:
            head, comp = n.children
            hL, cL = _leaves(head), _leaves(comp)
            for l in hL:
                l.disp += len(cL)
            for l in cL:
                l.disp -= len(hL)
        for c in n.children:
            walk(c, nd)

    walk(tree, 0)
    n = len(lv)
    hf_pos = [lv[i].idx + lv[i].disp for i in range(n)]
    hf_tokens = [None] * n
    for i in range(n):
        hf_tokens[hf_pos[i]] = hi_tokens[i]
    return {
        "pos":    [lv[i].pos for i in range(n)],
        "depth":  [lv[i].depth for i in range(n)],
        "disp":   [lv[i].disp for i in range(n)],
        "hf_pos": hf_pos,
        "hf_tokens": hf_tokens,
    }

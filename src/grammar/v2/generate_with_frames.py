"""Frame-based generation for Grammar v2.

Adds to v1 (grammar/generate_with_frames.py):
  - Relative clauses: CP_rel with subject-gap (S_subj_gap) and object-gap (S_obj_gap).
  - Adjunction: AdjP (left-adjoin to NP_singular), AdvP (right-adjoin in HI, left in HF).
    Adjunction is depth-1 only (no stacked modifiers).
  - Updated terminal taxonomy: N_singular (was NP_singular), N_proper (was NP_proper).
  - New `depth()` counts ANY CP (CP_sent or CP_rel) nesting depth.
  - New `tree_height()` utility: max leaf depth in the full constituency tree.
  - Position-tagged frame strings in FrameSentencePair.
  - `is_ambiguous` flag (cross-frame, set False here; computed in run.py).

CFG rules (HI / HF):
  S -> DP VP                                      / same
  DP -> D NP_singular                             / same
  DP -> NP_proper                                 / same (bare [DP word])
  NP_proper -> N_proper                           / same
  NP_singular -> N_singular                       / same
  NP_singular -> NP_singular_adj                  / same
  NP_singular_adj -> AdjP N_singular              / same
  NP_singular -> NP_singular CP_rel               / CP_rel NP_singular
  CP_rel -> C_rel S_subj_gap                      / S_subj_gap C_rel
  CP_rel -> C_rel S_obj_gap                       / S_obj_gap C_rel
  S_subj_gap -> VP                                / same
  S_obj_gap -> DP VP_obj_gap                      / same
  VP_obj_gap -> V_dp                              / same
  VP -> V_intrans                                 / same
  VP -> V_dp DP                                   / DP V_dp
  VP -> V_cp CP_sent                              / CP_sent V_cp
  VP -> VP_intrans_adv                            / same
  VP_intrans_adv -> V_intrans AdvP                / AdvP V_intrans
  VP -> VP_dp_adv DP                              / DP VP_dp_adv
  VP_dp_adv -> V_dp AdvP                          / AdvP V_dp
  VP -> VP_cp_adv CP_sent                         / CP_sent VP_cp_adv
  VP_cp_adv -> V_cp AdvP                          / AdvP V_cp
  CP_sent -> C S                                  / S C
  AdjP -> Adj                                     / same
  AdvP -> Adv                                     / same

To install: copy this file to src/grammar/v2/generate_with_frames.py

Usage:
  uv run python -m grammar.v2.generate_with_frames --max-depth 3
"""

from __future__ import annotations

import argparse
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Iterator, Union

from grammar.v2.cfg_vocab import VOCAB


# ---------------------------------------------------------------------------
# Generic tree node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    label: str
    children: "list[Node] | None" = None
    word: "str | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.word is not None


def bracketed(tree: Node) -> str:
    if tree.is_leaf:
        return f"[{tree.label} {tree.word}]"
    inner = " ".join(bracketed(c) for c in tree.children or [])
    return f"[{tree.label} {inner}]"


def linearize(tree: Node) -> list[str]:
    if tree.is_leaf:
        return [tree.word]  # type: ignore[list-item]
    words: list[str] = []
    for child in tree.children or []:
        words.extend(linearize(child))
    return words


def compute_tree_height(node: Node) -> int:
    """Max depth of any leaf, with the root counted as depth 1."""
    if node.is_leaf:
        return 1
    return 1 + max(compute_tree_height(c) for c in (node.children or []))


# ---------------------------------------------------------------------------
# Constituent layer (filled trees)
# ---------------------------------------------------------------------------

class Constituent(ABC):
    @abstractmethod
    def render(self, head_initial: bool) -> Node: ...


class Terminal(Constituent):
    def __init__(self, label: str, word: str) -> None:
        self.label = label
        self.word = word

    def render(self, head_initial: bool) -> Node:
        return Node(self.label, word=self.word)


class Phrase(Constituent):
    label: ClassVar[str] = "Phrase"

    def __init__(self, head: Constituent, complement: Constituent) -> None:
        self.head = head
        self.complement = complement

    def get_head_initial(self, head_initial: bool = True) -> Node:
        return Node(self.label, [
            self.head.render(head_initial),
            self.complement.render(head_initial),
        ])

    def get_head_final(self) -> Node:
        return Node(self.label, [
            self.complement.render(False),
            self.head.render(False),
        ])

    def render(self, head_initial: bool) -> Node:
        return self.get_head_initial() if head_initial else self.get_head_final()


class _Invariant(Phrase):
    """Phrase whose daughter order is fixed (same in HI and HF)."""

    def get_head_final(self) -> Node:
        return self.get_head_initial(head_initial=False)


# --- NP_singular variants ---------------------------------------------------

class NPsBar(Constituent):
    """NP_singular -> N_singular (bare noun)."""

    def __init__(self, noun: str) -> None:
        self.noun = noun

    def render(self, head_initial: bool) -> Node:
        return Node("NP_singular", [Node("N_singular", word=self.noun)])


class NPsAdj(Constituent):
    """NP_singular -> NP_singular_adj -> AdjP N_singular (order invariant HI/HF)."""

    def __init__(self, adj: str, noun: str) -> None:
        self.adj = adj
        self.noun = noun

    def render(self, head_initial: bool) -> Node:
        adj_node = Node("AdjP", [Node("Adj", word=self.adj)])
        n_node = Node("N_singular", word=self.noun)
        np_adj = Node("NP_singular_adj", [adj_node, n_node])
        return Node("NP_singular", [np_adj])


class NPsRel(Constituent):
    """NP_singular -> NP_singular CP_rel (HI) / CP_rel NP_singular (HF).

    core is the inner NP_singular: either bare (NPsBar) or adj-modified (NPsAdj).
    """

    def __init__(self, core: Constituent, cp_rel: "CP_rel") -> None:
        self.core = core
        self.cp_rel = cp_rel

    def render(self, head_initial: bool) -> Node:
        core_node = self.core.render(head_initial)
        rel_node = self.cp_rel.render(head_initial)
        if head_initial:
            return Node("NP_singular", [core_node, rel_node])
        else:
            return Node("NP_singular", [rel_node, core_node])


# --- Adjunct phrases ---------------------------------------------------------

class AdjP(Constituent):
    """AdjP -> Adj (invariant)."""

    def __init__(self, adj: str) -> None:
        self.adj = adj

    def render(self, head_initial: bool) -> Node:
        return Node("AdjP", [Node("Adj", word=self.adj)])


class AdvP(Constituent):
    """AdvP -> Adv (invariant)."""

    def __init__(self, adv: str) -> None:
        self.adv = adv

    def render(self, head_initial: bool) -> Node:
        return Node("AdvP", [Node("Adv", word=self.adv)])


# --- DP, ProperDP ------------------------------------------------------------

class DP(_Invariant):
    """DP -> D NP_singular (order invariant; NP_singular is a full phrase)."""

    label = "DP"

    def __init__(self, det: str, nps: Constituent) -> None:
        super().__init__(Terminal("D", det), nps)


class ProperDP(Constituent):
    """DP -> NP_proper. Renders as bare [DP word]."""

    label = "DP"

    def __init__(self, proper: str) -> None:
        self.proper = proper

    def render(self, head_initial: bool) -> Node:
        return Node(self.label, word=self.proper)


DPLike = Union[DP, ProperDP]


# --- Relative clause constituents --------------------------------------------

class VP_obj_gap(Constituent):
    """VP_obj_gap -> V_dp (transitive verb; object is the relativized head)."""

    def __init__(self, verb: str) -> None:
        self.verb = verb

    def render(self, head_initial: bool) -> Node:
        return Node("VP_obj_gap", [Node("V_dp", word=self.verb)])


class S_subj_gap(Constituent):
    """S_subj_gap -> VP (subject has been relativized out)."""

    def __init__(self, vp: "VP") -> None:
        self.vp = vp

    def render(self, head_initial: bool) -> Node:
        return Node("S_subj_gap", [self.vp.render(head_initial)])


class S_obj_gap(_Invariant):
    """S_obj_gap -> DP VP_obj_gap (object relativized; DP is the remaining subject)."""

    label = "S_obj_gap"

    def __init__(self, dp: DPLike, vp_gap: VP_obj_gap) -> None:
        super().__init__(dp, vp_gap)


SGapLike = Union[S_subj_gap, S_obj_gap]


class CP_rel(Phrase):
    """CP_rel -> C_rel S_gap (HI) / S_gap C_rel (HF)."""

    label = "CP_rel"

    def __init__(self, c_rel: str, s_gap: SGapLike) -> None:
        super().__init__(Terminal("C_rel", c_rel), s_gap)


# --- Main clause constituents ------------------------------------------------

class VP(Constituent):
    """Unified VP covering all six variants (±adverb × intrans/dp/cp)."""

    label = "VP"

    def __init__(
        self,
        verb: str,
        verb_pos: str,  # "V_intrans" | "V_dp" | "V_cp"
        complement: "DPLike | CP_sent | None" = None,
        adv: "str | None" = None,
    ) -> None:
        self.verb = verb
        self.verb_pos = verb_pos
        self.complement = complement
        self.adv = adv

    def render(self, head_initial: bool) -> Node:
        v_node = Node(self.verb_pos, word=self.verb)
        adv_node = (
            Node("AdvP", [Node("Adv", word=self.adv)]) if self.adv else None
        )

        if self.complement is None:
            # Intransitive: VP -> V_intrans [AdvP]
            if adv_node:
                # VP_intrans_adv: HI=[V Adv], HF=[Adv V]
                inner_children = [v_node, adv_node] if head_initial else [adv_node, v_node]
                inner = Node("VP_intrans_adv", inner_children)
                return Node(self.label, [inner])
            return Node(self.label, [v_node])

        comp_node = self.complement.render(head_initial)

        if adv_node:
            # VP_(dp|cp)_adv DP/CP:
            #   HI: [VP [VP_X_adv [V Adv]] [comp]]
            #   HF: [VP [comp] [VP_X_adv [Adv V]]]
            adv_label = "VP_dp_adv" if self.verb_pos == "V_dp" else "VP_cp_adv"
            inner_children = [v_node, adv_node] if head_initial else [adv_node, v_node]
            inner = Node(adv_label, inner_children)
            if head_initial:
                return Node(self.label, [inner, comp_node])
            else:
                return Node(self.label, [comp_node, inner])

        # No adverb: transitive or clause-taking
        if head_initial:
            return Node(self.label, [v_node, comp_node])
        return Node(self.label, [comp_node, v_node])


class CP_sent(Phrase):
    """CP_sent -> C S (HI) / S C (HF). (Renamed from CP in v1.)"""

    label = "CP_sent"

    def __init__(self, comp: str, clause: "S") -> None:
        super().__init__(Terminal("C", comp), clause)


class S(_Invariant):
    """S -> DP VP (invariant order)."""

    label = "S"

    def __init__(self, subject: DPLike, predicate: VP) -> None:
        super().__init__(subject, predicate)


# ---------------------------------------------------------------------------
# Frame layer (parse skeletons)
# ---------------------------------------------------------------------------

class Frame(ABC):
    @abstractmethod
    def bracketed_skeleton(self, fine: bool = True) -> str: ...

    @abstractmethod
    def to_constituent(self, rng: random.Random, constraints: "LexicalConstraints") -> Constituent: ...

    @abstractmethod
    def depth(self) -> int:
        """Max nesting depth of any CP (CP_sent or CP_rel) in this frame."""

    @abstractmethod
    def tree_height_estimate(self) -> int:
        """Approximate max tree height for this frame (from root S = depth 1)."""


# --- NP_singular frames ------------------------------------------------------

class FrameNPsBar(Frame):
    """NP_singular -> N_singular."""

    def bracketed_skeleton(self, fine: bool = True) -> str:
        label = "N_singular" if fine else "N"
        return f"[NP_singular [{label} _]]"

    def to_constituent(self, rng, constraints) -> NPsBar:
        return NPsBar(constraints.pick("N_singular", rng))

    def depth(self) -> int:
        return 0

    def tree_height_estimate(self) -> int:
        return 2  # [NP_singular [N_singular _]]


class FrameNPsAdj(Frame):
    """NP_singular -> NP_singular_adj -> AdjP N_singular."""

    def bracketed_skeleton(self, fine: bool = True) -> str:
        adj = "Adj" if fine else "Adj"
        n = "N_singular" if fine else "N"
        return f"[NP_singular [NP_singular_adj [AdjP [{adj} _]] [{n} _]]]"

    def to_constituent(self, rng, constraints) -> NPsAdj:
        return NPsAdj(constraints.pick("Adj", rng), constraints.pick("N_singular", rng))

    def depth(self) -> int:
        return 0

    def tree_height_estimate(self) -> int:
        return 4  # [NP_singular [NP_singular_adj [AdjP [Adj _]] [N _]]]


class FrameNPsRel(Frame):
    """NP_singular -> NP_singular CP_rel."""

    def __init__(self, core: "FrameNPsBar | FrameNPsAdj", cp_rel: "FrameCPrel") -> None:
        self.core = core
        self.cp_rel = cp_rel

    def bracketed_skeleton(self, fine: bool = True) -> str:
        return f"[NP_singular {self.core.bracketed_skeleton(fine)} {self.cp_rel.bracketed_skeleton(fine)}]"

    def to_constituent(self, rng, constraints) -> NPsRel:
        core = self.core.to_constituent(rng, constraints)
        cp = self.cp_rel.to_constituent(rng, constraints)
        return NPsRel(core, cp)

    def depth(self) -> int:
        return self.cp_rel.depth()

    def tree_height_estimate(self) -> int:
        return 1 + max(self.core.tree_height_estimate(), self.cp_rel.tree_height_estimate())


# --- DP frames ---------------------------------------------------------------

class FrameDPCommon(Frame):
    """DP -> D NP_singular."""

    def __init__(self, nps_frame: "FrameNPsBar | FrameNPsAdj | FrameNPsRel") -> None:
        self.nps_frame = nps_frame

    def bracketed_skeleton(self, fine: bool = True) -> str:
        return f"[DP [D _] {self.nps_frame.bracketed_skeleton(fine)}]"

    def to_constituent(self, rng, constraints) -> DP:
        det = constraints.pick("D", rng)
        nps = self.nps_frame.to_constituent(rng, constraints)
        return DP(det, nps)

    def depth(self) -> int:
        return self.nps_frame.depth()

    def tree_height_estimate(self) -> int:
        return 1 + self.nps_frame.tree_height_estimate()


class FrameDPProper(Frame):
    """DP -> NP_proper (bare [DP word])."""

    def bracketed_skeleton(self, fine: bool = True) -> str:
        return "[DP [N_proper _]]" if fine else "[DP _]"

    def to_constituent(self, rng, constraints) -> ProperDP:
        return ProperDP(constraints.pick("N_proper", rng))

    def depth(self) -> int:
        return 0

    def tree_height_estimate(self) -> int:
        return 1  # ProperDP renders as a leaf node


# --- VP frames ---------------------------------------------------------------

class FrameVPIntrans(Frame):
    """VP -> V_intrans."""

    def bracketed_skeleton(self, fine: bool = True) -> str:
        v = "V_intrans" if fine else "V"
        return f"[VP [{v} _]]"

    def to_constituent(self, rng, constraints) -> VP:
        return VP(constraints.pick("V_intrans", rng), "V_intrans")

    def depth(self) -> int:
        return 0

    def tree_height_estimate(self) -> int:
        return 2


class FrameVPIntransAdv(Frame):
    """VP -> VP_intrans_adv (V_intrans + AdvP)."""

    def bracketed_skeleton(self, fine: bool = True) -> str:
        v = "V_intrans" if fine else "V"
        return f"[VP [VP_intrans_adv [{v} _] [AdvP [Adv _]]]]"

    def to_constituent(self, rng, constraints) -> VP:
        return VP(
            constraints.pick("V_intrans", rng), "V_intrans",
            adv=constraints.pick("Adv", rng),
        )

    def depth(self) -> int:
        return 0

    def tree_height_estimate(self) -> int:
        return 4  # [VP [VP_intrans_adv [V _] [AdvP [Adv _]]]]


class FrameVPdp(Frame):
    """VP -> V_dp DP."""

    def __init__(self, dp: "FrameDPCommon | FrameDPProper") -> None:
        self.dp = dp

    def bracketed_skeleton(self, fine: bool = True) -> str:
        v = "V_dp" if fine else "V"
        return f"[VP [{v} _] {self.dp.bracketed_skeleton(fine)}]"

    def to_constituent(self, rng, constraints) -> VP:
        return VP(
            constraints.pick("V_dp", rng), "V_dp",
            complement=self.dp.to_constituent(rng, constraints),
        )

    def depth(self) -> int:
        return self.dp.depth()

    def tree_height_estimate(self) -> int:
        return 1 + max(2, self.dp.tree_height_estimate())


class FrameVPdpAdv(Frame):
    """VP -> VP_dp_adv DP (V_dp + AdvP + DP)."""

    def __init__(self, dp: "FrameDPCommon | FrameDPProper") -> None:
        self.dp = dp

    def bracketed_skeleton(self, fine: bool = True) -> str:
        v = "V_dp" if fine else "V"
        return f"[VP [VP_dp_adv [{v} _] [AdvP [Adv _]]] {self.dp.bracketed_skeleton(fine)}]"

    def to_constituent(self, rng, constraints) -> VP:
        return VP(
            constraints.pick("V_dp", rng), "V_dp",
            complement=self.dp.to_constituent(rng, constraints),
            adv=constraints.pick("Adv", rng),
        )

    def depth(self) -> int:
        return self.dp.depth()

    def tree_height_estimate(self) -> int:
        return 1 + max(4, self.dp.tree_height_estimate())


class FrameVPcp(Frame):
    """VP -> V_cp CP_sent."""

    def __init__(self, cp: "FrameCPsent") -> None:
        self.cp = cp

    def bracketed_skeleton(self, fine: bool = True) -> str:
        v = "V_cp" if fine else "V"
        return f"[VP [{v} _] {self.cp.bracketed_skeleton(fine)}]"

    def to_constituent(self, rng, constraints) -> VP:
        return VP(
            constraints.pick("V_cp", rng), "V_cp",
            complement=self.cp.to_constituent(rng, constraints),
        )

    def depth(self) -> int:
        return self.cp.depth()

    def tree_height_estimate(self) -> int:
        return 1 + self.cp.tree_height_estimate()


class FrameVPcpAdv(Frame):
    """VP -> VP_cp_adv CP_sent (V_cp + AdvP + CP_sent)."""

    def __init__(self, cp: "FrameCPsent") -> None:
        self.cp = cp

    def bracketed_skeleton(self, fine: bool = True) -> str:
        v = "V_cp" if fine else "V"
        return f"[VP [VP_cp_adv [{v} _] [AdvP [Adv _]]] {self.cp.bracketed_skeleton(fine)}]"

    def to_constituent(self, rng, constraints) -> VP:
        return VP(
            constraints.pick("V_cp", rng), "V_cp",
            complement=self.cp.to_constituent(rng, constraints),
            adv=constraints.pick("Adv", rng),
        )

    def depth(self) -> int:
        return self.cp.depth()

    def tree_height_estimate(self) -> int:
        return 1 + max(4, self.cp.tree_height_estimate())


# --- CP frames ---------------------------------------------------------------

class FrameCPsent(Frame):
    """CP_sent -> C S. Pushes a fresh clause scope before filling S."""

    def __init__(self, s: "FrameS") -> None:
        self.s = s

    def bracketed_skeleton(self, fine: bool = True) -> str:
        return f"[CP_sent [C _] {self.s.bracketed_skeleton(fine)}]"

    def to_constituent(self, rng, constraints) -> CP_sent:
        comp = constraints.pick("C", rng)
        constraints.push_clause()
        try:
            clause = self.s.to_constituent(rng, constraints)
        finally:
            constraints.pop_clause()
        return CP_sent(comp, clause)

    def depth(self) -> int:
        return 1 + self.s.depth()

    def tree_height_estimate(self) -> int:
        return 1 + self.s.tree_height_estimate()


class FrameVPobjGap(Frame):
    """VP_obj_gap -> V_dp (object has been relativized out)."""

    def bracketed_skeleton(self, fine: bool = True) -> str:
        v = "V_dp" if fine else "V"
        return f"[VP_obj_gap [{v} _]]"

    def to_constituent(self, rng, constraints) -> VP_obj_gap:
        return VP_obj_gap(constraints.pick("V_dp", rng))

    def depth(self) -> int:
        return 0

    def tree_height_estimate(self) -> int:
        return 2


class FrameSsubjGap(Frame):
    """S_subj_gap -> VP."""

    def __init__(self, vp: "FrameVP") -> None:
        self.vp = vp

    def bracketed_skeleton(self, fine: bool = True) -> str:
        return f"[S_subj_gap {self.vp.bracketed_skeleton(fine)}]"

    def to_constituent(self, rng, constraints) -> S_subj_gap:
        return S_subj_gap(self.vp.to_constituent(rng, constraints))

    def depth(self) -> int:
        return self.vp.depth()

    def tree_height_estimate(self) -> int:
        return 1 + self.vp.tree_height_estimate()


class FrameSobjGap(Frame):
    """S_obj_gap -> DP VP_obj_gap."""

    def __init__(self, dp: "FrameDP", vp_gap: FrameVPobjGap) -> None:
        self.dp = dp
        self.vp_gap = vp_gap

    def bracketed_skeleton(self, fine: bool = True) -> str:
        return f"[S_obj_gap {self.dp.bracketed_skeleton(fine)} {self.vp_gap.bracketed_skeleton(fine)}]"

    def to_constituent(self, rng, constraints) -> S_obj_gap:
        dp = self.dp.to_constituent(rng, constraints)
        vp_gap = self.vp_gap.to_constituent(rng, constraints)
        return S_obj_gap(dp, vp_gap)

    def depth(self) -> int:
        return self.dp.depth()

    def tree_height_estimate(self) -> int:
        return 1 + max(self.dp.tree_height_estimate(), self.vp_gap.tree_height_estimate())


class FrameCPrel(Frame):
    """CP_rel -> C_rel S_gap. Pushes a fresh clause scope (like FrameCPsent)."""

    def __init__(self, s_gap: "FrameSsubjGap | FrameSobjGap") -> None:
        self.s_gap = s_gap

    def bracketed_skeleton(self, fine: bool = True) -> str:
        c = "C_rel" if fine else "C"
        return f"[CP_rel [{c} _] {self.s_gap.bracketed_skeleton(fine)}]"

    def to_constituent(self, rng, constraints) -> CP_rel:
        c_rel = constraints.pick("C_rel", rng)
        constraints.push_clause()
        try:
            s_gap = self.s_gap.to_constituent(rng, constraints)
        finally:
            constraints.pop_clause()
        return CP_rel(c_rel, s_gap)

    def depth(self) -> int:
        return 1 + self.s_gap.depth()

    def tree_height_estimate(self) -> int:
        return 1 + self.s_gap.tree_height_estimate()


# --- S frame -----------------------------------------------------------------

FrameDP = Union[FrameDPCommon, FrameDPProper]
FrameVP = Union[
    FrameVPIntrans, FrameVPIntransAdv,
    FrameVPdp, FrameVPdpAdv,
    FrameVPcp, FrameVPcpAdv,
]


class FrameS(Frame):
    """S -> DP VP."""

    def __init__(self, dp: FrameDP, vp: FrameVP) -> None:
        self.dp = dp
        self.vp = vp

    def bracketed_skeleton(self, fine: bool = True) -> str:
        return f"[S {self.dp.bracketed_skeleton(fine)} {self.vp.bracketed_skeleton(fine)}]"

    def to_constituent(self, rng, constraints) -> S:
        return S(
            self.dp.to_constituent(rng, constraints),
            self.vp.to_constituent(rng, constraints),
        )

    def depth(self) -> int:
        return max(self.dp.depth(), self.vp.depth())

    def tree_height_estimate(self) -> int:
        return 1 + max(self.dp.tree_height_estimate(), self.vp.tree_height_estimate())


# ---------------------------------------------------------------------------
# Frame enumeration
# ---------------------------------------------------------------------------

def _enum_nps(max_depth: int) -> Iterator[FrameNPsBar | FrameNPsAdj | FrameNPsRel]:
    """All NP_singular frames with CP depth <= max_depth."""
    yield FrameNPsBar()
    yield FrameNPsAdj()
    if max_depth >= 1:
        for vp in _enum_vp(max_depth - 1):
            for core in (FrameNPsBar(), FrameNPsAdj()):
                yield FrameNPsRel(core, FrameCPrel(FrameSsubjGap(vp)))
        for dp in _enum_dp(max_depth - 1):
            for core in (FrameNPsBar(), FrameNPsAdj()):
                yield FrameNPsRel(core, FrameCPrel(FrameSobjGap(dp, FrameVPobjGap())))


def _enum_dp(max_depth: int) -> Iterator[FrameDP]:
    yield FrameDPProper()
    for nps in _enum_nps(max_depth):
        yield FrameDPCommon(nps)


def _enum_vp(max_depth: int) -> Iterator[FrameVP]:
    yield FrameVPIntrans()
    yield FrameVPIntransAdv()
    for dp in _enum_dp(max_depth):
        yield FrameVPdp(dp)
        yield FrameVPdpAdv(dp)
    if max_depth >= 1:
        for s in _enum_s(max_depth - 1):
            yield FrameVPcp(FrameCPsent(s))
            yield FrameVPcpAdv(FrameCPsent(s))


def _enum_s(max_depth: int) -> Iterator[FrameS]:
    for dp in _enum_dp(max_depth):
        for vp in _enum_vp(max_depth):
            yield FrameS(dp, vp)


def enumerate_frames(max_depth: int) -> Iterator[FrameS]:
    """Yield every S frame with all-CP nesting depth <= max_depth."""
    return _enum_s(max_depth)


# ---------------------------------------------------------------------------
# Lexical constraints
# ---------------------------------------------------------------------------

class LexicalConstraints:
    """Clause-scoped no-repeat for N_singular. Other POS sample freely.

    push_clause() / pop_clause() are called by FrameCPsent and FrameCPrel.
    """

    def __init__(self) -> None:
        self._used_n_singular: list[set[str]] = [set()]

    def push_clause(self) -> None:
        self._used_n_singular.append(set())

    def pop_clause(self) -> None:
        self._used_n_singular.pop()

    def pick(self, pos: str, rng: random.Random) -> str:
        pool = VOCAB[pos]
        if pos == "N_singular":
            available = [w for w in pool if w not in self._used_n_singular[-1]]
            if not available:
                available = pool
            word = rng.choice(available)
            self._used_n_singular[-1].add(word)
            return word
        return rng.choice(pool)


# ---------------------------------------------------------------------------
# Sampler utilities
# ---------------------------------------------------------------------------

def sample_from_frame(
    frame: Frame,
    rng: random.Random,
    constraints: "LexicalConstraints | None" = None,
) -> Constituent:
    if constraints is None:
        constraints = LexicalConstraints()
    return frame.to_constituent(rng, constraints)


def _collect_leaves_preorder(node: Node) -> list[Node]:
    """Left-to-right preorder leaf collection."""
    if node.is_leaf:
        return [node]
    leaves: list[Node] = []
    for c in node.children or []:
        leaves.extend(_collect_leaves_preorder(c))
    return leaves


def _build_tagged_frame(tree: Node, pos_map: dict[tuple[str, int], int]) -> str:
    """Serialize tree to bracketed string, replacing leaf words with POS_N tags.

    pos_map maps (word, occurrence_index) -> HI position number.
    """
    occ: dict[str, int] = {}

    def go(node: Node) -> str:
        if node.is_leaf:
            word = node.word or ""
            n = occ.get(word, 0)
            occ[word] = n + 1
            tag = pos_map[(word, n)]
            return f"[{node.label}_{tag}]"
        inner = " ".join(go(c) for c in (node.children or []))
        return f"[{node.label} {inner}]"

    return go(tree)


def _position_tagged_frames(hi_tree: Node, hf_tree: Node) -> tuple[str, str]:
    """Build (hi_frame_tagged, hf_frame_tagged) with 1-indexed position labels."""
    hi_leaves = _collect_leaves_preorder(hi_tree)
    # Build map: (word, occurrence_in_HI) -> position (1-indexed)
    word_occ: dict[str, int] = {}
    pos_map: dict[tuple[str, int], int] = {}
    for i, leaf in enumerate(hi_leaves, start=1):
        word = leaf.word or ""
        occ = word_occ.get(word, 0)
        pos_map[(word, occ)] = i
        word_occ[word] = occ + 1

    hi_tagged = _build_tagged_frame(hi_tree, pos_map)
    hf_tagged = _build_tagged_frame(hf_tree, pos_map)
    return hi_tagged, hf_tagged


# ---------------------------------------------------------------------------
# FrameSentencePair and pair sampling
# ---------------------------------------------------------------------------

@dataclass
class FrameSentencePair:
    hi_tokens: list[str]
    hf_tokens: list[str]
    hi_bracketed: str
    hf_bracketed: str
    hi_frame_tagged: str   # HI skeleton with 1-indexed terminal position tags
    hf_frame_tagged: str   # HF skeleton with same position tags, reordered
    frame_id: str          # coarse-label skeleton (defines split partition)
    depth: int             # max nesting of any CP (CP_sent or CP_rel)
    tree_height: int       # max leaf depth in the full constituency tree
    is_ambiguous: bool = field(default=False)  # set True in run.py cross-frame check

    @property
    def hi(self) -> str:
        return " ".join(self.hi_tokens)

    @property
    def hf(self) -> str:
        return " ".join(self.hf_tokens)

    def to_dict(self) -> dict:
        return {
            "hi": self.hi,
            "hf": self.hf,
            "hi_tokens": self.hi_tokens,
            "hf_tokens": self.hf_tokens,
            "hi_bracketed": self.hi_bracketed,
            "hf_bracketed": self.hf_bracketed,
            "hi_frame_tagged": self.hi_frame_tagged,
            "hf_frame_tagged": self.hf_frame_tagged,
            "frame_id": self.frame_id,
            "depth": self.depth,
            "tree_height": self.tree_height,
            "is_ambiguous": self.is_ambiguous,
            "n_tokens": len(self.hi_tokens),
        }


def sample_pair_from_frame(frame: FrameS, rng: random.Random) -> FrameSentencePair:
    constituent = sample_from_frame(frame, rng)
    hi_tree = constituent.get_head_initial()
    hf_tree = constituent.get_head_final()
    hi_tagged, hf_tagged = _position_tagged_frames(hi_tree, hf_tree)
    return FrameSentencePair(
        hi_tokens=linearize(hi_tree),
        hf_tokens=linearize(hf_tree),
        hi_bracketed=bracketed(hi_tree),
        hf_bracketed=bracketed(hf_tree),
        hi_frame_tagged=hi_tagged,
        hf_frame_tagged=hf_tagged,
        frame_id=frame.bracketed_skeleton(fine=False),
        depth=frame.depth(),
        tree_height=compute_tree_height(hi_tree),
    )


def sample_pairs_from_frame(
    frame: FrameS, k: int, rng: random.Random
) -> list[FrameSentencePair]:
    """Up to k unique (by HI string) lexicalizations of a frame."""
    pairs: list[FrameSentencePair] = []
    seen: set[str] = set()
    max_attempts = max(k * 50, k + 1)
    for _ in range(max_attempts):
        if len(pairs) >= k:
            break
        pair = sample_pair_from_frame(frame, rng)
        if pair.hi in seen:
            continue
        seen.add(pair.hi)
        pairs.append(pair)
    return pairs


# ---------------------------------------------------------------------------
# CLI: frame viewer
# ---------------------------------------------------------------------------

def _print_frames(max_depth: int, fine: bool) -> None:
    by_depth: dict[int, list[FrameS]] = {}
    for f in enumerate_frames(max_depth):
        by_depth.setdefault(f.depth(), []).append(f)
    cumulative = 0
    for d in sorted(by_depth):
        frames = by_depth[d]
        cumulative += len(frames)
        print(f"\n=== Exact depth {d}: {len(frames)} frames (cumulative: {cumulative}) ===")
        for i, frame in enumerate(frames):
            print(f"  [{d}.{i:02d}] {frame.bracketed_skeleton(fine=fine)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--coarse", action="store_true")
    args = parser.parse_args()
    _print_frames(args.max_depth, fine=not args.coarse)


if __name__ == "__main__":
    main()

"""CFG productions and lexicon for the toy head-initial / head-final grammar.

The HI and HF grammars from the planning doc are identical in structure and
lexicon. They differ only in child ordering for head-complement binaries
(DP -> D NP and VP -> V_trans DP). We encode one abstract grammar; ordering
is applied at linearization time (see tree.py).
"""
from dataclasses import dataclass

# Nonterminal / preterminal labels
S = "S"
DP = "DP"
NP = "NP"
VP = "VP"
D = "D"
N = "N"
V_TRANS = "V_trans"
V_INTRANS = "V_intrans"

NONTERMINALS = frozenset({S, DP, NP, VP})
PRETERMINALS = frozenset({D, N, V_TRANS, V_INTRANS})

TERMINALS: dict[str, list[str]] = {
    D: ["d1", "d2", "d3", "d4", "d5"],
    N: ["n1", "n2", "n3", "n4", "n5"],
    V_TRANS: ["V_trans1", "V_trans2", "V_trans3", "V_trans4"],
    V_INTRANS: ["V_intrans1", "V_intrans2", "V_intrans3", "V_intrans4"],
}


@dataclass(frozen=True)
class Production:
    """A CFG production. `head_index` marks the head child for reorderable
    head-complement binaries; None means the production is not reordered
    between HI and HF (either unary or structurally identical in both)."""
    lhs: str
    rhs: tuple[str, ...]
    head_index: int | None = None

    @property
    def reorderable(self) -> bool:
        return self.head_index is not None


# Children are stored in head-initial order. For reorderable productions,
# HF linearization reverses the children.
PRODUCTIONS: dict[str, list[Production]] = {
    S: [Production(S, (DP, VP), head_index=None)],
    DP: [Production(DP, (D, NP), head_index=0)],
    NP: [Production(NP, (N,), head_index=None)],
    VP: [
        Production(VP, (V_TRANS, DP), head_index=0),
        Production(VP, (V_INTRANS,), head_index=None),
    ],
}

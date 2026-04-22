"""Derivation tree with HI/HF linearization.

Children of internal nodes are stored in head-initial order. HF linearization
reverses children for nodes flagged `reorderable`.
"""
from dataclasses import dataclass, field
from typing import Literal

Order = Literal["HI", "HF"]


@dataclass
class Tree:
    label: str
    children: list["Tree"] = field(default_factory=list)
    terminal: str | None = None
    reorderable: bool = False

    @property
    def is_preterminal(self) -> bool:
        return self.terminal is not None

    def _ordered_children(self, order: Order) -> list["Tree"]:
        if order == "HF" and self.reorderable:
            return list(reversed(self.children))
        if order not in ("HI", "HF"):
            raise ValueError(f"order must be 'HI' or 'HF', got {order!r}")
        return self.children

    def linearize(self, order: Order) -> list[str]:
        if self.is_preterminal:
            return [self.terminal]  # type: ignore[list-item]
        return [tok for c in self._ordered_children(order) for tok in c.linearize(order)]

    def to_nltk_str(self, order: Order = "HI") -> str:
        """Penn Treebank bracket format, parseable by nltk.Tree.fromstring."""
        if self.is_preterminal:
            return f"({self.label} {self.terminal})"
        inner = " ".join(c.to_nltk_str(order) for c in self._ordered_children(order))
        return f"({self.label} {inner})"

    def __repr__(self) -> str:
        return self.to_nltk_str("HI")

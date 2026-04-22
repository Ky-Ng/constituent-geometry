"""Sample derivation trees from the toy CFG.

Productions are chosen uniformly at each nonterminal by default; terminals
are chosen uniformly from each preterminal's lexicon. Pass `weights` to
override per-nonterminal production weights.
"""
import random
from typing import Mapping, Sequence

from .rules import PRETERMINALS, PRODUCTIONS, TERMINALS, S
from .tree import Tree


def sample_tree(
    rng: random.Random,
    root: str = S,
    weights: Mapping[str, Sequence[float]] | None = None,
) -> Tree:
    """Sample a derivation tree rooted at `root`.

    Children are stored in head-initial order. Use `tree.linearize('HI'|'HF')`
    or `tree.to_nltk_str('HI'|'HF')` to get the two surface realizations.
    """
    if root in PRETERMINALS:
        return Tree(label=root, terminal=rng.choice(TERMINALS[root]))

    prods = PRODUCTIONS[root]
    if weights is not None and root in weights:
        w = list(weights[root])
        if len(w) != len(prods):
            raise ValueError(f"weights[{root!r}] has length {len(w)}, expected {len(prods)}")
        prod = rng.choices(prods, weights=w, k=1)[0]
    else:
        prod = rng.choice(prods)

    children = [sample_tree(rng, sym, weights=weights) for sym in prod.rhs]
    return Tree(label=root, children=children, reorderable=prod.reorderable)

"""Generate Head-Initial (HI) / Head-Final (HF) sentence pairs from the toy CFG.

The grammar (see ``GRAMMAR.md``) shares its nonterminal expansions between the
HI and HF variants; the two differ only in *head placement*. We build one
derivation as a tree of :class:`Constituent` objects and render it twice.

Every phrase is a binary ``head`` + ``complement`` node (:class:`Phrase`):

* Head-initial puts the head first: ``[Label HEAD complement]``.
* Head-final puts the head last:  ``[Label complement HEAD]``.

To add a new phrase type, subclass :class:`Phrase`, set ``label``, and adapt
the constructor. Flipping is the default. Phrases whose daughter order is fixed
across head directions (here ``S`` and ``DP``) override :meth:`get_head_final`
to defer to :meth:`get_head_initial`.

HI rules                HF rules
--------                --------
S  -> DP VP             S  -> DP VP
DP -> D NP              DP -> D NP
VP -> V DP             VP -> DP V
VP -> V PP             VP -> PP V
VP -> V CP             VP -> CP V
PP -> P DP             PP -> DP P
CP -> C S              CP -> S C

Terminals (from GRAMMAR.md). The section headed "DP" lists the determiners
``the``/``a``; these are the ``D`` terminals of the ``DP -> D NP`` rule.
"""

from __future__ import annotations

import argparse
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

# --- Terminal vocabulary (exactly as declared in GRAMMAR.md) -----------------
DETERMINERS = ["the", "a"]          # D  (listed under the "DP" header)
NOUNS = ["dog", "cat", "boy", "girl"]  # NP
VERBS = ["likes", "believes"]       # V
COMPLEMENTIZERS = ["that"]          # C
PREPOSITIONS = ["to", "at"]         # P


# --- Generic constituent tree (for bracketing / linearization) ---------------
@dataclass
class Node:
    """A labelled constituent tree node. Leaves carry ``word`` and no children."""

    label: str
    children: "list[Node] | None" = None
    word: "str | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.word is not None


def bracketed(tree: Node) -> str:
    """Serialize a tree as ``[Label child ...]`` (matches GRAMMAR.md examples)."""
    if tree.is_leaf:
        return f"[{tree.label} {tree.word}]"
    inner = " ".join(bracketed(c) for c in tree.children or [])
    return f"[{tree.label} {inner}]"


def linearize(tree: Node) -> list[str]:
    """Left-to-right terminal yield of a tree."""
    if tree.is_leaf:
        return [tree.word]  # type: ignore[list-item]
    words: list[str] = []
    for child in tree.children or []:
        words.extend(linearize(child))
    return words


# --- Derivation: a tree of constituents --------------------------------------
class Constituent(ABC):
    """Anything that can be rendered into a :class:`Node` in a head direction."""

    @abstractmethod
    def render(self, head_initial: bool) -> Node:
        """Render this constituent (and its descendants) head-initial or -final."""


class Terminal(Constituent):
    """A leaf carrying a part-of-speech ``label`` (e.g. ``V``) and its ``word``.

    Terminals have no head direction of their own, so ``render`` ignores it.
    """

    def __init__(self, label: str, word: str) -> None:
        self.label = label
        self.word = word

    def render(self, head_initial: bool) -> Node:
        return Node(self.label, word=self.word)


class Phrase(Constituent):
    """A binary phrase: a ``head`` and its ``complement``.

    The default phrase *flips*: head-initial leads with the head, head-final
    trails with it. Daughters are always rendered in the global head direction,
    so the whole tree stays internally consistent.
    """

    label: ClassVar[str] = "Phrase"

    def __init__(self, head: Constituent, complement: Constituent) -> None:
        self.head = head
        self.complement = complement

    def get_head_initial(self, head_initial: bool = True) -> Node:
        """Head-first layout ``[Label HEAD complement]``.

        ``head_initial`` is the direction passed down to daughters. It is
        ``True`` for normal head-initial rendering; invariant phrases call this
        with ``False`` so their daughters stay head-final while the layout is
        unchanged.
        """
        return Node(self.label, [self.head.render(head_initial), self.complement.render(head_initial)])

    def get_head_final(self) -> Node:
        """Head-last layout ``[Label complement HEAD]`` (daughters head-final)."""
        return Node(self.label, [self.complement.render(False), self.head.render(False)])

    def render(self, head_initial: bool) -> Node:
        return self.get_head_initial() if head_initial else self.get_head_final()


class _Invariant(Phrase):
    """Phrase whose daughter order is fixed across head directions.

    Its head-final form reuses the head-initial layout; only the direction
    propagated to daughters changes.
    """

    def get_head_final(self) -> Node:
        return self.get_head_initial(head_initial=False)


class DP(_Invariant):
    """DP -> D NP (order identical in HI and HF)."""

    label = "DP"

    def __init__(self, det: str, noun: str) -> None:
        super().__init__(Terminal("D", det), Terminal("NP", noun))


class S(_Invariant):
    """S -> DP VP (subject then predicate; order identical in HI and HF).

    ``head`` holds the subject DP and ``complement`` the predicate VP — here
    "head" just names the daughter that leads the fixed layout.
    """

    label = "S"

    def __init__(self, subject: "DP", predicate: "VP") -> None:
        super().__init__(subject, predicate)


class VP(Phrase):
    """VP -> V {DP|PP|CP} (HI) / {DP|PP|CP} V (HF)."""

    label = "VP"

    def __init__(self, verb: str, complement: "DP | PP | CP") -> None:
        super().__init__(Terminal("V", verb), complement)


class PP(Phrase):
    """PP -> P DP (HI) / DP P (HF)."""

    label = "PP"

    def __init__(self, prep: str, obj: "DP") -> None:
        super().__init__(Terminal("P", prep), obj)


class CP(Phrase):
    """CP -> C S (HI) / S C (HF)."""

    label = "CP"

    def __init__(self, comp: str, clause: "S") -> None:
        super().__init__(Terminal("C", comp), clause)


# --- Sampling ----------------------------------------------------------------
@dataclass
class SentencePair:
    """A single derivation rendered in both head directions."""

    hi_tokens: list[str]
    hf_tokens: list[str]
    hi_bracketed: str
    hf_bracketed: str

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
        }


def _gen_dp(rng: random.Random) -> DP:
    return DP(rng.choice(DETERMINERS), rng.choice(NOUNS))


def _gen_pp(rng: random.Random) -> PP:
    return PP(rng.choice(PREPOSITIONS), _gen_dp(rng))


def _gen_cp(rng: random.Random, depth: int, max_depth: int) -> CP:
    return CP(rng.choice(COMPLEMENTIZERS), _gen_s(rng, depth + 1, max_depth))


def _gen_vp(rng: random.Random, depth: int, max_depth: int) -> VP:
    # CP embeds another clause, so only allow it while we have depth budget.
    kinds = ["dp", "pp"]
    if depth < max_depth:
        kinds.append("cp")
    kind = rng.choice(kinds)
    verb = rng.choice(VERBS)
    if kind == "dp":
        return VP(verb, _gen_dp(rng))
    if kind == "pp":
        return VP(verb, _gen_pp(rng))
    return VP(verb, _gen_cp(rng, depth, max_depth))


def _gen_s(rng: random.Random, depth: int, max_depth: int) -> S:
    return S(_gen_dp(rng), _gen_vp(rng, depth, max_depth))


def generate_pair(rng: random.Random, max_depth: int = 2) -> SentencePair:
    """Sample one derivation and render it as an HI/HF pair.

    ``max_depth`` bounds the number of nested embedded clauses (CP). ``0`` forbids
    embedding entirely (only DP/PP complements).
    """
    derivation = _gen_s(rng, depth=0, max_depth=max_depth)
    hi_tree = derivation.get_head_initial()
    hf_tree = derivation.get_head_final()
    return SentencePair(
        hi_tokens=linearize(hi_tree),
        hf_tokens=linearize(hf_tree),
        hi_bracketed=bracketed(hi_tree),
        hf_bracketed=bracketed(hf_tree),
    )


def generate_pairs(
    n: int, max_depth: int = 2, seed: int = 42, unique: bool = True
) -> list[SentencePair]:
    """Sample ``n`` HI/HF pairs. With ``unique`` set, deduplicate by HI string.

    When ``unique`` is set and the grammar cannot supply ``n`` distinct sentences
    at the given depth, generation stops after a bounded number of attempts and
    returns however many distinct pairs were found.
    """
    rng = random.Random(seed)
    pairs: list[SentencePair] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = n * 50 if unique else n
    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        pair = generate_pair(rng, max_depth=max_depth)
        if unique:
            if pair.hi in seen:
                continue
            seen.add(pair.hi)
        pairs.append(pair)
    return pairs

# --- Enumeration ----------------------------------------------------------------
def count_sentences(max_depth: int) -> int:
    """Number of distinct sentences with CP-nesting depth <= max_depth.

    Closed form of the grammar's branching: for each S, DP has 8 forms and VP
    has nV * (nDP + nPP + [embed]) forms, where embed is the count of one-level-
    shallower sentences. With nV=2, nDP=8, nPP=16: VP(d) = 2 * (24 + S(d-1)).
    """
    s = 0
    for d in range(max_depth + 1):
        embedded = s if d >= 1 else 0  # S(d-1)
        vp = len(VERBS) * (len(DETERMINERS) * len(NOUNS) + len(PREPOSITIONS) * len(DETERMINERS) * len(NOUNS) + embedded)
        s = len(DETERMINERS) * len(NOUNS) * vp
    return s


def _enum_dp():
    for det in DETERMINERS:
        for noun in NOUNS:
            yield DP(det, noun)


def _enum_pp():
    for prep in PREPOSITIONS:
        for dp in _enum_dp():
            yield PP(prep, dp)


def _enum_vp(max_depth: int):
    for verb in VERBS:
        for dp in _enum_dp():
            yield VP(verb, dp)
        for pp in _enum_pp():
            yield VP(verb, pp)
        if max_depth >= 1:
            for comp in COMPLEMENTIZERS:
                for clause in _enum_s(max_depth - 1):
                    yield VP(verb, CP(comp, clause))


def _enum_s(max_depth: int):
    for dp in _enum_dp():
        for vp in _enum_vp(max_depth):
            yield S(dp, vp)


def enumerate_pairs(max_depth: int):
    """Lazily yield every distinct HI/HF SentencePair with depth <= max_depth.

    A generator (not a list), so callers can iterate it without holding every
    pair in memory at once.
    """
    for derivation in _enum_s(max_depth):
        hi = derivation.get_head_initial()
        hf = derivation.get_head_final()
        yield SentencePair(
            hi_tokens=linearize(hi),
            hf_tokens=linearize(hf),
            hi_bracketed=bracketed(hi),
            hf_bracketed=bracketed(hf),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="number of pairs to generate")
    parser.add_argument("--max-depth", type=int, default=2, help="max nested CP embeddings")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="do not deduplicate generated sentences",
    )
    parser.add_argument(
        "--format",
        choices=["text", "jsonl"],
        default="text",
        help="text: human-readable; jsonl: one JSON object per line",
    )
    parser.add_argument("--bracketed", action="store_true", help="also print bracketed parses (text mode)")
    args = parser.parse_args()

    pairs = generate_pairs(
        args.n, max_depth=args.max_depth, seed=args.seed, unique=not args.allow_duplicates
    )

    for i, pair in enumerate(pairs):
        if args.format == "jsonl":
            print(json.dumps(pair.to_dict()))
        else:
            print(f"[{i}] HI: {pair.hi}")
            print(f"    HF: {pair.hf}")
            if args.bracketed:
                print(f"    HI: {pair.hi_bracketed}")
                print(f"    HF: {pair.hf_bracketed}")


if __name__ == "__main__":
    main()

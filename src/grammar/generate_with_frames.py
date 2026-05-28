"""Frame-based generation of Head-Initial (HI) / Head-Final (HF) sentence pairs.

The grammar (see ``GRAMMAR.md``) shares its nonterminal expansions between the
HI and HF variants; the two differ only in *head placement*. Generation is two
layers:

1. **Frames** — parse-tree skeletons with POS slots at the leaves and no words.
   :func:`enumerate_frames` exhaustively yields every frame up to a CP-nesting
   depth (90 frames at depth 3, independent of lexicon size).

2. **Sampler** — :func:`sample_pair_from_frame` fills a frame's slots with
   words from :data:`cfg_vocab.VOCAB`, respecting a clause-scoped no-repeat
   constraint on ``NP_singular``, then renders the result as a HI/HF pair.

The CFG rules:

HI rules                  HF rules
--------                  --------
S  -> DP VP               S  -> DP VP
DP -> NP_proper           DP -> NP_proper          (bare [DP X])
DP -> D NP_singular       DP -> D NP_singular
VP -> V_dp DP             VP -> DP V_dp
VP -> V_cp CP             VP -> CP V_cp
VP -> V_intrans           VP -> V_intrans
CP -> C S                 CP -> S C

Running this module as a script prints the frame skeletons grouped by exact
CP-nesting depth, useful for sanity-checking the enumeration before sampling:

    uv run python -m grammar.generate_with_frames --max-depth 3
    uv run python -m grammar.generate_with_frames --max-depth 3 --coarse
"""

from __future__ import annotations

import argparse
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Iterator, Union

from grammar.cfg_vocab import VOCAB


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


# --- Filled layer: Constituent tree that renders to HI or HF ----------------
class Constituent(ABC):
    """Anything that can be rendered into a :class:`Node` in a head direction."""

    @abstractmethod
    def render(self, head_initial: bool) -> Node:
        """Render this constituent (and its descendants) head-initial or -final."""


class Terminal(Constituent):
    """A leaf carrying a part-of-speech ``label`` (e.g. ``V``) and its ``word``."""

    def __init__(self, label: str, word: str) -> None:
        self.label = label
        self.word = word

    def render(self, head_initial: bool) -> Node:
        return Node(self.label, word=self.word)


class Phrase(Constituent):
    """A binary phrase: ``head`` and ``complement``. HI puts the head first;
    HF puts it last. Daughters render in the global head direction so the tree
    stays internally consistent.
    """

    label: ClassVar[str] = "Phrase"

    def __init__(self, head: Constituent, complement: Constituent) -> None:
        self.head = head
        self.complement = complement

    def get_head_initial(self, head_initial: bool = True) -> Node:
        return Node(self.label, [self.head.render(head_initial), self.complement.render(head_initial)])

    def get_head_final(self) -> Node:
        return Node(self.label, [self.complement.render(False), self.head.render(False)])

    def render(self, head_initial: bool) -> Node:
        return self.get_head_initial() if head_initial else self.get_head_final()


class _Invariant(Phrase):
    """Phrase whose daughter order is fixed across head directions."""

    def get_head_final(self) -> Node:
        return self.get_head_initial(head_initial=False)


class DP(_Invariant):
    """DP -> D NP_singular (order identical in HI and HF)."""

    label = "DP"

    def __init__(self, det: str, noun: str) -> None:
        super().__init__(Terminal("D", det), Terminal("NP", noun))


class ProperDP(Constituent):
    """DP -> NP_proper. Renders as bare ``[DP John]`` per GRAMMAR.md."""

    label = "DP"

    def __init__(self, proper: str) -> None:
        self.proper = proper

    def render(self, head_initial: bool) -> Node:
        return Node(self.label, word=self.proper)


DPLike = Union[DP, ProperDP]


class S(_Invariant):
    """S -> DP VP (subject then predicate; order identical in HI and HF)."""

    label = "S"

    def __init__(self, subject: DPLike, predicate: "VP") -> None:
        super().__init__(subject, predicate)


class VP(Phrase):
    """VP has three forms:

    * ``VP -> V_dp DP``     (HI) / ``DP V_dp``  (HF)
    * ``VP -> V_cp CP``     (HI) / ``CP V_cp``  (HF)
    * ``VP -> V_intrans``   (no complement; HI and HF identical)
    """

    label = "VP"

    def __init__(self, verb: str, complement: "DPLike | CP | None" = None) -> None:
        head = Terminal("V", verb)
        if complement is None:
            self.head = head
            self.complement = None  # type: ignore[assignment]
        else:
            super().__init__(head, complement)

    def get_head_initial(self, head_initial: bool = True) -> Node:
        if self.complement is None:
            return Node(self.label, [self.head.render(head_initial)])
        return super().get_head_initial(head_initial)

    def get_head_final(self) -> Node:
        if self.complement is None:
            return Node(self.label, [self.head.render(False)])
        return super().get_head_final()


class CP(Phrase):
    """CP -> C S (HI) / S C (HF)."""

    label = "CP"

    def __init__(self, comp: str, clause: S) -> None:
        super().__init__(Terminal("C", comp), clause)


# --- Frame layer: parse-tree skeletons (POS slots, no words) -----------------
class Frame(ABC):
    """A skeletal derivation: parse tree with POS slots at the leaves, no words.

    Subclasses parallel the :class:`Constituent` classes one-to-one with the
    CFG rules. Once the slots are filled (via :meth:`to_constituent`), the
    result is an ordinary :class:`Constituent` tree that the HI/HF rendering
    machinery handles.
    """

    @abstractmethod
    def bracketed_skeleton(self, fine: bool = True) -> str:
        """Render as a bracketed POS skeleton.

        ``fine=True``  -> fine slot labels (``V_dp``, ``NP_singular``, etc.).
        ``fine=False`` -> surface labels (``V``, ``NP``); matches the structure
        the filled bracketed parse will have.
        """

    @abstractmethod
    def to_constituent(
        self, rng: random.Random, constraints: "LexicalConstraints"
    ) -> Constituent:
        """Fill every slot and return a Constituent ready for HI/HF rendering."""

    @abstractmethod
    def depth(self) -> int:
        """Maximum CP-nesting depth in this frame (0 if no CPs)."""


class FrameDPCommon(Frame):
    """DP -> D NP_singular."""

    def bracketed_skeleton(self, fine: bool = True) -> str:
        n = "NP_singular" if fine else "NP"
        return f"[DP [D _] [{n} _]]"

    def to_constituent(self, rng, constraints) -> DP:
        det = constraints.pick("D", rng)
        noun = constraints.pick("NP_singular", rng)
        return DP(det, noun)

    def depth(self) -> int:
        return 0


class FrameDPProper(Frame):
    """DP -> NP_proper. Renders as bare [DP _] in surface mode."""

    def bracketed_skeleton(self, fine: bool = True) -> str:
        return "[DP [NP_proper _]]" if fine else "[DP _]"

    def to_constituent(self, rng, constraints) -> ProperDP:
        return ProperDP(constraints.pick("NP_proper", rng))

    def depth(self) -> int:
        return 0


class FrameVPIntrans(Frame):
    """VP -> V_intrans."""

    def bracketed_skeleton(self, fine: bool = True) -> str:
        v = "V_intrans" if fine else "V"
        return f"[VP [{v} _]]"

    def to_constituent(self, rng, constraints) -> VP:
        return VP(constraints.pick("V_intrans", rng))

    def depth(self) -> int:
        return 0


class FrameVPdp(Frame):
    """VP -> V_dp DP."""

    def __init__(self, dp: Frame) -> None:
        self.dp = dp

    def bracketed_skeleton(self, fine: bool = True) -> str:
        v = "V_dp" if fine else "V"
        return f"[VP [{v} _] {self.dp.bracketed_skeleton(fine)}]"

    def to_constituent(self, rng, constraints) -> VP:
        verb = constraints.pick("V_dp", rng)
        dp = self.dp.to_constituent(rng, constraints)
        return VP(verb, dp)

    def depth(self) -> int:
        return 0


class FrameVPcp(Frame):
    """VP -> V_cp CP."""

    def __init__(self, cp: "FrameCP") -> None:
        self.cp = cp

    def bracketed_skeleton(self, fine: bool = True) -> str:
        v = "V_cp" if fine else "V"
        return f"[VP [{v} _] {self.cp.bracketed_skeleton(fine)}]"

    def to_constituent(self, rng, constraints) -> VP:
        verb = constraints.pick("V_cp", rng)
        cp = self.cp.to_constituent(rng, constraints)
        return VP(verb, cp)

    def depth(self) -> int:
        return self.cp.depth()


class FrameCP(Frame):
    """CP -> C S. Pushes a fresh clause scope before filling the embedded S."""

    def __init__(self, s: "FrameS") -> None:
        self.s = s

    def bracketed_skeleton(self, fine: bool = True) -> str:
        return f"[CP [C _] {self.s.bracketed_skeleton(fine)}]"

    def to_constituent(self, rng, constraints) -> CP:
        comp = constraints.pick("C", rng)
        constraints.push_clause()
        try:
            clause = self.s.to_constituent(rng, constraints)
        finally:
            constraints.pop_clause()
        return CP(comp, clause)

    def depth(self) -> int:
        return 1 + self.s.depth()


class FrameS(Frame):
    """S -> DP VP."""

    def __init__(self, dp: Frame, vp: Frame) -> None:
        self.dp = dp
        self.vp = vp

    def bracketed_skeleton(self, fine: bool = True) -> str:
        return f"[S {self.dp.bracketed_skeleton(fine)} {self.vp.bracketed_skeleton(fine)}]"

    def to_constituent(self, rng, constraints) -> S:
        subj = self.dp.to_constituent(rng, constraints)
        pred = self.vp.to_constituent(rng, constraints)
        return S(subj, pred)

    def depth(self) -> int:
        # DP has no depth, so the clause's depth is whatever the VP carries.
        return self.vp.depth()


# --- Frame enumeration -------------------------------------------------------
def _enum_frame_dp() -> Iterator[Frame]:
    yield FrameDPCommon()
    yield FrameDPProper()


def _enum_frame_vp(max_depth: int) -> Iterator[Frame]:
    yield FrameVPIntrans()
    for dp in _enum_frame_dp():
        yield FrameVPdp(dp)
    if max_depth >= 1:
        for s in _enum_frame_s(max_depth - 1):
            yield FrameVPcp(FrameCP(s))


def _enum_frame_s(max_depth: int) -> Iterator[FrameS]:
    for dp in _enum_frame_dp():
        for vp in _enum_frame_vp(max_depth):
            yield FrameS(dp, vp)


def enumerate_frames(max_depth: int) -> Iterator[FrameS]:
    """Yield every S frame with CP-nesting depth <= max_depth (cumulative)."""
    return _enum_frame_s(max_depth)


def count_frames(max_depth: int) -> int:
    """Closed-form count of S frames cumulative up to ``max_depth``.

    Recurrence:  N(d) = 2 * (3 + N(d-1)),   N(-1) = 0.
    Yields       N(0..3) = 6, 18, 42, 90.
    """
    n = 0
    for _ in range(max_depth + 1):
        n = 2 * (3 + n)
    return n


# --- Lexical sampler ---------------------------------------------------------
class LexicalConstraints:
    """Clause-scoped no-repeat for ``NP_singular`` only.

    Entering a new clause via :class:`FrameCP` pushes a fresh used-set;
    exiting pops it. Other categories sample freely. If a clause exhausts its
    ``NP_singular`` pool (would need a 5th distinct noun but only 4 exist),
    the constraint relaxes for that pick — should not happen in practice
    since each clause has at most 2 DPs.
    """

    def __init__(self) -> None:
        self._used_np_singular: list[set[str]] = [set()]

    def push_clause(self) -> None:
        self._used_np_singular.append(set())

    def pop_clause(self) -> None:
        self._used_np_singular.pop()

    def pick(self, pos: str, rng: random.Random) -> str:
        pool = VOCAB[pos]
        if pos == "NP_singular":
            available = [w for w in pool if w not in self._used_np_singular[-1]]
            if not available:
                available = pool
            word = rng.choice(available)
            self._used_np_singular[-1].add(word)
            return word
        return rng.choice(pool)


def sample_from_frame(
    frame: Frame,
    rng: random.Random,
    constraints: "LexicalConstraints | None" = None,
) -> Constituent:
    """Fill every slot in a frame and return a fully-lexicalized Constituent."""
    if constraints is None:
        constraints = LexicalConstraints()
    return frame.to_constituent(rng, constraints)


# --- Frame-based sentence pairs ----------------------------------------------
@dataclass
class FrameSentencePair:
    """A single derivation rendered as HI/HF, tagged with its source frame."""

    hi_tokens: list[str]
    hf_tokens: list[str]
    hi_bracketed: str
    hf_bracketed: str
    frame_id: str
    depth: int

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
            "frame_id": self.frame_id,
            "depth": self.depth,
        }


def sample_pair_from_frame(frame: FrameS, rng: random.Random) -> FrameSentencePair:
    """Fill the frame once and render HI and HF."""
    constituent = sample_from_frame(frame, rng)
    hi_tree = constituent.get_head_initial()
    hf_tree = constituent.get_head_final()
    return FrameSentencePair(
        hi_tokens=linearize(hi_tree),
        hf_tokens=linearize(hf_tree),
        hi_bracketed=bracketed(hi_tree),
        hf_bracketed=bracketed(hf_tree),
        frame_id=frame.bracketed_skeleton(fine=False),
        depth=frame.depth(),
    )


def sample_pairs_from_frame(
    frame: FrameS, k: int, rng: random.Random
) -> list[FrameSentencePair]:
    """Up to ``k`` unique lexicalizations of ``frame`` (deduped by HI string).

    May return fewer than ``k`` rows when the frame has fewer unique fillings
    under the constraint policy (e.g. ``S -> DP_proper VP_intrans`` has only
    ``4 * 3 = 12`` distinct fillings).
    """
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


# --- CLI: frame viewer -------------------------------------------------------
def _print_frames(max_depth: int, fine: bool) -> None:
    by_depth: dict[int, list[FrameS]] = {}
    for f in enumerate_frames(max_depth):
        by_depth.setdefault(f.depth(), []).append(f)
    cumulative = 0
    for d in sorted(by_depth):
        frames = by_depth[d]
        cumulative += len(frames)
        print(f"\n=== Exact depth {d}: {len(frames)} frames "
              f"(cumulative: {cumulative}) ===")
        for i, frame in enumerate(frames):
            print(f"  [{d}.{i:02d}] {frame.bracketed_skeleton(fine=fine)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-depth", type=int, default=2, help="max CP-nesting depth to enumerate")
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="show surface POS labels (V, NP) instead of fine (V_dp, NP_singular)",
    )
    args = parser.parse_args()
    _print_frames(args.max_depth, fine=not args.coarse)


if __name__ == "__main__":
    main()

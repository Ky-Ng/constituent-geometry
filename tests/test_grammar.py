"""Tests for the toy CFG: filled-layer constituents AND frame layer."""

import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grammar.generate_with_frames import (
    CP,
    DP,
    Frame,
    FrameCP,
    FrameDPCommon,
    FrameDPProper,
    FrameS,
    FrameVPIntrans,
    FrameVPcp,
    FrameVPdp,
    LexicalConstraints,
    ProperDP,
    S,
    Slot,
    VP,
    bracketed,
    count_frames,
    enumerate_frames,
    linearize,
    sample_pair_from_frame,
    sample_pairs_from_frame,
)


# =============================================================================
# Filled-layer: HI/HF rendering of hand-built derivations
# =============================================================================

def test_transitive_clause_renders_per_grammar_example():
    """Mirror GRAMMAR.md example 1: [DP [D the] [NP dog]] etc."""
    derivation = S(DP("the", "dog"), VP("likes", DP("a", "cat")))
    hi = derivation.get_head_initial()
    hf = derivation.get_head_final()
    assert bracketed(hi) == "[S [DP [D the] [NP dog]] [VP [V likes] [DP [D a] [NP cat]]]]"
    assert bracketed(hf) == "[S [DP [D the] [NP dog]] [VP [DP [D a] [NP cat]] [V likes]]]"
    assert linearize(hi) == ["the", "dog", "likes", "a", "cat"]
    assert linearize(hf) == ["the", "dog", "a", "cat", "likes"]


def test_proper_dp_renders_bare():
    """DP -> NP_proper produces [DP John], not [DP [NP John]]."""
    derivation = S(ProperDP("John"), VP("swims"))
    hi = derivation.get_head_initial()
    assert bracketed(hi) == "[S [DP John] [VP [V swims]]]"
    assert linearize(hi) == ["John", "swims"]


def test_intransitive_hi_hf_identical():
    """VP -> V_intrans has no complement to flip, so HI and HF agree."""
    derivation = S(DP("the", "cat"), VP("dances"))
    assert linearize(derivation.get_head_initial()) == linearize(derivation.get_head_final())


def test_embedded_clause_head_placement():
    """CP -> C S (HI) vs S C (HF); verb stays at its phrase's head edge."""
    inner = S(DP("a", "cat"), VP("likes", DP("the", "dog")))
    derivation = S(DP("the", "girl"), VP("believes", CP("that", inner)))
    hi = linearize(derivation.get_head_initial())
    hf = linearize(derivation.get_head_final())
    assert hi == ["the", "girl", "believes", "that", "a", "cat", "likes", "the", "dog"]
    assert hf == ["the", "girl", "a", "cat", "the", "dog", "likes", "that", "believes"]


# =============================================================================
# Frame layer: enumeration counts
# =============================================================================

@pytest.mark.parametrize("d, expected", [(0, 6), (1, 18), (2, 42), (3, 90)])
def test_count_frames_matches_closed_form(d, expected):
    assert count_frames(d) == expected


@pytest.mark.parametrize("d, expected", [(0, 6), (1, 18), (2, 42), (3, 90)])
def test_enumerate_frames_yields_count_frames(d, expected):
    assert sum(1 for _ in enumerate_frames(d)) == expected


def test_per_exact_depth_breakdown_is_6_12_24_48():
    """At exact depth d the count is 3 * 2^(d+1): 6, 12, 24, 48."""
    by_depth: dict[int, int] = {}
    for f in enumerate_frames(3):
        by_depth[f.depth()] = by_depth.get(f.depth(), 0) + 1
    assert by_depth == {0: 6, 1: 12, 2: 24, 3: 48}


def test_frame_ids_are_unique():
    ids = [f.bracketed_skeleton(fine=False) for f in enumerate_frames(3)]
    assert len(ids) == len(set(ids))


# =============================================================================
# Frame layer: bracketed skeleton matches filled bracket structure
# =============================================================================

def _strip_words(bracketed_filled: str) -> str:
    """Replace every word at a leaf '[Label word]' with '_'."""
    return re.sub(r"\[([A-Za-z_]+) ([^\[\]\s]+)\]", r"[\1 _]", bracketed_filled)


def test_filled_bracket_strips_to_frame_skeleton_surface():
    rng = random.Random(0)
    for frame in enumerate_frames(2):
        pair = sample_pair_from_frame(frame, rng)
        assert _strip_words(pair.hi_bracketed) == frame.bracketed_skeleton(fine=False)


# =============================================================================
# Frame layer: sampler respects clause-scoped NP_singular constraint
# =============================================================================

def _np_singular_per_clause(filled_bracket: str) -> list[list[str]]:
    """Return one list of NP_singular nouns per clause-scope. The split on
    ``[CP`` is a heuristic that works for this grammar because (a) each
    clause has at most one CP daughter and (b) all NP_singular leaves of a
    clause appear before the start of its embedded CP (if any)."""
    return [re.findall(r"\[NP (\w+)\]", part) for part in filled_bracket.split("[CP")]


def test_no_repeated_np_singular_within_a_clause():
    rng = random.Random(0)
    for frame in enumerate_frames(3):
        for _ in range(20):
            pair = sample_pair_from_frame(frame, rng)
            for clause_nouns in _np_singular_per_clause(pair.hi_bracketed):
                assert len(clause_nouns) == len(set(clause_nouns)), (
                    f"clause has repeated NP_singular: {clause_nouns} "
                    f"in {pair.hi_bracketed}"
                )


# =============================================================================
# Frame layer: sample_pairs_from_frame caps at K or at unique fillings
# =============================================================================

def test_k_cap_respected_when_few_unique_fillings_exist():
    """S -> DP_proper VP_intrans has 4 * 3 = 12 unique fillings."""
    frame = FrameS(FrameDPProper(), FrameVPIntrans())
    pairs = sample_pairs_from_frame(frame, k=1000, rng=random.Random(0))
    assert len(pairs) == 12
    assert len({p.hi for p in pairs}) == 12


def test_k_cap_respected_when_many_unique_fillings_exist():
    """A frame with many unique fillings stops at K."""
    frame = FrameS(FrameDPCommon(), FrameVPdp(FrameDPCommon()))
    pairs = sample_pairs_from_frame(frame, k=20, rng=random.Random(0))
    assert len(pairs) == 20
    assert len({p.hi for p in pairs}) == 20


def test_sampler_is_seed_deterministic():
    frame = next(enumerate_frames(2))
    a = [p.hi for p in sample_pairs_from_frame(frame, k=10, rng=random.Random(7))]
    b = [p.hi for p in sample_pairs_from_frame(frame, k=10, rng=random.Random(7))]
    assert a == b


# =============================================================================
# Frame layer: HI/HF property preserved on sampled rows
# =============================================================================

def test_sampled_hi_and_hf_are_token_multiset_equal():
    rng = random.Random(0)
    for frame in enumerate_frames(2):
        pair = sample_pair_from_frame(frame, rng)
        assert sorted(pair.hi_tokens) == sorted(pair.hf_tokens)

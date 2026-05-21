import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grammar.generate import (
    CP,
    DETERMINERS,
    DP,
    NOUNS,
    PP,
    PREPOSITIONS,
    S,
    VERBS,
    VP,
    bracketed,
    generate_pairs,
    linearize,
)


def test_dp_and_s_order_is_identical_across_variants():
    """DP -> D NP and S -> DP VP keep the same order in HI and HF."""
    derivation = S(DP("the", "dog"), VP("likes", DP("a", "cat")))
    hi = derivation.get_head_initial()
    hf = derivation.get_head_final()
    # Subject DP linearizes identically in both directions.
    assert linearize(hi)[:2] == ["the", "dog"]
    assert linearize(hf)[:2] == ["the", "dog"]


def test_transitive_clause_matches_grammar_examples():
    """Mirror GRAMMAR.md example 1 structure (with declared terminals)."""
    derivation = S(DP("the", "dog"), VP("likes", DP("a", "cat")))
    hi = derivation.get_head_initial()
    hf = derivation.get_head_final()
    assert bracketed(hi) == "[S [DP [D the] [NP dog]] [VP [V likes] [DP [D a] [NP cat]]]]"
    assert bracketed(hf) == "[S [DP [D the] [NP dog]] [VP [DP [D a] [NP cat]] [V likes]]]"
    assert linearize(hi) == ["the", "dog", "likes", "a", "cat"]
    assert linearize(hf) == ["the", "dog", "a", "cat", "likes"]


def test_pp_complement_head_placement():
    """PP -> P DP (HI) vs DP P (HF)."""
    derivation = S(DP("the", "boy"), VP("believes", PP("at", DP("a", "girl"))))
    assert linearize(derivation.get_head_initial()) == ["the", "boy", "believes", "at", "a", "girl"]
    assert linearize(derivation.get_head_final()) == ["the", "boy", "a", "girl", "at", "believes"]


def test_embedded_clause_head_placement():
    """CP -> C S (HI) vs S C (HF); verb stays at its phrase's head edge."""
    inner = S(DP("a", "cat"), VP("likes", DP("the", "dog")))
    derivation = S(DP("the", "girl"), VP("believes", CP("that", inner)))
    hi = linearize(derivation.get_head_initial())
    hf = linearize(derivation.get_head_final())
    assert hi == ["the", "girl", "believes", "that", "a", "cat", "likes", "the", "dog"]
    assert hf == ["the", "girl", "a", "cat", "the", "dog", "likes", "that", "believes"]


def test_pairs_use_only_declared_terminals():
    vocab = set(DETERMINERS + NOUNS + VERBS + ["that"] + PREPOSITIONS)
    for pair in generate_pairs(50, max_depth=3, seed=1):
        assert set(pair.hi_tokens) <= vocab
        assert set(pair.hf_tokens) <= vocab
        # HI and HF are reorderings of the same multiset of terminals.
        assert sorted(pair.hi_tokens) == sorted(pair.hf_tokens)


def test_generation_is_seed_deterministic():
    a = [p.hi for p in generate_pairs(20, seed=123)]
    b = [p.hi for p in generate_pairs(20, seed=123)]
    assert a == b


def test_max_depth_zero_forbids_embedding():
    for pair in generate_pairs(30, max_depth=0, seed=5):
        assert "that" not in pair.hi_tokens

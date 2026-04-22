import random

import pytest

from grammar import (
    D,
    DP,
    N,
    NP,
    PRETERMINALS,
    S,
    SPECIAL_TOKENS,
    TERMINALS,
    Tree,
    V_INTRANS,
    V_TRANS,
    VP,
    build_vocab,
    sample_tree,
)


def _all_terminals():
    return [t for cat in TERMINALS.values() for t in cat]


def _leaf(label: str, tok: str) -> Tree:
    return Tree(label=label, terminal=tok)


def _dp(d_tok: str, n_tok: str) -> Tree:
    return Tree(
        label=DP,
        children=[_leaf(D, d_tok), Tree(label=NP, children=[_leaf(N, n_tok)])],
        reorderable=True,
    )


class TestTreeLinearize:
    def test_preterminal_identical_in_both_orders(self):
        leaf = _leaf(D, "d1")
        assert leaf.linearize("HI") == ["d1"]
        assert leaf.linearize("HF") == ["d1"]

    def test_dp_reorders(self):
        dp = _dp("d1", "n1")
        assert dp.linearize("HI") == ["d1", "n1"]
        assert dp.linearize("HF") == ["n1", "d1"]

    def test_vp_trans_reorders(self):
        vp = Tree(
            label=VP,
            children=[_leaf(V_TRANS, "V_trans1"), _dp("d2", "n2")],
            reorderable=True,
        )
        assert vp.linearize("HI") == ["V_trans1", "d2", "n2"]
        assert vp.linearize("HF") == ["n2", "d2", "V_trans1"]

    def test_vp_intrans_not_reordered(self):
        vp = Tree(label=VP, children=[_leaf(V_INTRANS, "V_intrans1")], reorderable=False)
        assert vp.linearize("HI") == ["V_intrans1"]
        assert vp.linearize("HF") == ["V_intrans1"]

    def test_s_not_reordered(self):
        s = Tree(
            label=S,
            children=[_dp("d1", "n1"), Tree(label=VP, children=[_leaf(V_INTRANS, "V_intrans1")])],
            reorderable=False,
        )
        # S subject stays left in both
        assert s.linearize("HI")[0] == "d1"
        assert s.linearize("HF")[0] == "n1"  # (because DP under S reorders to NP D)
        # but whatever DP yields first, VP follows
        assert s.linearize("HI")[-1] == "V_intrans1"
        assert s.linearize("HF")[-1] == "V_intrans1"

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError):
            _dp("d1", "n1").linearize("WEIRD")  # type: ignore[arg-type]


class TestToNltkStr:
    def test_preterminal(self):
        assert _leaf(D, "d1").to_nltk_str("HI") == "(D d1)"

    def test_dp_hi_and_hf(self):
        dp = _dp("d1", "n1")
        assert dp.to_nltk_str("HI") == "(DP (D d1) (NP (N n1)))"
        assert dp.to_nltk_str("HF") == "(DP (NP (N n1)) (D d1))"

    def test_full_sentence_matches_planning_doc(self):
        # (S (DP d1 n1) (VP V_trans1 (DP d2 n2))) per the Output CSV example.
        s = Tree(
            label=S,
            children=[
                _dp("d1", "n1"),
                Tree(
                    label=VP,
                    children=[_leaf(V_TRANS, "V_trans1"), _dp("d2", "n2")],
                    reorderable=True,
                ),
            ],
            reorderable=False,
        )
        assert s.to_nltk_str("HI") == (
            "(S (DP (D d1) (NP (N n1))) "
            "(VP (V_trans V_trans1) (DP (D d2) (NP (N n2)))))"
        )
        assert s.to_nltk_str("HF") == (
            "(S (DP (NP (N n1)) (D d1)) "
            "(VP (DP (NP (N n2)) (D d2)) (V_trans V_trans1)))"
        )
        assert s.linearize("HI") == ["d1", "n1", "V_trans1", "d2", "n2"]
        assert s.linearize("HF") == ["n1", "d1", "n2", "d2", "V_trans1"]

    def test_parses_with_nltk(self):
        nltk = pytest.importorskip("nltk")
        s_tree = Tree(
            label=S,
            children=[
                _dp("d1", "n1"),
                Tree(
                    label=VP,
                    children=[_leaf(V_TRANS, "V_trans1"), _dp("d2", "n2")],
                    reorderable=True,
                ),
            ],
            reorderable=False,
        )
        for order in ("HI", "HF"):
            t = nltk.Tree.fromstring(s_tree.to_nltk_str(order))
            assert t.label() == "S"
            assert t.leaves() == s_tree.linearize(order)


class TestSampler:
    def test_root_is_S(self):
        rng = random.Random(0)
        assert sample_tree(rng).label == S

    def test_sampled_terminals_in_lexicon(self):
        rng = random.Random(0)
        lex = set(_all_terminals())
        for _ in range(200):
            for tok in sample_tree(rng).linearize("HI"):
                assert tok in lex

    def test_hi_hf_same_multiset(self):
        rng = random.Random(1)
        for _ in range(200):
            tree = sample_tree(rng)
            assert sorted(tree.linearize("HI")) == sorted(tree.linearize("HF"))

    def test_reproducibility(self):
        a = sample_tree(random.Random(42)).to_nltk_str("HI")
        b = sample_tree(random.Random(42)).to_nltk_str("HI")
        assert a == b

    def test_different_seeds_eventually_differ(self):
        # A weak but non-flaky check: 50 draws from seed 0 != 50 draws from seed 1.
        rng0, rng1 = random.Random(0), random.Random(1)
        a = [sample_tree(rng0).to_nltk_str("HI") for _ in range(50)]
        b = [sample_tree(rng1).to_nltk_str("HI") for _ in range(50)]
        assert a != b

    def test_wellformed(self):
        rng = random.Random(2)
        for _ in range(100):
            tree = sample_tree(rng)
            _assert_wellformed(tree)

    def test_weights_can_force_intransitive_vp(self):
        # Force VP -> V_intrans (the second production in PRODUCTIONS[VP]).
        rng = random.Random(3)
        for _ in range(20):
            tree = sample_tree(rng, weights={VP: [0.0, 1.0]})
            # Locate VP and check it's unary over V_intrans.
            vps = _find(tree, VP)
            for vp in vps:
                assert len(vp.children) == 1
                assert vp.children[0].label == V_INTRANS

    def test_weights_length_mismatch_raises(self):
        rng = random.Random(0)
        with pytest.raises(ValueError):
            sample_tree(rng, weights={VP: [1.0]})  # VP has 2 productions


class TestVocab:
    def test_includes_specials_and_terminals(self):
        tokens, stoi = build_vocab()
        for s in SPECIAL_TOKENS:
            assert s in stoi
        for t in _all_terminals():
            assert t in stoi

    def test_tokens_unique(self):
        tokens, _ = build_vocab()
        assert len(tokens) == len(set(tokens))

    def test_lexicon_size(self):
        # 5 D + 5 N + 4 V_trans + 4 V_intrans = 18
        assert len(_all_terminals()) == 18


def _assert_wellformed(t: Tree) -> None:
    if t.label in PRETERMINALS:
        assert t.terminal is not None
        assert t.terminal in TERMINALS[t.label]
        assert t.children == []
    else:
        assert t.terminal is None
        assert len(t.children) >= 1
        for c in t.children:
            _assert_wellformed(c)


def _find(t: Tree, label: str) -> list[Tree]:
    hits = [t] if t.label == label else []
    for c in t.children:
        hits.extend(_find(c, label))
    return hits

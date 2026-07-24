"""Unit tests for the decoder-only RASP-L HI -> HF translator.

Progress log (per README engineering step 1 -> 2):
  1. Lexical / phrase minimal pairs: intransitives, adverb flips, transitive
     object flips, adjectives (invariant), sentential complements.
  2. Relative clauses: subject-gap, object-gap, repeated words across clauses.
  3. Depth-2 recursion: the README's own example, GRAMMAR.md example 5, and an
     object-gap relative whose subject DP carries its own relative clause.
  4. Randomized property tests against grammar/generate_with_frames.py as the
     ground-truth oracle (view-only; loaded via sys.modules aliasing because it
     expects to live at grammar.v2.*).
  5. Faithfulness guards: every select() call is causal (runtime spy), the
     program's source touches no causal flag and no non-RASP numpy shortcuts,
     and teacher-forced predictions equal autoregressive rollout.
  6. The depth budget is real: a depth-2 sentence translated with depth=1
     fails, exposing the O(d) program-family wall (Zhou et al. 2023).

Run:  uv run pytest experiments/03_rasp_l_decoder_only/translation_unit_tests.py -q
"""
from __future__ import annotations

import importlib.util
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# --- oracle import: alias grammar/ files to the grammar.v2.* names they expect
_GRAMMAR_DIR = _HERE / "grammar"


def _load_as(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if "grammar" not in sys.modules:
    _pkg = types.ModuleType("grammar")
    _pkg.__path__ = [str(_GRAMMAR_DIR)]
    sys.modules["grammar"] = _pkg
if "grammar.v2" not in sys.modules:
    _pkg_v2 = types.ModuleType("grammar.v2")
    _pkg_v2.__path__ = [str(_GRAMMAR_DIR)]
    sys.modules["grammar.v2"] = _pkg_v2

_load_as("grammar.v2.cfg_vocab", _GRAMMAR_DIR / "cfg_vocab.py")
gen = _load_as("grammar.v2.generate_with_frames", _GRAMMAR_DIR / "generate_with_frames.py")

import rasp_core.core as rcore
import rasp_core.lib as rlib
import hi_hf_translation_raspl as prog
from hi_hf_translation_raspl import (
    BOS_ID, EOS_ID, SEP_ID, encode, next_token, predict_tokens, translate,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def teacher_forced_mismatches(hi: list[str], hf: list[str], depth: int = 2):
    """One causal pass over '<bos> hi <sep> hf <eos>'; compare next-token
    predictions on the HF side (positions sep .. len-2) with the targets."""
    seq = [BOS_ID] + encode(hi) + [SEP_ID] + encode(hf) + [EOS_ID]
    preds = predict_tokens(np.array(seq, dtype=int), depth=depth)
    sep_idx = seq.index(SEP_ID)
    bad = []
    for p in range(sep_idx, len(seq) - 1):
        if int(preds[p]) != seq[p + 1]:
            bad.append((p, prog.I2W[seq[p + 1]], prog.I2W[int(preds[p])]))
    return bad


def check_pair(hi: str, hf: str, depth: int = 2):
    hi_w, hf_w = hi.split(), hf.split()
    got = translate(hi_w, depth=depth)
    assert got == hf_w, f"rollout: {' '.join(got)!r} != {hf!r}"
    bad = teacher_forced_mismatches(hi_w, hf_w, depth=depth)
    assert not bad, f"teacher-forced mismatches {bad} for {hi!r}"


# ---------------------------------------------------------------------------
# 1. minimal phrase types (GRAMMAR.md examples 1-3 + adjunct/adjective pairs)
# ---------------------------------------------------------------------------

def test_intransitive_identity():
    check_pair("Mary swims", "Mary swims")


def test_transitive_object_flip():
    check_pair("John chases a cat", "John a cat chases")           # GRAMMAR.md ex1
    check_pair("Iskarous likes this dog", "Iskarous this dog likes")  # ex3


def test_adverb_flips():
    check_pair("Mary swims quickly", "Mary quickly swims")
    check_pair("John chases quickly a cat", "John a cat quickly chases")


def test_adjective_is_invariant_inside_dp():
    check_pair("John chases a happy cat", "John a happy cat chases")


def test_common_subject_dp():
    check_pair("the boy chases a cat", "the boy a cat chases")


# ---------------------------------------------------------------------------
# 2. clausal complements and relative clauses (depth 1)
# ---------------------------------------------------------------------------

def test_sentential_complement():                                   # GRAMMAR.md ex4
    check_pair(
        "Jia knows that Iskarous likes this dog",
        "Jia Iskarous this dog likes that knows",
    )


def test_sentential_complement_with_adverb():
    check_pair(
        "Jia knows happily that Mary swims",
        "Jia Mary swims that happily knows",
    )


def test_subject_gap_relative():
    check_pair(
        "the boy that swims chases a cat",
        "the swims that boy a cat chases",
    )


def test_object_gap_relative():
    check_pair(
        "the boy that Mary likes chases a cat",
        "the Mary likes that boy a cat chases",
    )


def test_repeated_noun_across_clauses():
    # 'dog' appears twice; addressing is positional, so repeats are harmless.
    check_pair(
        "the dog that chases a dog swims",
        "the a dog chases that dog swims",
    )


# ---------------------------------------------------------------------------
# 3. depth-2 recursion
# ---------------------------------------------------------------------------

def test_grammar_md_example_5():
    check_pair(
        "Jia knows that Iskarous likes that this dog dances",
        "Jia Iskarous this dog dances that likes that knows",
    )


def test_readme_depth2_example():
    check_pair(
        "Jia pursues the pitiful musician that knows happily a cat that sings eagerly",
        "Jia the a eagerly sings that cat happily knows that pitiful musician pursues",
    )


def test_object_gap_with_relativized_subject():
    check_pair(
        "the cat that the boy that swims likes dances",
        "the the swims that boy likes that cat dances",
    )


# ---------------------------------------------------------------------------
# 4. randomized property tests against the generator oracle
# ---------------------------------------------------------------------------

def _rand_nps(rng: random.Random, depth: int):
    cores = [gen.FrameNPsBar(), gen.FrameNPsAdj()]
    if depth <= 0 or rng.random() < 0.4:
        return rng.choice(cores)
    if rng.random() < 0.5:
        cp = gen.FrameCPrel(gen.FrameSsubjGap(_rand_vp(rng, depth - 1)))
    else:
        cp = gen.FrameCPrel(gen.FrameSobjGap(_rand_dp(rng, depth - 1), gen.FrameVPobjGap()))
    return gen.FrameNPsRel(rng.choice(cores), cp)


def _rand_dp(rng: random.Random, depth: int):
    if rng.random() < 0.3:
        return gen.FrameDPProper()
    return gen.FrameDPCommon(_rand_nps(rng, depth))


def _rand_vp(rng: random.Random, depth: int):
    roll = rng.random()
    if depth >= 1 and roll < 0.45:
        cp = gen.FrameCPsent(_rand_s(rng, depth - 1))
        return gen.FrameVPcpAdv(cp) if rng.random() < 0.3 else gen.FrameVPcp(cp)
    if roll < 0.75:
        dp = _rand_dp(rng, depth)
        return gen.FrameVPdpAdv(dp) if rng.random() < 0.3 else gen.FrameVPdp(dp)
    return gen.FrameVPIntransAdv() if rng.random() < 0.4 else gen.FrameVPIntrans()


def _rand_s(rng: random.Random, depth: int):
    return gen.FrameS(_rand_dp(rng, depth), _rand_vp(rng, depth))


def _property_check(frames, rng: random.Random):
    for frame in frames:
        pair = gen.sample_pair_from_frame(frame, rng)
        bad = teacher_forced_mismatches(pair.hi_tokens, pair.hf_tokens, depth=2)
        assert not bad, (
            f"mismatch {bad}\n  hi: {pair.hi}\n  hf: {pair.hf}\n  frame: {pair.frame_id}"
        )


def test_oracle_all_depth0_frames():
    rng = random.Random(0)
    frames = [f for f in gen.enumerate_frames(0)]
    assert len(frames) == 24          # 3 DP frames x 8 VP frames
    _property_check(frames, rng)


def test_oracle_random_depth1_frames():
    rng = random.Random(1)
    frames = []
    while len(frames) < 20:
        f = _rand_s(rng, 1)
        if f.depth() == 1:
            frames.append(f)
    _property_check(frames, rng)


def test_oracle_random_depth2_frames():
    rng = random.Random(2)
    frames = []
    while len(frames) < 15:
        f = _rand_s(rng, 2)
        if f.depth() == 2:
            frames.append(f)
    _property_check(frames, rng)


# ---------------------------------------------------------------------------
# 5. faithfulness guards
# ---------------------------------------------------------------------------

def test_every_select_is_causal():
    """Runtime spy: the program must never build a non-causal selector."""
    calls = {"n": 0}
    orig = rcore.select

    def spy(k, q, pred, causal=True):
        assert causal is True, "non-causal select() reached the library"
        calls["n"] += 1
        return orig(k, q, pred, causal=causal)

    rcore.select, rlib.select = spy, spy
    try:
        assert translate("John chases a cat".split()) == "John a cat chases".split()
    finally:
        rcore.select, rlib.select = orig, orig
    assert calls["n"] > 0, "spy never engaged; wiring is broken"


def test_source_purity():
    """The program imports cross-position machinery only from rasp_core and
    never bypasses it with numpy shortcuts or a causal flag."""
    src = Path(prog.__file__).read_text()
    assert "causal=False" not in src, "the causal flag must never be overridden"
    for banned in ("np.roll", "np.flip", "argsort", "np.cumsum", "np.take",
                   "[::-1]", ".sort(", "np.searchsorted", "np.repeat"):
        assert banned not in src, f"forbidden numpy shortcut: {banned}"
    import_lines = [l for l in src.splitlines() if l.startswith(("import ", "from "))]
    for line in import_lines:
        assert line.split()[1].split(".")[0] in {
            "annotations", "numpy", "rasp_core", "grammar", "__future__"
        }, f"unexpected import: {line}"


def test_rollout_equals_teacher_forcing_stepwise():
    hi = "Jia knows that Iskarous likes this dog".split()
    hf = "Jia Iskarous this dog likes that knows".split()
    seq = [BOS_ID] + encode(hi) + [SEP_ID]
    for target in encode(hf) + [EOS_ID]:
        pred = next_token(seq)
        assert pred == target
        seq.append(pred)


def test_eos_termination():
    hi = "Mary swims".split()
    out = translate(hi)
    assert out == ["Mary", "swims"]
    full = [BOS_ID] + encode(hi) + [SEP_ID] + encode(out)
    assert next_token(full) == EOS_ID


def test_lexicon_has_no_cross_class_collisions():
    classes = {}
    for pos, cls in (("D", "D"), ("N_singular", "N"), ("N_proper", "P"),
                     ("Adj", "J"), ("Adv", "R"), ("C", "C"), ("C_rel", "C"),
                     ("V_dp", "V"), ("V_cp", "V"), ("V_intrans", "V")):
        for w in gen.VOCAB[pos]:
            assert classes.setdefault(w, cls) == cls, f"{w} in two classes"


# ---------------------------------------------------------------------------
# 6. the depth budget is real (the length-generalization wall)
# ---------------------------------------------------------------------------

def test_depth_budget_wall():
    # The budget binds on SUBJECT-side (center-embedded) nesting, where the
    # span oracle must recurse.  Right-branching depth (the README example) is
    # resolved by last-child span inheritance and does not consume budget.
    hi = "the cat that the boy that swims likes dances".split()
    hf = "the the swims that boy likes that cat dances".split()
    assert translate(hi, depth=2) == hf
    assert translate(hi, depth=1) != hf, "depth budget 1 should fail on center-embedded depth 2"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

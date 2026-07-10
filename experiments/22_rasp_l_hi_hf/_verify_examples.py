"""Verify the worked examples for displacement_examples_handout against the oracle.

Prints, per sentence: HI tokens with indices, fine POS, per-token disp, hf_pos,
and the resulting HF string.  These numbers are copied verbatim into the LaTeX.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from grammar_oracle import oracle_features

EXAMPLES = {
    "rel-subj-intrans": "the boy that swims dances",
    "rel-subj-trans":   "the boy that chases the dog swims",
    "rel-obj":          "the dog that the boy chases swims",
    "adjective":        "John likes the attractive dog",
    "adverb":           "John chases quickly the dog",
    "determiner":       "the dog chases the cat",
}

for name, sent in EXAMPLES.items():
    hi = sent.split()
    f = oracle_features(hi)
    hf = " ".join(f["hf_tokens"])
    print("=" * 70)
    print(f"{name}:  HI = {sent!r}")
    print(f"           HF = {hf!r}")
    print(f"  {'i':>2} {'tok':<10} {'pos':<16} {'depth':>5} {'disp':>5} {'hf_pos':>6}")
    for i, tok in enumerate(hi):
        print(f"  {i:>2} {tok:<10} {f['pos'][i]:<16} {f['depth'][i]:>5} "
              f"{f['disp'][i]:>+5} {f['hf_pos'][i]:>6}")

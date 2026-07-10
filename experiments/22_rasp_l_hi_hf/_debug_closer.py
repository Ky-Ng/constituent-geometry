"""Tiny driver to run / step through _s2_closer.hi_to_hf on a sample prompt.

Plain run (prints HI -> HF):
    uv run python _debug_closer.py
    uv run python _debug_closer.py "John knows that Mary swims"

Under the debugger (break inside the program, inspect the s-ops):
    uv run python -m pdb _debug_closer.py "the boy that chases the dog swims"
      (Pdb) b _s2_closer.py:223     # the 'disp = ...' line: role/e/depth/d_* all exist here
      (Pdb) c                       # run to that line
      (Pdb) p list(depth)           # s-ops are numpy arrays -> wrap in list() to read them
      (Pdb) p list(e)
      (Pdb) p list(role)
      (Pdb) c                       # finish

Or set breakpoint() below and just: uv run python _debug_closer.py
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                    # _s2_closer, rasp_l, grammar_oracle
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))  # grammar.v2.cfg_vocab

from _s2_closer import hi_to_hf

prompt = sys.argv[1] if len(sys.argv) > 1 else "the boy that chases the dog swims"
tokens = prompt.split()

breakpoint()          # <-- uncomment: drops into pdb here; then `s` steps INTO hi_to_hf
out = hi_to_hf(tokens)
print("HI:", " ".join(tokens))
print("HF:", " ".join(out))

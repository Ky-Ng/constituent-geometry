"""A minimal, clean-room RASP-L interpreter (NumPy), integer-valued s-ops.

RASP-L is the length-generalizing, causal, decoder-only restriction of RASP
introduced by Zhou et al., "What Algorithms Can Transformers Learn? A Study in
Length Generalization" (ICLR 2024, https://arxiv.org/abs/2310.16028). This file
re-implements its small primitive set from the paper's description so this
experiment is self-contained; the semantics mirror Apple's reference
`np_rasp` (https://github.com/apple/ml-np-rasp). No Apple source is copied.

Model of computation
---------------------
An *s-op* is a length-n integer sequence (one value per token position). Every
function here is either:
  * elementwise (`tok_map`, `seq_map`)              -> an MLP / token-embedding,
  * an attention aggregate (`select`+`aggr`/`kqv`)  -> one attention head,
  * a width count (`sel_width`)                      -> one attention head.
A program is a straight-line composition of these, i.e. a transformer.

Length generalization (why RASP-L, not RASP): the paper's thesis is that a
transformer length-generalizes on a task iff the task has a *short* RASP-L
program whose `select` predicates use only order comparisons on values and
*relative* positions (never absolute indices). We keep to that discipline and
annotate any deviation. `select` is causal by default (a decoder only sees the
past); we pass `causal=False` only for reads over a region that, in the real
`... <sep> <output>` decoder layout, lies entirely in the past (see LOG.md).
"""
from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# base s-ops
# ---------------------------------------------------------------------------

def full(x, const):
    """Constant s-op with the same length as x."""
    return np.full(len(x), const, dtype=int)

def indices(x):
    """Position s-op [0, 1, ..., n-1]."""
    return np.arange(len(x), dtype=int)

def tok_map(x, func):
    """Elementwise unary map (one MLP)."""
    return np.array([int(func(xi)) for xi in x], dtype=int)

def seq_map(x, y, func):
    """Elementwise binary map (one MLP over two s-ops)."""
    return np.array([int(func(xi, yi)) for xi, yi in zip(x, y)], dtype=int)

# ---------------------------------------------------------------------------
# attention: select + aggregate
# ---------------------------------------------------------------------------

def select(k, q, pred, causal=True):
    """Boolean attention pattern A[qi, kj] = pred(k[kj], q[qi]).

    causal=True restricts to kj <= qi (a decoder attends only to the past).
    """
    n = len(k)
    A = np.zeros((n, n), dtype=bool)
    for qi in range(n):
        krange = range(qi + 1) if causal else range(n)
        for kj in krange:
            A[qi, kj] = bool(pred(k[kj], q[qi]))
    return A

def sel_width(A):
    """Number of selected keys per query (one head). Exact integer counting."""
    return A.sum(axis=1).astype(int)

def aggr(A, v, default=0, reduction="mean"):
    """Aggregate values v over the selected keys per query."""
    if reduction == "mean":
        num = A @ v.astype(float)
        w = sel_width(A)
        out = np.divide(num, w, out=np.full(len(v), default, dtype=float), where=(w != 0))
        return np.rint(out).astype(int)
    if reduction == "max":
        out = np.full(len(v), default, dtype=int)
        for i, row in enumerate(A):
            idx = np.flatnonzero(row)
            if len(idx):
                out[i] = int(v[idx].max())
        return out
    if reduction == "min":
        return -aggr(A, -v, default=-default, reduction="max")
    raise ValueError(reduction)

def kqv(k, q, v, pred, default=0, reduction="mean", causal=True):
    """select then aggregate — one attention head."""
    return aggr(select(k, q, pred, causal=causal), v, default=default, reduction=reduction)

# ---------------------------------------------------------------------------
# library helpers (all reducible to the primitives above)
# ---------------------------------------------------------------------------

def equals(k, q): return k == q
def leq(k, q):    return k <= q
def lt(k, q):     return k < q
def geq(k, q):    return k >= q
def gt(k, q):     return k > q

def shift_right(x, n, default=0):
    """x shifted right by n (value from position i-n). Relative-position read."""
    return kqv(indices(x) + n, indices(x), x, equals, default=default)

def read_rel(x, delta, default=0):
    """Read the value delta positions away: out[i] = x[i+delta] (default if OOB).

    A relative-position attention head (RASP-L-legal: predicate is on relative
    index only). Reading the future (delta > 0) is non-causal, so it is realized
    in the post-<sep> region where the read region lies in the past.
    """
    causal = delta <= 0
    return kqv(indices(x), indices(x), x, lambda kj, qi: kj == qi + delta,
               default=default, causal=causal)

def cumsum_incl(boolseq):
    """Inclusive prefix count of True (# of j <= i with boolseq[j])."""
    b = np.asarray(boolseq, dtype=int)
    return sel_width(select(b, b, lambda k, q: k, causal=True))

def where(cond, a, b):
    """Elementwise np.where via masking (one MLP)."""
    return np.array([int(ai) if ci else int(bi) for ci, ai, bi in zip(cond, a, b)], dtype=int)

def  index_select(x, idx, default=0, causal=True):
    """Gather: out[i] = x[idx[i]] (default if idx[i] out of causal range)."""
    return kqv(indices(x), idx, x, equals, default=default, causal=causal)

def firsts(x, queries, default=-1, causal=True):
    """Index of the first key j (<= i if causal) with x[j] == queries[i]."""
    return kqv(x, queries, indices(x), equals, default=default, reduction="min", causal=causal)

def lasts(x, queries, default=-1, causal=True):
    """Index of the last key j (<= i if causal) with x[j] == queries[i]."""
    return kqv(x, queries, indices(x), equals, default=default, reduction="max", causal=causal)

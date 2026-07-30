"""Decoder-only RASP-L program for Head-Initial -> Head-Final (Zapanese) translation.

Readable-name edition: identical logic to ``hi_hf_translation_raspl.py``, with
every variable renamed to be self-explanatory and a worked example in every
function docstring.

Contract (matches README.md):
  * Input is the token sequence  <bos> w1 .. wn <sep> v1 .. vk  (the HF prefix
    emitted so far; k may be 0).  The program is a next-token predictor: run it
    on the prefix, read the value at the LAST position, append, repeat.  It
    emits the HF permutation of w1..wn followed by <eos>.
  * Cross-position information flows ONLY through the unmodified vendored
    library `rasp_core/` (Apple ml-np-rasp `core.py` + `lib.py`).  Those
    primitives are causal by construction: `kqv` never exposes a non-causal
    selector and `index_select(x, idx)` yields its default whenever idx[i] > i.
    This file passes no `causal` flag anywhere because the API has none.
  * The hardcoded word -> id and word -> lexical-class tables are the token
    embedding (sanctioned by README: "hardcoded POS lexicons are allowed").
    The program reads nothing but the plain token sequence.

Why the computation lives at the generation frontier (the causality argument):
  A value stored at HI position i may only depend on tokens <= i.  But every
  interesting structural fact here — "where does the clause opened at this
  'that' end?", "how far does this verb's object extend?" — depends on tokens
  AFTER i ("John knows that Mary swims" vs "John knows that Mary chases the
  dog" agree on every prefix through "Mary", yet the spans differ).  So those
  facts simply cannot exist in HI residual streams.  They CAN exist at any
  position at or after <sep>, where the entire HI block is past context.  Every
  post-<sep> position therefore independently resolves its own output slot:

      output_slot     = index (1-based) of the HF token this position must
                        emit, computed as a count of positions since <sep>;
      source_position = HI position holding that token, found by walking the
                        constituent tree top-down from S, narrowing one child
                        per step;
      emit index_select(token_ids, source_position), or <eos> once
      output_slot == n+1.

  The walk needs one nontrivial span oracle: the extent of a SUBJECT DP (every
  other constituent is the last child of its parent, so its right edge is
  inherited for free).  That oracle is a recursion unrolled to a fixed clause-
  nesting budget MAX_CLAUSE_DEPTH — the O(d) "program family" the README
  allows.  A sentence nested deeper than the budget is resolved wrongly: that
  is the length-generalization wall of Zhou et al. (2023), made explicit.

Every retrieval is an `index_select` at a pointer that is unique by
construction, so every attention selector has width exactly 1 — no reliance on
`aggr_mean` averaging, and repeated words are harmless because addressing is
structural (pointer-valued), never by token identity.  Pointers are built only
from: a landmark found by content (`firsts` at <sep>), counts (`cumsum`), and
constant offsets — the sanctioned RASP-L idioms; raw `indices()` is never
touched by this file.

References:
  * Zhou et al., 2023. "What Algorithms Can Transformers Learn? A Study in
    Length Generalization." arXiv:2310.16028 (RASP-L, length generalization).
  * apple/ml-np-rasp — the vendored `rasp_core/core.py`, `rasp_core/lib.py`.
  * grammar/GRAMMAR.md and grammar/generate_with_frames.py — the v2 toy CFG
    this program inverts (view-only ground truth for the HI->HF mapping).
"""
from __future__ import annotations

import numpy as np

from rasp_core.core import full, seq_map, tok_map
from rasp_core.lib import cumsum, firsts, has_seen, index_select, where

from grammar.cfg_vocab import VOCAB, grammar_words

# ---------------------------------------------------------------------------
# Token table (the "embedding"): specials + every grammar word.
# cfg_vocab.py fixes <pad>=0 <bos>=1 <eos>=2 <unk>=3 but has no <sep>; the
# README format requires one, so it is appended as id 4 (deviation documented
# in the experiment notes).  Grammar words follow from id 5.
# ---------------------------------------------------------------------------
PAD, BOS, EOS, UNK, SEP = "<pad>", "<bos>", "<eos>", "<unk>", "<sep>"
SPECIALS = [PAD, BOS, EOS, UNK, SEP]

WORDS: list[str] = list(dict.fromkeys(grammar_words()))
WORD_TO_ID: dict[str, int] = {w: i for i, w in enumerate(SPECIALS + WORDS)}
ID_TO_WORD: dict[int, str] = {i: w for w, i in WORD_TO_ID.items()}

PAD_ID, BOS_ID, EOS_ID, UNK_ID, SEP_ID = (WORD_TO_ID[t] for t in SPECIALS)

# Lexical classes (the hardcoded POS lexicon).  A word's class is context-free;
# context-dependent role (V_dp vs V_cp, C vs C_rel) is recovered structurally.
#
# Example-docstring legend (one letter per position in the `cls` rows below):
#   -  CLS_OTHER          (specials: <bos> <sep> ...)
#   D  CLS_DETERMINER     (the, a, this)
#   P  CLS_PROPER_NOUN    (John, Mary, ...)
#   N  CLS_SINGULAR_NOUN  (dog, cat, ...)
#   A  CLS_ADJECTIVE      (happy, lazy, ...)
#   R  CLS_ADVERB         (quickly, ...)          [R for adveRb]
#   T  CLS_THAT           (that — both C and C_rel)
#   V  CLS_VERB           (swims, chases, knows, ...)
(
    CLS_OTHER,
    CLS_DETERMINER,
    CLS_PROPER_NOUN,
    CLS_SINGULAR_NOUN,
    CLS_ADJECTIVE,
    CLS_ADVERB,
    CLS_THAT,
    CLS_VERB,
) = range(8)

_SYNTACTIC_CLASS_OF_WORD: dict[str, int] = {}
for _pos_tag, _class in (
    ("D", CLS_DETERMINER), ("N_proper", CLS_PROPER_NOUN),
    ("N_singular", CLS_SINGULAR_NOUN),
    ("Adj", CLS_ADJECTIVE), ("Adv", CLS_ADVERB),
    ("C", CLS_THAT), ("C_rel", CLS_THAT),
    ("V_dp", CLS_VERB), ("V_cp", CLS_VERB), ("V_intrans", CLS_VERB),
):
    for _word in VOCAB[_pos_tag]:
        _SYNTACTIC_CLASS_OF_WORD.setdefault(_word, _class)


def _syntactic_class_of_id(token_id: int) -> int:
    """Lexical class of one token id (plain Python lookup, not a RASP op).

    Example:
        _syntactic_class_of_id(WORD_TO_ID["the"])   -> CLS_DETERMINER
        _syntactic_class_of_id(WORD_TO_ID["swims"]) -> CLS_VERB
        _syntactic_class_of_id(SEP_ID)              -> CLS_OTHER
    """
    return _SYNTACTIC_CLASS_OF_WORD.get(ID_TO_WORD.get(int(token_id), ""), CLS_OTHER)


# Node kinds carried through the top-down descent.
(
    NODE_SENTENCE,               # S       -> DP VP
    NODE_DP,                     # DP      -> NPROP | D [Adj] N [CP_rel]
    NODE_RELATIVE_CLAUSE,        # CP_rel  -> that + S_gap
    NODE_SENTENTIAL_COMPLEMENT,  # CP_sent -> that + S
    NODE_VERB_PHRASE,            # VP      -> V [Adv] [DP | CP_sent]
) = 1, 2, 3, 4, 5

MAX_CLAUSE_DEPTH = 2             # clause-nesting budget d (the dataset is d<=2)
DESCENT_STEPS_PER_CLAUSE_LEVEL = 5   # S -> DP -> CP_rel -> (DP|VP) -> ... per level
EXTRA_DESCENT_STEPS = 4


def _num_descent_steps(clause_depth: int) -> int:
    """Loop iterations needed to fully resolve any sentence of that depth.

    Example:
        _num_descent_steps(2) -> 5 * 3 + 4 = 19
    """
    return DESCENT_STEPS_PER_CLAUSE_LEVEL * (clause_depth + 1) + EXTRA_DESCENT_STEPS


# ---------------------------------------------------------------------------
# Span oracle: right edge of a constituent, given its start pointer.
# Evaluated at every sequence position in parallel; the pointer (and result)
# values are only meaningful at post-<sep> positions, where every read lands in
# the past (index_select is causal: a pointer into the future yields the
# default).  All docstring examples therefore read "evaluated at the last
# position".  `remaining_clause_budget` bounds clause nesting; the Python
# recursion below is the depth-d unrolling of the program family.
# ---------------------------------------------------------------------------

def _syntactic_class_at(class_per_pos: np.ndarray, position_ptr: np.ndarray) -> np.ndarray:
    """Lexical class at a pointer (width-1 causal retrieval; CLS_OTHER off the edges).

    Example (evaluated at the last position, where all reads are causal):
        tok          = <bos> the dog swims <sep>
        pos          =   0    1   2    3     4
        cls          =   -    D   N    V     -
        position_ptr = 2
        returns      -> N  (CLS_SINGULAR_NOUN, the class of "dog")
    """
    return index_select(class_per_pos, position_ptr, default=CLS_OTHER)


def _end_of_determiner_phrase(
    class_per_pos: np.ndarray, dp_start: np.ndarray, remaining_clause_budget: int
) -> np.ndarray:
    """Right edge of a DP starting at dp_start:  NPROP | D [Adj] N [CP_rel].

    Example 1 — proper noun (a DP is just the one name):
        tok      = <bos> John swims <sep>
        pos      =   0    1     2     3
        cls      =   -    P     V     -
        dp_start = 1
        returns  -> 1                       # "John" spans 1..1

    Example 2 — determiner + adjective + noun:
        tok      = <bos> the happy dog swims <sep>
        pos      =   0    1    2    3    4     5
        cls      =   -    D    A    N    V     -
        dp_start = 1
        returns  -> 3                       # "the happy dog" spans 1..3

    Example 3 — noun modified by a relative clause:
        tok      = <bos> the dog that chases the cat swims <sep>
        pos      =   0    1   2    3     4    5   6    7     8
        cls      =   -    D   N    T     V    D   N    V     -
        dp_start = 1
        returns  -> 6                       # "the dog that chases the cat"
    """
    class_after_start = _syntactic_class_at(class_per_pos, dp_start + 1)
    core_noun_end = where(
        class_after_start == CLS_ADJECTIVE, dp_start + 2, dp_start + 1
    )
    end = core_noun_end
    if remaining_clause_budget >= 1:
        has_relative_clause = (
            _syntactic_class_at(class_per_pos, core_noun_end + 1) == CLS_THAT
        )
        end = where(
            has_relative_clause,
            _end_of_relative_clause(
                class_per_pos, core_noun_end + 1, remaining_clause_budget
            ),
            core_noun_end,
        )
    return where(
        _syntactic_class_at(class_per_pos, dp_start) == CLS_PROPER_NOUN,
        dp_start,
        end,
    )


def _end_of_relative_clause(
    class_per_pos: np.ndarray, that_position: np.ndarray, remaining_clause_budget: int
) -> np.ndarray:
    """Right edge of a relative clause whose C_rel 'that' sits at that_position.

    Subject gap: that + VP.   Object gap: that + DP + V_dp (verb is last).
    The two are told apart by the class right after 'that' (VERB => subject gap).

    Example 1 — subject gap ("the dog that __ chases the cat"):
        tok           = <bos> the dog that chases the cat swims <sep>
        pos           =   0    1   2    3     4    5   6    7     8
        cls           =   -    D   N    T     V    D   N    V     -
        that_position = 3
        returns       -> 6                  # "that chases the cat"

    Example 2 — object gap ("the dog that the cat chases __"):
        tok           = <bos> the dog that the cat chases swims <sep>
        pos           =   0    1   2    3    4   5    6     7     8
        cls           =   -    D   N    T    D   N    V     V     -
        that_position = 3
        returns       -> 6                  # "that the cat chases"
    """
    is_subject_gap = (
        _syntactic_class_at(class_per_pos, that_position + 1) == CLS_VERB
    )
    return where(
        is_subject_gap,
        _end_of_verb_phrase(
            class_per_pos, that_position + 1, remaining_clause_budget - 1
        ),
        _end_of_determiner_phrase(
            class_per_pos, that_position + 1, remaining_clause_budget - 1
        ) + 1,
    )


def _end_of_sentential_complement(
    class_per_pos: np.ndarray, that_position: np.ndarray, remaining_clause_budget: int
) -> np.ndarray:
    """Right edge of a sentential complement: that + S, where S = DP VP (VP last).

    Example ("knows that John swims"):
        tok           = <bos> Mary knows that John swims <sep>
        pos           =   0    1     2    3     4     5     6
        cls           =   -    P     V    T     P     V     -
        that_position = 3
        returns       -> 5                  # "that John swims"
        (subject DP "John" ends at 4; the VP "swims" then ends at 5)
    """
    subject_dp_end = _end_of_determiner_phrase(
        class_per_pos, that_position + 1, remaining_clause_budget - 1
    )
    return _end_of_verb_phrase(
        class_per_pos, subject_dp_end + 1, remaining_clause_budget - 1
    )


def _end_of_verb_phrase(
    class_per_pos: np.ndarray, verb_position: np.ndarray, remaining_clause_budget: int
) -> np.ndarray:
    """Right edge of a VP headed at verb_position:  V [Adv] [DP | CP_sent].

    Example 1 — intransitive with adverb:
        tok           = <bos> the dog swims quickly <sep>
        pos           =   0    1   2    3      4      5
        cls           =   -    D   N    V      R      -
        verb_position = 3
        returns       -> 4                  # "swims quickly"

    Example 2 — transitive (DP complement):
        tok           = <bos> the dog chases the cat <sep>
        pos           =   0    1   2    3     4   5    6
        cls           =   -    D   N    V     D   N    -
        verb_position = 3
        returns       -> 5                  # "chases the cat"

    Example 3 — clause-taking (CP_sent complement):
        tok           = <bos> Mary knows that John swims <sep>
        pos           =   0    1     2    3     4     5     6
        cls           =   -    P     V    T     P     V     -
        verb_position = 2
        returns       -> 5                  # "knows that John swims"
    """
    has_adverb = (
        _syntactic_class_at(class_per_pos, verb_position + 1) == CLS_ADVERB
    )
    complement_start = where(has_adverb, verb_position + 2, verb_position + 1)
    complement_class = _syntactic_class_at(class_per_pos, complement_start)
    end = where(
        (complement_class == CLS_DETERMINER)
        | (complement_class == CLS_PROPER_NOUN),
        _end_of_determiner_phrase(
            class_per_pos, complement_start, remaining_clause_budget
        ),
        where(has_adverb, verb_position + 1, verb_position),  # bare/adv intransitive
    )
    if remaining_clause_budget >= 1:
        end = where(
            complement_class == CLS_THAT,
            _end_of_sentential_complement(
                class_per_pos, complement_start, remaining_clause_budget
            ),
            end,
        )
    return end


# ---------------------------------------------------------------------------
# The program: one causal pass over the whole prefix.
# ---------------------------------------------------------------------------

def _snapshot(x) -> np.ndarray:
    """Trace-snapshot helper (also narrows the vendored ops' loose types)."""
    return np.asarray(x).copy()


def predict_tokens(
    token_ids: np.ndarray,
    max_clause_depth: int = MAX_CLAUSE_DEPTH,
    trace: list | None = None,
) -> np.ndarray:
    """Next-token prediction at every position (meaningful from <sep> onward).

    The value at position p >= sep_position is the (p - sep_position + 1)-th HF
    token, or <eos> once the permutation is exhausted.  Values before <sep> are
    unspecified filler — an autoregressive driver never reads them.

    Example (HI "the dog chases the cat" -> HF "the dog the cat chases"):
        tok = <bos> the dog chases the cat <sep> the dog
        pos =   0    1   2    3     4   5    6    7   8
        cls =   -    D   N    V     D   N    -    D   N
        output at pos 6 -> id("the")    # 1st HF token (output_slot = 1)
        output at pos 7 -> id("dog")    # 2nd HF token
        output at pos 8 -> id("the")    # 3rd HF token — the driver appends
                                        # this and calls the program again
        full rollout continues: the dog the cat chases <eos>

    Pass a list as `trace` to capture per-step state snapshots (introspection
    for teaching material only; it has no effect on the computation).
    """
    token_ids = np.asarray(token_ids, dtype=int)
    class_per_pos = tok_map(token_ids, _syntactic_class_of_id)

    # --- segment geometry, as counts and one content landmark ---------------
    is_at_or_after_sep = has_seen(token_ids, full(token_ids, SEP_ID))
    output_slot = cumsum(is_at_or_after_sep)             # 1-based HF output slot
    is_head_initial_word = seq_map(
        class_per_pos != CLS_OTHER, is_at_or_after_sep,
        lambda is_word, past_sep: is_word and not past_sep,
    )
    num_hi_words = cumsum(is_head_initial_word)          # |HI| once past <sep>
    sep_position = firsts(token_ids, full(token_ids, SEP_ID), default=0)

    # --- descent state: node kind, span [span_start, span_end], slot within it
    # slot_in_span starts as the global output slot and is re-based every time
    # the walk narrows into a child constituent.
    slot_in_span = output_slot
    node_kind = full(token_ids, NODE_SENTENCE)
    span_start = full(token_ids, 1)                      # HI starts after <bos>
    span_end = sep_position - 1
    resolved = full(token_ids, 0) == 1
    source_position = full(token_ids, 0)

    if trace is not None:
        trace.append({
            "phase": "init",
            "class_per_pos": _snapshot(class_per_pos),
            "is_at_or_after_sep": _snapshot(is_at_or_after_sep),
            "output_slot": _snapshot(output_slot),
            "num_hi_words": _snapshot(num_hi_words),
            "sep_position": _snapshot(sep_position),
            "node_kind": _snapshot(node_kind),
            "span_start": _snapshot(span_start),
            "span_end": _snapshot(span_end),
            "slot_in_span": _snapshot(slot_in_span),
            "source_position": _snapshot(source_position),
            "resolved": _snapshot(resolved),
        })

    for _ in range(_num_descent_steps(max_clause_depth)):
        class_after_span_start = _syntactic_class_at(class_per_pos, span_start + 1)

        # NODE_SENTENCE: S -> subject-DP VP (order-invariant in HF); the only
        # place the span oracle is needed.  e.g. HI "the dog chases the cat":
        # the subject DP spans 1..2, so slots 1..2 narrow into the DP and
        # slots 3.. narrow into the VP with the slot re-based (slot - 2).
        subject_dp_end = _end_of_determiner_phrase(
            class_per_pos, span_start, max_clause_depth
        )
        subject_dp_length = subject_dp_end - span_start + 1
        slot_in_subject_dp = slot_in_span <= subject_dp_length
        s_case_node_kind = where(
            slot_in_subject_dp,
            full(token_ids, NODE_DP),
            full(token_ids, NODE_VERB_PHRASE),
        )
        s_case_span_start = where(slot_in_subject_dp, span_start, subject_dp_end + 1)
        s_case_span_end = where(slot_in_subject_dp, subject_dp_end, span_end)
        s_case_slot = where(
            slot_in_subject_dp, slot_in_span, slot_in_span - subject_dp_length
        )

        # NODE_DP: HI "D [Adj] N [CP_rel]" emits HF as "D  HF(CP_rel)  [Adj] N".
        # e.g. HI "the dog that swims" -> HF "the  swims that  dog":
        #   slot 1 -> the determiner (span_start); slots 2..1+rc_len -> narrow
        #   into the relative clause; later slots -> the [Adj] N core, copied
        #   directly from span_start + (slot - 1 - rc_len).
        # A proper-noun DP is a single token: slot 1 copies it directly.
        is_proper_noun = (
            _syntactic_class_at(class_per_pos, span_start) == CLS_PROPER_NOUN
        )
        core_noun_end = where(
            class_after_span_start == CLS_ADJECTIVE, span_start + 2, span_start + 1
        )
        relative_clause_length = span_end - core_noun_end   # 0 when no relative
        slot_is_determiner = (~is_proper_noun) & (slot_in_span == 1)
        slot_in_relative_clause = (
            (~is_proper_noun)
            & (slot_in_span > 1)
            & (slot_in_span <= 1 + relative_clause_length)
        )
        dp_case_resolved = (
            is_proper_noun | slot_is_determiner | ~slot_in_relative_clause
        )
        dp_case_source = where(
            is_proper_noun | slot_is_determiner,
            span_start,
            span_start + (slot_in_span - 1 - relative_clause_length),
        )
        dp_case_node_kind = full(token_ids, NODE_RELATIVE_CLAUSE)
        dp_case_span_start = core_noun_end + 1
        dp_case_span_end = span_end
        dp_case_slot = slot_in_span - 1

        # NODE_RELATIVE_CLAUSE: HI "that + S_gap" -> HF "HF(S_gap) + that".
        # Subject gap ("that chases the cat"): S_gap is just a VP.
        # Object gap ("that the cat chases"): S_gap = subject-DP + final verb.
        # e.g. HI "that the cat chases" -> HF "the cat chases that":
        #   the last slot -> "that" (span_start); for an object gap the verb
        #   (span_end) fills the slot right after the subject DP; earlier slots
        #   narrow into the subject DP at span [span_start+1, span_end-1].
        span_length = span_end - span_start + 1
        relclause_slot_is_that = slot_in_span == span_length
        is_subject_gap = class_after_span_start == CLS_VERB
        objgap_subject_dp_length = span_end - span_start - 1
        slot_in_objgap_subject_dp = (
            (~relclause_slot_is_that)
            & (~is_subject_gap)
            & (slot_in_span <= objgap_subject_dp_length)
        )
        slot_is_objgap_verb = (
            (~relclause_slot_is_that)
            & (~is_subject_gap)
            & (slot_in_span > objgap_subject_dp_length)
        )
        relclause_case_resolved = relclause_slot_is_that | slot_is_objgap_verb
        relclause_case_source = where(relclause_slot_is_that, span_start, span_end)
        relclause_case_node_kind = where(
            is_subject_gap,
            full(token_ids, NODE_VERB_PHRASE),
            full(token_ids, NODE_DP),
        )
        relclause_case_span_start = span_start + 1
        relclause_case_span_end = where(is_subject_gap, span_end, span_end - 1)
        relclause_case_slot = slot_in_span

        # NODE_SENTENTIAL_COMPLEMENT: HI "that + S" -> HF "HF(S) + that".
        # e.g. HI "that John swims" -> HF "John swims that": the last slot is
        # "that" (span_start); every other slot re-enters NODE_SENTENCE on the
        # embedded S at span [span_start+1, span_end].
        sentcomp_slot_is_that = slot_in_span == span_length
        sentcomp_case_resolved = sentcomp_slot_is_that
        sentcomp_case_source = span_start
        sentcomp_case_span_start = span_start + 1

        # NODE_VERB_PHRASE: HI "V [Adv] [complement]" -> HF
        # "HF(complement) [Adv] V".  e.g. HI "chases quickly the cat" -> HF
        # "the cat quickly chases":
        #   slots 1..complement_length narrow into the complement (a DP or a
        #   CP_sent starting at complement_start); the next slot is the adverb
        #   (span_start+1) if present; the final slot is the verb (span_start).
        has_adverb = (
            (span_start + 1 <= span_end) & (class_after_span_start == CLS_ADVERB)
        )
        complement_start = where(has_adverb, span_start + 2, span_start + 1)
        has_complement = complement_start <= span_end
        complement_class = _syntactic_class_at(class_per_pos, complement_start)
        complement_length = span_end - complement_start + 1
        slot_in_complement = has_complement & (slot_in_span <= complement_length)
        slot_is_adverb = has_adverb & (
            (has_complement & (slot_in_span == complement_length + 1))
            | (~has_complement & (slot_in_span == 1))
        )
        vp_case_resolved = ~slot_in_complement
        vp_case_source = where(slot_is_adverb, span_start + 1, span_start)
        vp_case_node_kind = where(
            complement_class == CLS_THAT,
            full(token_ids, NODE_SENTENTIAL_COMPLEMENT),
            full(token_ids, NODE_DP),
        )
        vp_case_span_start = complement_start
        vp_case_slot = slot_in_span

        # ---- merge the five cases by current node kind, freezing finished rows
        def merge_by_node_kind(if_sentence, if_dp, if_relclause, if_sentcomp, if_vp):
            out = where(node_kind == NODE_SENTENCE, if_sentence,
                  where(node_kind == NODE_DP, if_dp,
                  where(node_kind == NODE_RELATIVE_CLAUSE, if_relclause,
                  where(node_kind == NODE_SENTENTIAL_COMPLEMENT, if_sentcomp,
                        if_vp))))
            return out

        newly_resolved = merge_by_node_kind(
            full(token_ids, 0), dp_case_resolved, relclause_case_resolved,
            sentcomp_case_resolved, vp_case_resolved,
        ) == 1
        merged_source = merge_by_node_kind(
            full(token_ids, 0), dp_case_source, relclause_case_source,
            sentcomp_case_source, vp_case_source,
        )
        merged_node_kind = merge_by_node_kind(
            s_case_node_kind, dp_case_node_kind, relclause_case_node_kind,
            full(token_ids, NODE_SENTENCE), vp_case_node_kind,
        )
        merged_span_start = merge_by_node_kind(
            s_case_span_start, dp_case_span_start, relclause_case_span_start,
            sentcomp_case_span_start, vp_case_span_start,
        )
        merged_span_end = merge_by_node_kind(
            s_case_span_end, dp_case_span_end, relclause_case_span_end,
            span_end, span_end,
        )
        merged_slot = merge_by_node_kind(
            s_case_slot, dp_case_slot, relclause_case_slot,
            slot_in_span, vp_case_slot,
        )

        source_position = where(
            resolved, source_position,
            where(newly_resolved, merged_source, source_position),
        )
        node_kind = where(resolved, node_kind, merged_node_kind)
        span_start = where(resolved, span_start, merged_span_start)
        span_end = where(resolved, span_end, merged_span_end)
        next_slot = where(resolved, slot_in_span, merged_slot)
        resolved = resolved | newly_resolved
        slot_in_span = next_slot

        if trace is not None:
            trace.append({
                "phase": "step",
                "subject_dp_end": _snapshot(subject_dp_end),
                "node_kind": _snapshot(node_kind),
                "span_start": _snapshot(span_start),
                "span_end": _snapshot(span_end),
                "slot_in_span": _snapshot(slot_in_span),
                "source_position": _snapshot(source_position),
                "resolved": _snapshot(resolved),
            })

    # --- emit ----------------------------------------------------------------
    copied_token_ids = index_select(token_ids, source_position, default=UNK_ID)
    predicted_token_ids = where(
        output_slot == num_hi_words + 1, full(token_ids, EOS_ID), copied_token_ids
    )
    if trace is not None:
        trace.append({
            "phase": "emit",
            "copied_token_ids": _snapshot(copied_token_ids),
            "final": _snapshot(predicted_token_ids),
        })
    return predicted_token_ids


# ---------------------------------------------------------------------------
# Autoregressive driver and word-level convenience wrappers.
# ---------------------------------------------------------------------------

def encode(words: list[str]) -> list[int]:
    """Words -> token ids.

    Example:  encode(["the", "dog"]) -> [WORD_TO_ID["the"], WORD_TO_ID["dog"]]
    """
    return [WORD_TO_ID.get(w, UNK_ID) for w in words]


def decode(token_ids: list[int]) -> list[str]:
    """Token ids -> words (inverse of encode; unknown ids become <unk>).

    Example:  decode(encode(["the", "dog"])) -> ["the", "dog"]
    """
    return [ID_TO_WORD.get(int(i), UNK) for i in token_ids]


def next_token(prefix_token_ids: list[int], max_clause_depth: int = MAX_CLAUSE_DEPTH) -> int:
    """One decoder step: the model's prediction at the last position.

    Example:
        prefix  = <bos> the dog chases the cat <sep>
        returns -> WORD_TO_ID["the"]   (the 1st HF token)
    """
    return int(
        predict_tokens(
            np.array(prefix_token_ids, dtype=int), max_clause_depth=max_clause_depth
        )[-1]
    )


def translate(head_initial_words: list[str], max_clause_depth: int = MAX_CLAUSE_DEPTH) -> list[str]:
    """Greedy autoregressive rollout: <bos> hi <sep>  ->  hf ... <eos>.

    Example:
        translate(["the", "dog", "chases", "the", "cat"])
        -> ["the", "dog", "the", "cat", "chases"]
    """
    sequence_ids = [BOS_ID] + encode(head_initial_words) + [SEP_ID]
    output_ids: list[int] = []
    for _ in range(len(head_initial_words) + 1):
        predicted_id = next_token(sequence_ids, max_clause_depth=max_clause_depth)
        if predicted_id == EOS_ID:
            break
        output_ids.append(predicted_id)
        sequence_ids.append(predicted_id)
    return decode(output_ids)

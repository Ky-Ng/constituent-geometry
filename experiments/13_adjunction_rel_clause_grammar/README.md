# Experiment 13: Adjunction + Relative Clause Grammar (Grammar v2 Dataset)

## Overview

This experiment builds a new HI/HF dataset using an extended grammar (`src/grammar/v2/`) that adds relative clauses and adjunction — constructions absent from the original grammar (v1) used in experiments 05–12. It is a data-generation experiment (like 05), not a training experiment.

---

## Motivation

The v1 grammar (experiments 01–12) supports only sentential complement clauses (`CP_sent`) and flat DP/VP structures. Two linguistically important constructions are missing:

1. **Relative clauses (`CP_rel`)** — noun-modifying clauses with subject-gap and object-gap variants. These introduce a second source of CP-depth, enabling richer center-embedding and cross-serial dependency structures.
2. **Adjunction (AdjP, AdvP)** — adjectives modifying nouns and adverbs modifying VPs. These add non-recursive word-order alternations and are common in naturalistic data.

The grammar spec is pinned in `GRAMMAR.md` (this folder). The implementation is isolated in `src/grammar/v2/` so that v1 experiments remain reproducible and future grammar versions can be tracked by which `src/grammar/vN/` module they import.

---

## Grammar Changes (v1 → v2)

### Terminal taxonomy correction
| v1 POS key | v2 POS key | Phrase label |
|---|---|---|
| `NP_singular` | `N_singular` | `NP_singular` |
| `NP_proper` | `N_proper` | `NP_proper` |

In v1, `NP_singular` was both a phrase label and a POS key. In v2, `NP_singular` is a phrasal rule (`NP_singular -> N_singular`) and `N_singular` is the terminal POS.

### New rules (see GRAMMAR.md for full spec)
- `CP_rel -> C_rel S_subj_gap | C_rel S_obj_gap`  (HI) / `S_subj_gap C_rel | S_obj_gap C_rel` (HF)
- `S_subj_gap -> VP` (subject relativized out)
- `S_obj_gap -> DP VP_obj_gap` (object relativized out)
- `VP_obj_gap -> V_dp` (transitive verb, gap in object position)
- `NP_singular -> NP_singular CP_rel` (HI) / `CP_rel NP_singular` (HF)
- `NP_singular -> NP_singular_adj`; `NP_singular_adj -> AdjP N_singular`
- `VP -> VP_intrans_adv | VP_dp_adv DP | VP_cp_adv CP_sent`  (with adverb variants)
- `AdjP -> Adj`; `AdvP -> Adv`

Adjectives and adverbs are restricted to **depth-1 adjunction** (no stacked modifiers) to keep the frame count tractable.

### New vocabulary
- `C_rel`: relative clause complementizer(s) — see GRAMMAR.md
- `Adj`: ~70 adjectives
- `Adv`: 12 adverbs

---

## Architecture: `src/grammar/v2/`

```
src/grammar/v2/
├── __init__.py                       # public API (mirrors v1)
├── cfg_vocab_proposal.py             # → cfg_vocab.py
└── generate_with_frames_proposal.py  # → generate_with_frames.py
```

Consumers import via:
```python
from grammar.v2 import enumerate_frames, sample_pairs_from_frame, FrameSentencePair
```

v1 (`src/grammar/`) is **not modified**.

---

## Dataset Schema (new columns vs v1)

| Column | Type | Definition |
|---|---|---|
| `hi` | str | Head-initial surface string |
| `hf` | str | Head-final surface string |
| `hi_tokens` | list[str] | HI token list |
| `hf_tokens` | list[str] | HF token list |
| `hi_bracketed` | str | HI bracketed constituency parse |
| `hf_bracketed` | str | HF bracketed constituency parse |
| `hi_frame_tagged` | str | **NEW** HI skeleton with position tags (e.g. `[N_proper_1 [V_dp_2 N_proper_3]]`) |
| `hf_frame_tagged` | str | **NEW** HF skeleton with same position tags reordered |
| `frame_id` | str | Coarse-label bracketed skeleton (defines split partition, same as v1) |
| `depth` | int | **NEW MEANING** Max nesting depth of any CP (CP_sent or CP_rel); v1 `depth` only counted CP_sent |
| `tree_height` | int | **NEW** Max bracket-nesting depth of any leaf (number of brackets to open from S to reach the deepest terminal) |
| `is_ambiguous` | bool | **NEW** True if this row's HI surface string is also produced by a different frame |
| `n_tokens` | int | Terminal count (same as v1) |

### Position tags
Tags are assigned by **preorder traversal of the HI constituency tree** (depth-first, left-to-right). The same integer appears on the corresponding terminal in the HF tagged frame, so positions can be compared across orders.

Example:
```
HI: [N_proper_1 [V_dp_2 N_proper_3 [C_rel_4 V_intrans_5]]]
HF: [N_proper_1 [N_proper_3 [V_intrans_5 C_rel_4] V_dp_2]]
```

### Depth vs tree_height examples
```
[N_proper [V_dp N_proper]]                              depth=0, tree_height=2
[[D N_singular] [V_dp N_proper]]                        depth=0, tree_height=3
[N_proper [V_cp [C [N_proper VP]]]]                     depth=1, tree_height=4  (via CP_sent)
[[[Adj N_singular] [C_rel VP]] [V_dp N_proper]]         depth=1, tree_height=4  (via CP_rel)
```

---

## Pipeline

### Step 1: Build dataset (`run_proposal.py` → `run.py`)

```bash
uv run python experiments/13_adjunction_rel_clause_grammar/run.py \
    --k 100 \
    --split-seed 0 \
    --out experiments/13_adjunction_rel_clause_grammar/artifacts/datasets/frames \
    [--push kylelovesllms/hi_hf_v2_frames_d3_k100]
```

**Intermediate artifacts (always written):**

| File | Contents |
|---|---|
| `artifacts/datasets/debug/01_frames.csv` | All enumerated frames: `frame_id`, `depth`, `tree_height` |
| `artifacts/datasets/debug/02_sentences.csv` | All lexicalized rows before split (all columns) |
| `artifacts/datasets/debug/03_ambiguities.csv` | Subset where `is_ambiguous=True` |

**Split:** `random_frame` 80/10/10 by `frame_id`, `--split-seed 0`.

**Push:** If `--push` is provided, saves to HuggingFace Hub after local write.

### Step 2: Inspect and resolve ambiguities (`postprocess_proposal.py` → `postprocess.py`)

```bash
# Interactive mode
uv run python experiments/13_adjunction_rel_clause_grammar/postprocess.py \
    --ambiguities artifacts/datasets/debug/03_ambiguities.csv

# Apply saved decisions to produce a cleaned dataset
uv run python experiments/13_adjunction_rel_clause_grammar/postprocess.py \
    --apply artifacts/decisions/ambiguity_decisions_v1.csv \
    --sentences artifacts/datasets/debug/02_sentences.csv \
    --out artifacts/datasets/frames_cleaned
```

The postprocess script shows competing bracketed parses for each ambiguous surface string and records per-group decisions:
- `[K]eep both` — no change
- `[1]` / `[2]` — keep only the first/second frame's rows
- `[D]rop both` — remove all rows for this surface string
- `[S]kip` — defer (not written to decisions file)

Decisions are saved to `artifacts/decisions/ambiguity_decisions_v<N>.csv` (auto-versioned, never overwritten). The `--apply` mode re-applies decisions without prompting for reproducibility.

---

## Parameters

| Parameter | Value |
|---|---|
| Max depth | 3 (all-CP nesting: CP_sent + CP_rel) |
| k (lexicalizations per frame) | set via `--k` (required) |
| Split strategy | `random_frame` 80/10/10 |
| Split seed | 0 (default) |
| Adjunction recursion cap | 1 (hard-coded in enumerator) |

---

## Verification

```bash
# Check frame count at depth 3 (expect >> 90 due to new constructions)
uv run python -c "from grammar.v2 import enumerate_frames; print(sum(1 for _ in enumerate_frames(3)))"

# Small k smoke test
uv run python experiments/13_adjunction_rel_clause_grammar/run.py --k 5 --split-seed 0 \
    --out experiments/13_adjunction_rel_clause_grammar/artifacts/datasets/frames_test

# Spot-check position tags
uv run python -c "
from datasets import load_from_disk
ds = load_from_disk('experiments/13_adjunction_rel_clause_grammar/artifacts/datasets/frames_test')
for row in ds['train'].select(range(5)):
    print('HI tagged:', row['hi_frame_tagged'])
    print('HF tagged:', row['hf_frame_tagged'])
    print('depth:', row['depth'], '  tree_height:', row['tree_height'])
    print()
"

# Check schema
uv run python -c "
from datasets import load_from_disk
ds = load_from_disk('experiments/13_adjunction_rel_clause_grammar/artifacts/datasets/frames_test')
print(ds['train'].features)
print('Ambiguous rows:', sum(ds['train']['is_ambiguous']))
"
```

---

## Results

*(To be filled in after running.)*

---

## Open Questions

- What is the actual frame count at depth 3 with the new grammar? (Run smoke test above.)
- Are there many cross-frame ambiguities? (Check `03_ambiguities.csv` row count.)
- Should `C_rel` be the same token as `C` (`that`) or distinct? (Currently treated as the same word `that` but distinct POS.)

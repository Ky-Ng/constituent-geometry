# Experiment 17: Subject-Position Generalization of Adjunction & Relative Clauses

## Overview

A **syntactic-position generalization** dataset built on Grammar v2 (`src/grammar/v2/`).
Training contains **only atomic subjects** — every subject (matrix, embedded-clause,
and the residual subject inside a relative clause) is a proper name or a bare `D N`.
Objects may freely carry adjectives and relative clauses. The held-out val/test sets
are exactly the frames in which **some subject is modified** (adjective, relative
clause, or both).

**Question.** Having seen adjunction and relativization *only on objects*, can the
model generalize the same operation to the *subject* position (correctly producing
the HI↔HF reordering for a modified subject)?

This is a data-generation experiment (like 05/13). Training is a downstream step
(see *Downstream training* below) that reuses the exp-16 runner pattern on the
dataset this builds.

---

## Where is the grammar modified? — **Nowhere.**

`src/grammar/v2/` is imported **unchanged**. The "restricted grammar" (atomic
subjects) is not a new grammar at all — it is a **split predicate over the frames
v2 already enumerates**:

```python
frame_has_branching_subject(frame)  # in run.py
  False  -> training  (every subject atomic)
  True   -> held out  (some subject adj/RC-modified)
```

Rationale (vs. editing v2 or forking a `v3`):

- **Reproducibility.** v2 is untouched, so experiments 13–16 are unaffected. Any
  edit to v2's enumerator risks reordering frames and shifting every downstream
  RNG draw.
- **Same generative process.** Train and eval are drawn from one frame universe and
  differ *only* in the held-out construction — the cleanest possible contrast.
- **Mirrors existing design.** Exp 13 already partitions the same frames by a
  structural property (`held_out_depth`); this is the analogous `held_out_subject_branching`.

The only experiment-specific code lives in this folder's `run.py`:
1. `frame_has_branching_subject()` / `subject_mod_kinds()` — the predicate (a pure,
   read-only walk over v2 frame objects).
2. `--split-policy held_out_subject_branching`.

If a later experiment reuses the predicate, promote it into `src/grammar/v2/`.

---

## What counts as a "modified subject"

The predicate walks the whole frame and flags **any** subject DP that is
`NP_singular_adj` (adjective) or `NP_singular CP_rel` (relative clause). Subject
positions inspected:

| Position | Source rule |
|---|---|
| Matrix subject | `S -> DP VP` |
| Embedded clause subject | `CP_sent -> C S` (recurse into the embedded `S`) |
| Residual subject of an object-gap RC | `S_obj_gap -> DP VP_obj_gap` |
| Subjects inside a subject-gap RC's VP | `S_subj_gap -> VP` (recurse) |

Objects are never flagged for their *own* modifiers (those are allowed in training);
they are only walked *through* to reach subjects living inside their relative clauses.

Examples (held out — the **bold** noun is a modified subject):
- adj on matrix subj: *the **fancy friend** laughs*
- adj on embedded subj: *John likes that this **attractive brother** claps*
- adj on RC residual subj: *Betty tickles a cat that this **repulsive father** vexes*
- RC on subj: *the **dog that swims** likes the cat*

Examples (in training — subjects atomic, objects branch):
- *Mary hugs the teacher that dances* (object RC)
- *Betty soothes a mysterious brother that dances* (object adj + RC)

---

## Frame inventory (strict holdout ⇒ small training frame set)

Because *any* modified subject anywhere sends a frame to eval, the training **frame**
inventory is small (each is still lexicalized `k_train` times into many rows):

| `--max-depth` | total frames | train (atomic subj) | eval (subj modified) | eval kinds (adj / rel / adj+rel) |
|---|---|---|---|---|
| 0 | 24 | 16 | 8 | 8 / 0 / 0 |
| 1 | 2,500 | 160 | 2,340 | 140 / 800 / 1,400 |
| 2 | 1,393,524 | 1,312 | 1,392,212 | 1,748 / 214,512 / 1,175,952 |

Notes / design knobs:
- **Adjective holdout** appears already at depth 0; **relative-clause holdout** needs
  depth ≥ 1. Use **`--max-depth 2`** for the real run (matches exp 15/16).
- The eval set is dominated by `adj+rel` frames. The **`subject_mod`** column
  (`"" | "adj" | "rel" | "adj+rel"`) lets you slice eval cleanly — e.g. filter
  `subject_mod == "rel"` for *relative-clause-only* subject generalization. A
  uniform `--eval-frame-cap` will under-sample the rare pure-`adj` subject frames;
  stratifying by `subject_mod` is a possible enhancement if you want balanced slices.
- **Stricter vs. looser holdout.** The default holds out modified subjects at *every*
  depth (so the model never sees one). A looser variant (hold out only the *matrix*
  subject) would keep more training diversity but would leak modified subjects in
  embedded positions — not recommended if the claim is "never seen a modified subject."

---

## Schema (vs experiment 13)

Identical to exp 13 plus two columns:

| Column | Type | Definition |
|---|---|---|
| `subject_branching` | bool | True iff some subject in the frame is adj/RC-modified (defines the split) |
| `subject_mod` | str | `""` (train) / `"adj"` / `"rel"` / `"adj+rel"` — eval analysis slice |

All other columns (`hi`, `hf`, `hi_bracketed`, `hf_bracketed`, `hi_frame_tagged`,
`hf_frame_tagged`, `frame_id`, `depth`, `tree_height`, `is_ambiguous`, `n_tokens`,
`hi_structure`, `hf_structure`) are unchanged.

---

## Pipeline

### Step 0 (one-time): install the proposal
```bash
cp experiments/17_relative_clause_generalization/run_proposal.py \
   experiments/17_relative_clause_generalization/run.py
```

### Step 1: smoke test (inspect the partition, tiny k)
```bash
uv run python experiments/17_relative_clause_generalization/run.py \
    --max-depth 2 --split-policy held_out_subject_branching \
    --k-train 3 --k-eval 3 --eval-frame-cap 200 \
    --out experiments/17_relative_clause_generalization/artifacts/datasets/frames_test
```
Check `.../heldout_subj_branching_2/debug/01_frames.csv` — every `subject_branching=False`
row must have `subject_mod=""`, and no train row may contain a modified subject.

### Step 2: the real build
```bash
uv run python experiments/17_relative_clause_generalization/run.py \
    --max-depth 2 \
    --split-policy held_out_subject_branching \
    --k-train 100 --k-eval 50 --eval-frame-cap 500 \
    --split-seed 0 \
    --out experiments/17_relative_clause_generalization/artifacts/datasets/frames \
    --push kylelovesllms/hi-hf-v2-frames-subjholdout-d2-ktrain100-keval50
```

**Intermediate artifacts** (under `.../heldout_subj_branching_2/`):
`debug/01_frames.csv` (frame inventory + `subject_branching`/`subject_mod`),
`debug/02_sentences.csv` (all rows pre-split), `debug/03_ambiguities.csv`,
`frames_csv/{train,val,test}.csv`, and the Arrow dataset.

### Optional: full distribution (no holdout) for inspection
```bash
uv run python experiments/17_relative_clause_generalization/run.py \
    --max-depth 2 --split-policy random_frame --k 5 \
    --out experiments/17_relative_clause_generalization/artifacts/datasets/frames
```

---

## Downstream training

Same as exp 14–16: point an exp-16-style runner at the pushed Hub id. The held-out
eval split already isolates the subject-modification generalization, so standard
`eval_exact_match` on val/test measures it directly; slice by `subject_mod` to report
adjective-subject vs. relative-clause-subject generalization separately.

---

## Verification
```bash
# frame partition by depth
uv run python -c "
import sys; sys.path.insert(0, 'experiments/17_relative_clause_generalization')
import run_proposal as rp
from grammar.v2.generate_with_frames import enumerate_frames
for D in (0,1,2):
    fs = list(enumerate_frames(D))
    tr = sum(not rp.frame_has_branching_subject(f) for f in fs)
    print(f'depth<={D}: total={len(fs)} train={tr} eval={len(fs)-tr}')
"

# after build: train must be all subject_mod=='' ; eval none
uv run python -c "
from datasets import load_from_disk
ds = load_from_disk('experiments/17_relative_clause_generalization/artifacts/datasets/frames/heldout_subj_branching_2')
assert all(m=='' for m in ds['train']['subject_mod'])
assert all(m!='' for m in ds['validation']['subject_mod']+ds['test']['subject_mod'])
print('OK: train atomic-subject, val/test subject-modified')
"
```

---

## Results
*(To be filled in after running.)*

---

## Open Questions
- Depth choice: depth 2 gives 1,312 training frames — enough lexical diversity, or
  prefer a mixed depth-{1,2} build?
- Should the eval cap be **stratified** by `subject_mod` so adj / rel / adj+rel are
  comparably represented (a uniform cap is ~84% adj+rel at depth 2)?
- Report subject-adjective and subject-RC generalization separately, or pooled?

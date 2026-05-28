# 05 — Build HI/HF dataset from CFG frames

## Goal

Produce a HuggingFace `DatasetDict` of Head-Initial / Head-Final sentence pairs
that's tractable to materialize **and** supports structural-generalization
splits, replacing the exhaustive enumeration used in experiment 01.

The new sub-categorized grammar (`V_dp` / `V_cp` / `V_intrans`,
`NP_singular` / `NP_proper`, etc. — see [src/grammar/GRAMMAR.md](../../src/grammar/GRAMMAR.md))
blows the depth-3 sentence count from ~1.68M (the count experiment 01 was
sized for) to ~3.3B — too large to hold in a Python list. The fix is to
decouple structural enumeration from lexical sampling.

## Design

| layer | what | count at depth 3 |
|---|---|---|
| Frame (parse-tree skeleton, POS slots, no words) | exhaustively enumerated | **90** |
| Lexical fillings per frame | sampled, capped at `--k` | up to K |
| Final rows | K × #frames (upper bound) | e.g. K=100 → ≤ 9,000 |

A **frame** is a parse-tree whose leaves carry POS labels (`D`, `NP_singular`,
`V_dp`, `C`, ...) rather than words. The frame count is independent of the
lexicon, so adding nouns/verbs no longer multiplies the dataset size.

**Per-frame sampling** with a clause-scoped no-repeat constraint on
`NP_singular` eliminates the `the cat that likes the cat that likes the cat ...`
artifact that fully-lexicalized enumeration produces.

**Frame-based splits**: train/val/test partition **frames**, never rows. Every
lexicalization of a given parse skeleton lands in the same split — train and
test share no parse skeletons, so eval measures structural generalization.

Two split policies:
| policy | train | val | test | probes |
|---|---|---|---|---|
| `random_frame` | random 80% of frames | random 10% | random 10% | mixed (some structural variation between splits, no depth shift) |
| `held_out_depth` | all frames of depth `< max_depth` | half of depth `== max_depth` | other half | **recursion generalization** — model must apply the rule one level deeper than it saw at training time |

## Setup

```bash
# Random-frame split at depth 2 (42 frames, ~9k rows with K=100)
uv run python experiments/05_build_dataset_with_frames/run.py \
    --max-depth 2 --k 100 --split-policy random_frame \
    --out data/hi_hf_frames_d2_random

# Held-out-depth split at depth 2 (train depth 0-1, val/test depth 2)
uv run python experiments/05_build_dataset_with_frames/run.py \
    --max-depth 2 --k 100 --split-policy held_out_depth \
    --out data/hi_hf_frames_d2_heldout

# Depth 3 if you want the full 90-frame set
uv run python experiments/05_build_dataset_with_frames/run.py \
    --max-depth 3 --k 100 --split-policy held_out_depth \
    --out data/hi_hf_frames_d3_heldout

# Push to the Hub
uv run python experiments/05_build_dataset_with_frames/run.py \
    --max-depth 2 --k 100 --split-policy random_frame \
    --push your-username/hi-hf-toy-cfg-frames
```

Flags:
- `--max-depth N` — max CP-nesting depth to enumerate. Cumulative frame counts: 6 (d=0), 18 (d=1), 42 (d=2), 90 (d=3).
- `--k K` — upper bound on lexical samples per frame. Smaller frames (e.g. `S → DP_proper VP_intrans` has only 4·3 = 12 unique fillings) cap at fewer rows.
- `--split-seed S` — controls both the frame partition and the per-split sampler RNG. Three derived seeds: `S+1` (train), `S+2` (val), `S+3` (test).

## Schema

One row per unique lexicalization of a frame:

| column | type | meaning |
|---|---|---|
| `frame_id` | str | bracketed POS skeleton (surface labels); identical for all lexicalizations of the same frame; defines the split partition |
| `depth` | int | CP-nesting depth of the frame (0 .. `max_depth`) |
| `hi` | str | head-initial sentence |
| `hf` | str | head-final sentence |
| `hi_bracketed` | str | head-initial constituency parse (with words) |
| `hf_bracketed` | str | head-final constituency parse (with words) |
| `n_tokens` | int | terminal token count (same for `hi` and `hf` by the multiset property) |

## Results

_No runs yet._

## Notes

- **Sanity-check the enumeration first.** Before sampling, run
  `uv run python -m grammar.generate_with_frames --max-depth 3` to inspect the
  90 frame skeletons grouped by exact depth (6 / 12 / 24 / 48). Add `--coarse`
  for surface labels.
- **K=100 is a default, not a recommendation.** The right K depends on what
  the model needs to see. A depth-2 frame with two V_cp embeddings has
  hundreds of unique lexicalizations under the no-repeat constraint; K=100
  samples a small fraction. A depth-0 `S → DP_proper VP_intrans` frame has
  only 12 — K=100 will cap at 12 (and the loader sees the warning in
  `sample_pairs_from_frame`'s dedup loop).
- **No data regen needed for follow-up experiments.** The two output paths
  (`*_random/` and `*_heldout/`) are independent `DatasetDict`s; train two
  models, evaluate each on its own splits to compare random-frame vs
  held-out-depth generalization.
- **Backwards compat with experiment 01.** Experiment 01 still works against
  the legacy grammar/code path; this experiment is the new entry point for
  the sub-categorized grammar and frame-based pipeline.

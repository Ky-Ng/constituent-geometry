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

### Random, K = 100, depth=3

Link: [kylelovesllms/hi_hf_frames_d3_random_100](https://huggingface.co/datasets/kylelovesllms/hi_hf_frames_d3_random_100/viewer/default/train?p=1)

Command
```
sbatch slurm/run_cpu.sbatch experiments/05_build_dataset_with_frames/run.py --max-depth 3 --k 100 --split-policy random_frame --out experiments/05_build_dataset_with_frames/artifacts/hi_hf_frames_d3_random_100 --push kylelovesllms/hi_hf_frames_d3_random_100
```

Results
```
=== Dataset summary ===
Split            Rows       %  Frames   n_tokens (min/avg/max)
--------------------------------------------------------------
train           7,200   80.3%      72   3 / 11.6 / 17
validation        864    9.6%       9   2 / 11.6 / 15
test              900   10.0%       9   8 / 10.6 / 14
--------------------------------------------------------------
TOTAL           8,964               90

=== Depth breakdown (frames / rows per depth) ===
Split              depth=0       depth=1       depth=2       depth=3
--------------------------------------------------------------------
train            5 /   500    10 / 1,000    15 / 1,500    42 / 4,200
validation       1 /    64     0 /     0     4 /   400     4 /   400
test             0 /     0     2 /   200     5 /   500     2 /   200
--------------------------------------------------------------------
TOTAL            6 /   564    12 / 1,200    24 / 2,400    48 / 4,800

=== Rows per (split, depth) ===
Split          depth=0   depth=1   depth=2   depth=3     Total
--------------------------------------------------------------
train              500     1,000     1,500     4,200     7,200
validation          64         0       400       400       864
test                 0       200       500       200       900
--------------------------------------------------------------
TOTAL              564     1,200     2,400     4,800     8,964
```

### Heldout Depth 3, K = 100, depth=3

Link: [kylelovesllms/hi_hf_frames_d3_100_heldoutdepth_3](https://huggingface.co/datasets/kylelovesllms/hi_hf_frames_d3_100_heldoutdepth_3)

Command
```
sbatch slurm/run_cpu.sbatch experiments/05_build_dataset_with_frames/run.py --max-depth 3 --k 100 --split-policy held_out_depth --out experiments/05_build_dataset_with_frames/artifacts/hi_hf_frames_d3_100_heldoutdepth_3 --push kylelovesllms/hi_hf_frames_d3_100_heldoutdepth_3
```

Results
```
=== Dataset summary ===
Split            Rows       %  Frames   n_tokens (min/avg/max)
--------------------------------------------------------------
train           4,164   46.5%      42   2 / 8.6 / 13
validation      2,400   26.8%      24   11 / 13.8 / 17
test            2,400   26.8%      24   12 / 14.2 / 16
--------------------------------------------------------------
TOTAL           8,964               90

=== Depth breakdown (frames / rows per depth) ===
Split              depth=0       depth=1       depth=2       depth=3
--------------------------------------------------------------------
train            6 /   564    12 / 1,200    24 / 2,400     0 /     0
validation       0 /     0     0 /     0     0 /     0    24 / 2,400
test             0 /     0     0 /     0     0 /     0    24 / 2,400
--------------------------------------------------------------------
TOTAL            6 /   564    12 / 1,200    24 / 2,400    48 / 4,800

=== Rows per (split, depth) ===
Split          depth=0   depth=1   depth=2   depth=3     Total
--------------------------------------------------------------
train              564     1,200     2,400         0     4,164
validation           0         0         0     2,400     2,400
test                 0         0         0     2,400     2,400
--------------------------------------------------------------
TOTAL              564     1,200     2,400     4,800     8,964
```

### Heldout Depth 4, K = 100, depth=4

Link: [kylelovesllms/hi_hf_frames_d4_100_heldoutdepth_4](https://huggingface.co/datasets/kylelovesllms/hi_hf_frames_d4_100_heldoutdepth_4/viewer/default/train?p=1)

Command
```
sbatch slurm/run_cpu.sbatch experiments/05_build_dataset_with_frames/run.py --max-depth 4 --k 100 --split-policy held_out_depth --out experiments/05_build_dataset_with_frames/artifacts/hi_hf_frames_d4_100_heldoutdepth_4 --push kylelovesllms/hi_hf_frames_d4_100_heldoutdepth_4
```

Results
```
enumerating frames up to depth 4: 186 total frames
split policy = held_out_depth: train=90 frames, val=48 frames, test=48 frames

=== Dataset summary ===
Split            Rows       %  Frames   n_tokens (min/avg/max)
--------------------------------------------------------------
train           8,964   48.3%      90   2 / 11.5 / 17
validation      4,800   25.9%      48   15 / 17.7 / 21
test            4,800   25.9%      48   14 / 17.3 / 20
--------------------------------------------------------------
TOTAL          18,564              186

=== Depth breakdown (frames / rows per depth) ===
Split              depth=0       depth=1       depth=2       depth=3       depth=4
----------------------------------------------------------------------------------
train            6 /   564    12 / 1,200    24 / 2,400    48 / 4,800     0 /     0
validation       0 /     0     0 /     0     0 /     0     0 /     0    48 / 4,800
test             0 /     0     0 /     0     0 /     0     0 /     0    48 / 4,800
----------------------------------------------------------------------------------
TOTAL            6 /   564    12 / 1,200    24 / 2,400    48 / 4,800    96 / 9,600

=== Rows per (split, depth) ===
Split          depth=0   depth=1   depth=2   depth=3   depth=4     Total
------------------------------------------------------------------------
train              564     1,200     2,400     4,800         0     8,964
validation           0         0         0         0     4,800     4,800
test                 0         0         0         0     4,800     4,800
------------------------------------------------------------------------
TOTAL              564     1,200     2,400     4,800     9,600    18,564

saved 18,564 rows to experiments/05_build_dataset_with_frames/artifacts/hi_hf_frames_d4_100_heldoutdepth_4
pushed to https://huggingface.co/datasets/kylelovesllms/hi_hf_frames_d4_100_heldoutdepth_4
```

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

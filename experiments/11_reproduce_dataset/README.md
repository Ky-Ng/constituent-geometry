# 11_reproduce_dataset — Observations

## Goal
Regenerate the published dataset `kylelovesllms/hi_hf_frames_d3_random_100`
**locally and deterministically**, and **verify** the local copy is identical to the
Hub version — so future analyses (multi-seed work, golden/hypothesis attention
templates) can rely on the *exact same* train/val/test split that trained 06–09.
Also records which frames are in which split as a manifest.

## Why the splits are reproducible
The builder [05_build_dataset_with_frames/run.py](../05_build_dataset_with_frames/run.py)
is fully deterministic:
- frames are enumerated in fixed order (`enumerate_frames` — nested loops, no RNG);
- partitioned by a seeded shuffle `random.Random(--split-seed)` (default `0`) in `split_frames`;
- each split materialized with seeds `split_seed+1/+2/+3` (`materialize` → `sample_pair_from_frame`).

`VOCAB` is an ordered dict-of-lists and sets are used only for membership (not
iteration), so there is **no `PYTHONHASHSEED` / multiprocessing nondeterminism**, and
`random.Random` (Mersenne Twister) is stable across CPython versions. The published
dataset does **not** store the seed, but every row carries `frame_id`, which lets us
verify the partition.

## What `run.py` does
A thin orchestrator (copy `run_proposal.py` → `run.py`) — no build logic duplicated:
1. **Regenerate** (default): shells out to the exp-05 builder with the original args
   and **no `--push`** (so the Hub repo is never overwritten):
   `--max-depth 3 --k 100 --split-policy random_frame --split-seed 0 --out artifacts/hi_hf_frames_d3_random_100_repro`.
2. **Verify** per split (`train`/`validation`/`test`): row counts, column set,
   **order-robust** content equality (rows sorted by `frame_id, depth, hi, hf,
   hi_bracketed, hf_bracketed, n_tokens`), and the `frame_id` partition. Prints
   PASS/FAIL and exits non-zero on any mismatch.
3. Writes `results/split_manifest.json` = `{split: {n_rows, n_frames, frame_ids, rows_per_depth}}`.

## How to run
```bash
# regenerate locally + verify against the Hub dataset
uv run python experiments/11_reproduce_dataset/run.py

# verify an already-regenerated local copy (fast; proves determinism is in the build)
uv run python experiments/11_reproduce_dataset/run.py --no-regenerate

# just (re)build the local copy, skip verification
uv run python experiments/11_reproduce_dataset/run.py --no-verify
```
The build is CPU-only and quick; submit via `slurm/run_cpu.sbatch` if preferred.

## Expected result
- Splits: **train 7,200 / validation 864 / test 900** rows; **72 / 9 / 9** frames at depth 3.
- All three splits report **PASS** → `ALL SPLITS IDENTICAL ✅`.
- If a split FAILs, the most likely cause is drift in the installed `datasets`/grammar
  code since publication; the report names the differing rows/columns so the cause is visible.

## Results
_(Regenerated locally with `--split-seed 0` and verified against the Hub — verify run
2026-06-10 via `run.py --no-regenerate`.)_

**`ALL SPLITS IDENTICAL ✅` — the local rebuild is byte-identical to
`kylelovesllms/hi_hf_frames_d3_random_100`.** Every split PASSed all four checks
(row count, column set, order-robust content equality, `frame_id` partition):

```
[train]      PASS: rows 7200 vs 7200; cols ok; content ok; frames 72
[validation] PASS: rows  864 vs  864; cols ok; content ok; frames  9
[test]       PASS: rows  900 vs  900; cols ok; content ok; frames  9
```

This matches the expected counts exactly — the depth-3 frames are partitioned
**72 / 9 / 9** across train / val / test, and the published split that trained 06–09
is fully reproducible from the exp-05 builder.

`results/split_manifest.json` records the frame→split membership and per-depth rows:

| split | rows | frames | rows per depth |
|---|---|---|---|
| train | 7,200 | 72 | d0 500, d1 1000, d2 1500, **d3 4200** |
| validation | 864 | 9 | d0 64, d2 400, **d3 400** |
| test | 900 | 9 | d1 200, d2 500, **d3 200** |

(The `random_frame` policy mixes all depths into every split, so each split sees
depth-3 frames — unlike the `held_out_depth` split used in exp 10.)

## Notes
- `--no-push` is hardcoded in the regenerate step — this experiment can never overwrite the Hub dataset.

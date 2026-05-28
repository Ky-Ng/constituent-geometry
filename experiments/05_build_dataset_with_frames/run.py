"""Build a HuggingFace dataset of Head-Initial / Head-Final sentence pairs
from CFG frames + per-frame lexical sampling.

Background — why this exists alongside experiment 01:
   Experiment 01 enumerates *every* fully-lexicalized sentence (~1.68M at
   depth 3 with the old grammar). Under the new sub-categorized grammar
   (V_dp / V_cp / V_intrans, NP_singular / NP_proper, etc.) that count
   explodes to ~3.3B at depth 3 — too large to materialize. The fix is to
   enumerate over *frames* (parse-tree skeletons, only 90 at depth 3,
   independent of lexicon size) and sample K lexical fillings per frame.

Per-frame sampling also enables structural splits: train/val/test partition
**frames**, never rows, so every lexicalization of a given parse skeleton lands
in the same split — train and test share no parse skeletons.

Two split policies are supported (pick one per run; output paths should differ):

* ``random_frame``   — 80/10/10 random partition of frames into train/val/test.
* ``held_out_depth`` — train on frames of depth < max_depth, val/test = 50/50
  split of frames at depth == max_depth (probes recursion generalization).

Clause-scoped no-repeat: within any single S, the sampler does not reuse an
``NP_singular`` lemma. Embedded clauses get a fresh pool. This kills the
"the cat that likes a cat that likes the cat ..." artifact.

Generate a depth-2 dataset with random-frame splits (~42 frames × K rows):
    uv run python experiments/05_build_dataset_with_frames/run.py \\
        --max-depth 2 --k 100 --split-policy random_frame \\
        --out data/hi_hf_frames_d2_random

Same depth with held-out-depth splits (train on depth 0-1, val/test on depth 2):
    uv run python experiments/05_build_dataset_with_frames/run.py \\
        --max-depth 2 --k 100 --split-policy held_out_depth \\
        --out data/hi_hf_frames_d2_heldout

Push to the Hub:
    uv run python experiments/05_build_dataset_with_frames/run.py \\
        --max-depth 2 --k 100 --split-policy random_frame \\
        --push your-username/hi-hf-toy-cfg-frames

Schema (one row per unique lexicalization of a frame):
    frame_id      str   bracketed POS skeleton (surface labels); shared by all
                        lexicalizations of the same frame; defines the split.
    depth         int   CP-nesting depth of the frame (0..max_depth).
    hi            str   head-initial sentence
    hf            str   head-final sentence
    hi_bracketed  str   head-initial constituency parse (with words)
    hf_bracketed  str   head-final constituency parse (with words)
    n_tokens      int   number of terminal tokens (same for hi and hf)
"""

import argparse
import random

from datasets import Dataset, DatasetDict
from tqdm import tqdm

from grammar.generate_with_frames import (
    FrameS,
    count_frames,
    enumerate_frames,
    sample_pairs_from_frame,
)


def _row(pair) -> dict:
    return {
        "frame_id": pair.frame_id,
        "depth": pair.depth,
        "hi": pair.hi,
        "hf": pair.hf,
        "hi_bracketed": pair.hi_bracketed,
        "hf_bracketed": pair.hf_bracketed,
        "n_tokens": len(pair.hi_tokens),
    }


def split_frames(
    frames: list[FrameS], policy: str, max_depth: int, split_seed: int
) -> tuple[list[FrameS], list[FrameS], list[FrameS]]:
    """Partition frames into (train, val, test) by policy."""
    if policy == "random_frame":
        rng = random.Random(split_seed)
        shuffled = list(frames)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        return (
            shuffled[:n_train],
            shuffled[n_train : n_train + n_val],
            shuffled[n_train + n_val :],
        )
    if policy == "held_out_depth":
        shallow = [f for f in frames if f.depth() < max_depth]
        deep = [f for f in frames if f.depth() == max_depth]
        rng = random.Random(split_seed)
        rng.shuffle(deep)
        n_val = len(deep) // 2
        return shallow, deep[:n_val], deep[n_val:]
    raise ValueError(f"unknown --split-policy: {policy!r}")


def materialize(frames: list[FrameS], k: int, sampler_seed: int, desc: str) -> Dataset:
    """Sample up to ``k`` unique pairs per frame and return a Dataset."""
    rng = random.Random(sampler_seed)
    rows: list[dict] = []
    for frame in tqdm(frames, desc=desc, unit="frame"):
        for pair in sample_pairs_from_frame(frame, k, rng):
            rows.append(_row(pair))
    return Dataset.from_list(rows)


def build(max_depth: int, k: int, policy: str, split_seed: int) -> DatasetDict:
    total_frames = count_frames(max_depth)
    print(f"enumerating frames up to depth {max_depth}: {total_frames} total frames")

    frames = list(enumerate_frames(max_depth))
    assert len(frames) == total_frames, "frame count mismatch — generator bug?"

    train_f, val_f, test_f = split_frames(frames, policy, max_depth, split_seed)
    print(
        f"split policy = {policy}: "
        f"train={len(train_f)} frames, val={len(val_f)} frames, test={len(test_f)} frames"
    )

    # Distinct sampler seeds per split so train != val != test even for shared K.
    return DatasetDict(
        {
            "train": materialize(train_f, k, split_seed + 1, desc="train"),
            "validation": materialize(val_f, k, split_seed + 2, desc="val"),
            "test": materialize(test_f, k, split_seed + 3, desc="test"),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-depth", type=int, default=2, help="max CP-nesting depth")
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="lexical samples per frame (upper bound; small frames may yield fewer)",
    )
    parser.add_argument(
        "--split-policy",
        choices=["random_frame", "held_out_depth"],
        default="random_frame",
        help="how to partition frames into train/val/test",
    )
    parser.add_argument("--split-seed", type=int, default=0, help="seed for split + sampling")
    parser.add_argument(
        "--out",
        type=str,
        default="data/hi_hf_frames_dataset",
        help="local path to save_to_disk (data/ is gitignored)",
    )
    parser.add_argument(
        "--push",
        type=str,
        default=None,
        help="Hub repo id to push to, e.g. your-username/hi-hf-toy-cfg-frames",
    )
    parser.add_argument("--private", action="store_true", help="push the Hub dataset as private")
    args = parser.parse_args()

    ds = build(args.max_depth, args.k, args.split_policy, args.split_seed)
    print(ds)

    ds.save_to_disk(args.out)
    print(f"saved to {args.out}")

    if args.push:
        ds.push_to_hub(args.push, private=args.private)
        print(f"pushed to https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()

"""Build a HuggingFace dataset from the v2 grammar (relative clauses + adjuncts).

This is the dataset-builder for experiment 13. It follows the same frame-based
approach as experiment 05 but uses src/grammar/v2/ and extends the schema with:
  - hi_frame_tagged / hf_frame_tagged: position-tagged POS skeletons
  - depth: now counts ALL CPs (CP_sent + CP_rel), not just CP_sent
  - tree_height: max bracket-nesting depth of any leaf in the full tree
  - is_ambiguous: True if this row's HI surface string is also produced by
    a different frame (cross-frame syntactic ambiguity)

All outputs are nested under --out/depth_<N>/ so runs at different depths
never collide:
    --out/depth_<N>/            Arrow dataset (save_to_disk)
    --out/depth_<N>/debug/      intermediate CSVs (01_frames, 02_sentences, 03_ambiguities)
    --out/depth_<N>/frames_csv/ train/val/test CSVs

Usage:
    uv run python experiments/13_adjunction_rel_clause_grammar/run.py \\
        --k 100 \\
        --out experiments/13_adjunction_rel_clause_grammar/artifacts/datasets/frames

    # sample 2500 frames instead of enumerating all (useful for depth >= 2)
    uv run python experiments/13_adjunction_rel_clause_grammar/run.py \\
        --k 100 --n-frames 2500 \\
        --out experiments/13_adjunction_rel_clause_grammar/artifacts/datasets/frames

    # also push to Hub
    uv run python experiments/13_adjunction_rel_clause_grammar/run.py \\
        --k 100 \\
        --out experiments/13_adjunction_rel_clause_grammar/artifacts/datasets/frames \\
        --push your-username/hi-hf-v2-frames-d3-k100

    # depth-generalization split: train on ALL depth-{0,1} frames (k=100 each),
    # eval on a random 500 depth-2 frames (k=50 each), eval split 50/50 val/test
    uv run python experiments/13_adjunction_rel_clause_grammar/run.py \\
        --max-depth 2 \\
        --split-policy held_out_depth --train-depths 0,1 --eval-depths 2 \\
        --k-train 100 --k-eval 50 --eval-frame-cap 500 \\
        --out experiments/13_adjunction_rel_clause_grammar/artifacts/datasets/frames

Split policy (--split-policy):
  random_frame   80/10/10 over frame_ids (default; same as experiment 05). Uses
                 --k and (optionally) --n-frames. Output lands in --out/depth_<N>/.
  held_out_depth depth-generalization split. The depth groups are sized
                 independently:
                   train  = frames whose depth is in --train-depths, optionally
                            capped to --train-frame-cap, lexicalized k=--k-train.
                   eval   = frames whose depth is in --eval-depths, optionally
                            capped to --eval-frame-cap, lexicalized k=--k-eval, then
                            split frame-disjoint 50/50 into validation + test.
                 Frame caps reservoir-sample using --frame-seed (all frames kept
                 when fewer than the cap). --k-train/--k-eval fall back to --k.
                 Defaults: train = depths < --max-depth, eval = {--max-depth}.
                 Output lands in --out/heldout_depth_<N>/ (never overwrites a
                 random_frame run). --n-frames is ignored here.

Schema (one row per unique lexicalization of a frame):
    hi_structure      str   structure tag (Subj/Verb/Obj + depth), HI order  <- new
    hf_structure      str   same structure tag in HF order  <- new
    frame_id          str   coarse-label bracketed skeleton (defines split)
    depth             int   max CP nesting (CP_sent + CP_rel)  <- new meaning
    tree_height       int   max leaf depth in full constituency tree  <- new
    hi                str   head-initial sentence
    hf                str   head-final sentence
    hi_bracketed      str   head-initial constituency parse (with words)
    hf_bracketed      str   head-final constituency parse (with words)
    hi_frame_tagged   str   HI skeleton with 1-indexed terminal positions  <- new
    hf_frame_tagged   str   HF skeleton with same positions reordered  <- new
    is_ambiguous      bool  True if HI string also from another frame  <- new
    n_tokens          int   terminal count

To install: copy this file to
    experiments/13_adjunction_rel_clause_grammar/run.py
"""

import argparse
import csv
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict
from tqdm import tqdm

from grammar.v2.generate_with_frames import (
    FrameS,
    FrameSentencePair,
    enumerate_frames,
    structure,
    sample_pairs_from_frame,
)

EXP_DIR = Path(__file__).parent
DECISIONS_DIR = EXP_DIR / "artifacts" / "decisions"


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------

def _pair_to_row(pair: FrameSentencePair) -> dict:
    return {
        "hi_structure": pair.hi_structure,
        "hf_structure": pair.hf_structure,
        "frame_id": pair.frame_id,
        "depth": pair.depth,
        "tree_height": pair.tree_height,
        "hi": pair.hi,
        "hf": pair.hf,
        "hi_bracketed": pair.hi_bracketed,
        "hf_bracketed": pair.hf_bracketed,
        "hi_frame_tagged": pair.hi_frame_tagged,
        "hf_frame_tagged": pair.hf_frame_tagged,
        "is_ambiguous": pair.is_ambiguous,
        "n_tokens": len(pair.hi_tokens),
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows):,} rows -> {path}")


# ---------------------------------------------------------------------------
# Reservoir sampling
# ---------------------------------------------------------------------------

def _reservoir_sample(iterable, k: int, rng: random.Random) -> list:
    """Draw k items uniformly at random from an iterable without full materialization.

    Uses Algorithm R (Vitter 1985). If the iterable has fewer than k items,
    all items are returned.
    """
    reservoir = []
    for i, item in enumerate(iterable):
        if len(reservoir) < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir


# ---------------------------------------------------------------------------
# Stage 1: enumerate frames
# ---------------------------------------------------------------------------

def stage1_enumerate(
    max_depth: int,
    debug_dir: Path,
    n_frames: int | None = None,
    frame_seed: int = 0,
) -> list[FrameS]:
    if n_frames is None:
        print(f"\n[Stage 1] Enumerating all frames (max_depth={max_depth})...")
        frames = list(enumerate_frames(max_depth))
    else:
        print(
            f"\n[Stage 1] Reservoir-sampling {n_frames} frames "
            f"(max_depth={max_depth}, frame_seed={frame_seed})..."
        )
        rng = random.Random(frame_seed)
        frames = _reservoir_sample(enumerate_frames(max_depth), n_frames, rng)

    print(f"  frames selected: {len(frames)}")

    frame_rows = [
        {
            "hi_structure": structure(f, head_initial=True),
            "hf_structure": structure(f, head_initial=False),
            "depth": f.depth(),
            "tree_height_est": f.tree_height_estimate(),
            "frame_id": f.bracketed_skeleton(fine=False),
            "frame_id_fine": f.bracketed_skeleton(fine=True),
        }
        for f in frames
    ]
    _write_csv(frame_rows, debug_dir / "01_frames.csv")
    return frames


# ---------------------------------------------------------------------------
# Stage 2: lexicalize frames
# ---------------------------------------------------------------------------

def _lexicalize_frames(frames: list[FrameS], k: int, rng: random.Random) -> list[dict]:
    """Sample up to k unique lexicalizations per frame. No I/O; rng is advanced in place."""
    rows: list[dict] = []
    for frame in tqdm(frames, desc="frames", unit="frame"):
        for pair in sample_pairs_from_frame(frame, k, rng):
            rows.append(_pair_to_row(pair))
    return rows


def _select_frames_by_depth(
    frames: list[FrameS], depths: set[int], cap: int | None, rng: random.Random
) -> list[FrameS]:
    """Keep frames whose depth is in `depths`; if `cap` is set and there are more
    than `cap`, reservoir-sample that many (all kept when fewer than `cap`)."""
    subset = [f for f in frames if f.depth() in depths]
    if cap is not None and len(subset) > cap:
        subset = _reservoir_sample(subset, cap, rng)
    return subset


def stage2_lexicalize(frames: list[FrameS], k: int, split_seed: int, debug_dir: Path) -> list[dict]:
    print(f"\n[Stage 2] Lexicalizing {len(frames)} frames (k={k})...")
    rng = random.Random(split_seed + 10)  # distinct from split seed
    all_rows = _lexicalize_frames(frames, k, rng)
    print(f"  total rows: {len(all_rows):,}")
    _write_csv(all_rows, debug_dir / "02_sentences.csv")
    return all_rows


def stage2_lexicalize_grouped(
    train_frames: list[FrameS],
    eval_frames: list[FrameS],
    k_train: int,
    k_eval: int,
    split_seed: int,
    debug_dir: Path,
) -> list[dict]:
    """held_out_depth Stage 2: lexicalize the two depth groups with separate k and
    return the combined rows (one shared rng so the run is deterministic)."""
    print(
        f"\n[Stage 2] Lexicalizing {len(train_frames)} train-depth frames "
        f"(k_train={k_train}) + {len(eval_frames)} eval-depth frames (k_eval={k_eval})..."
    )
    rng = random.Random(split_seed + 10)  # distinct from split seed
    train_rows = _lexicalize_frames(train_frames, k_train, rng)
    eval_rows = _lexicalize_frames(eval_frames, k_eval, rng)
    all_rows = train_rows + eval_rows
    print(f"  total rows: {len(all_rows):,} (train {len(train_rows):,} / eval {len(eval_rows):,})")
    _write_csv(all_rows, debug_dir / "02_sentences.csv")
    return all_rows


# ---------------------------------------------------------------------------
# Stage 3: cross-frame ambiguity detection
# ---------------------------------------------------------------------------

def stage3_ambiguity(rows: list[dict], debug_dir: Path) -> list[dict]:
    """Mark is_ambiguous=True for any row whose HI string comes from >1 frame."""
    print("\n[Stage 3] Detecting cross-frame ambiguities...")

    hi_to_frames: dict[str, set] = defaultdict(set)
    for row in rows:
        hi_to_frames[row["hi"]].add(row["frame_id"])

    ambiguous_hi = {hi for hi, frames in hi_to_frames.items() if len(frames) > 1}

    n_ambiguous = 0
    for row in rows:
        row["is_ambiguous"] = row["hi"] in ambiguous_hi
        if row["is_ambiguous"]:
            n_ambiguous += 1

    print(f"  ambiguous HI strings: {len(ambiguous_hi):,}")
    print(f"  ambiguous rows: {n_ambiguous:,} / {len(rows):,}")

    ambig_rows = [r for r in rows if r["is_ambiguous"]]
    _write_csv(ambig_rows, debug_dir / "03_ambiguities.csv")
    return rows


# ---------------------------------------------------------------------------
# Stage 4: split
# ---------------------------------------------------------------------------

def _parse_depths(spec: str | None) -> set[int] | None:
    """Parse a comma-separated depth spec like "0,1" into {0, 1}. None stays None."""
    if spec is None:
        return None
    return {int(x) for x in spec.split(",") if x.strip()}


def stage4_split(
    rows: list[dict],
    frames: list[FrameS],
    split_seed: int,
    policy: str = "random_frame",
    max_depth: int | None = None,
    train_depths: set[int] | None = None,
    eval_depths: set[int] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Partition rows into (train, val, test).

    policy="random_frame": 80/10/10 over frame_ids (default; original behavior).
    policy="held_out_depth": depth-generalization split. train = rows whose depth
        is in ``train_depths``; the rows at ``eval_depths`` are split frame-disjoint
        50/50 into validation + test (so all lexicalizations of a held-out frame
        stay in one eval split, and val/test share no frame). Rows whose depth is
        in neither set are dropped from all splits.
    """
    if policy == "random_frame":
        print("\n[Stage 4] Splitting (random_frame 80/10/10)...")

        all_frame_ids = [f.bracketed_skeleton(fine=False) for f in frames]
        rng = random.Random(split_seed)
        shuffled = list(all_frame_ids)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        train_ids = set(shuffled[:n_train])
        val_ids = set(shuffled[n_train : n_train + n_val])
        test_ids = set(shuffled[n_train + n_val :])

        train_rows = [r for r in rows if r["frame_id"] in train_ids]
        val_rows = [r for r in rows if r["frame_id"] in val_ids]
        test_rows = [r for r in rows if r["frame_id"] in test_ids]

        print(
            f"  train: {len(train_ids)} frames / {len(train_rows):,} rows\n"
            f"  val:   {len(val_ids)} frames / {len(val_rows):,} rows\n"
            f"  test:  {len(test_ids)} frames / {len(test_rows):,} rows"
        )
        return train_rows, val_rows, test_rows

    if policy == "held_out_depth":
        # Defaults mirror experiment 05: train on everything below max_depth,
        # hold out exactly the deepest level.
        if train_depths is None:
            train_depths = set(range(max_depth)) if max_depth is not None else set()
        if eval_depths is None:
            eval_depths = {max_depth} if max_depth is not None else set()
        if not train_depths.isdisjoint(eval_depths):
            raise ValueError(
                f"--train-depths {sorted(train_depths)} and --eval-depths "
                f"{sorted(eval_depths)} must be disjoint"
            )
        print(
            f"\n[Stage 4] Splitting (held_out_depth: train={sorted(train_depths)}, "
            f"eval={sorted(eval_depths)}, val/test 50/50 by frame)..."
        )

        train_rows = [r for r in rows if r["depth"] in train_depths]

        # Split the held-out frames (not rows) so val/test are frame-disjoint.
        # Sort before shuffling so the partition is reproducible under split_seed.
        eval_ids = sorted({r["frame_id"] for r in rows if r["depth"] in eval_depths})
        rng = random.Random(split_seed)
        rng.shuffle(eval_ids)
        n_val = len(eval_ids) // 2
        val_ids = set(eval_ids[:n_val])
        test_ids = set(eval_ids[n_val:])
        val_rows = [r for r in rows if r["frame_id"] in val_ids]
        test_rows = [r for r in rows if r["frame_id"] in test_ids]

        assigned = train_depths | eval_depths
        dropped_depths = sorted({r["depth"] for r in rows if r["depth"] not in assigned})
        if dropped_depths:
            n_dropped = sum(1 for r in rows if r["depth"] in set(dropped_depths))
            print(
                f"  WARNING: dropped {n_dropped:,} rows at depths {dropped_depths} "
                f"(in neither --train-depths nor --eval-depths)"
            )

        n_train_frames = len({r["frame_id"] for r in train_rows})
        print(
            f"  train: {n_train_frames} frames / {len(train_rows):,} rows\n"
            f"  val:   {len(val_ids)} frames / {len(val_rows):,} rows\n"
            f"  test:  {len(test_ids)} frames / {len(test_rows):,} rows"
        )
        return train_rows, val_rows, test_rows

    raise ValueError(f"unknown --split-policy: {policy!r}")


# ---------------------------------------------------------------------------
# Stage 5: write outputs
# ---------------------------------------------------------------------------

def stage5_write(
    train_rows: list[dict],
    val_rows: list[dict],
    test_rows: list[dict],
    depth_dir: Path,
    push: str | None,
    private: bool,
) -> None:
    print(f"\n[Stage 5] Writing outputs to {depth_dir}...")

    # CSV splits
    csv_dir = depth_dir / "frames_csv"
    _write_csv(train_rows, csv_dir / "train.csv")
    _write_csv(val_rows, csv_dir / "val.csv")
    _write_csv(test_rows, csv_dir / "test.csv")

    # Arrow dataset (save_to_disk writes into depth_dir directly)
    ds = DatasetDict({
        "train": Dataset.from_list(train_rows),
        "validation": Dataset.from_list(val_rows),
        "test": Dataset.from_list(test_rows),
    })
    depth_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(depth_dir))
    total = sum(len(s) for s in ds.values())
    print(f"  saved {total:,} rows (Arrow) -> {depth_dir}")
    print(ds)

    _print_stats(ds)

    if push:
        ds.push_to_hub(push, private=private)
        print(f"  pushed -> https://huggingface.co/datasets/{push}")


def _print_stats(ds: DatasetDict) -> None:
    total = sum(len(s) for s in ds.values())
    print("\n=== Dataset summary ===")
    header = f"{'Split':<12} {'Rows':>8} {'Frames':>8} {'% ambig':>9}"
    print(header)
    print("-" * len(header))
    for name, split in ds.items():
        n_ambig = sum(split["is_ambiguous"])
        pct_ambig = n_ambig / len(split) * 100 if split else 0
        print(f"{name:<12} {len(split):>8,} {len(set(split['frame_id'])):>8} {pct_ambig:>8.1f}%")
    print("-" * len(header))
    print(f"{'TOTAL':<12} {total:>8,}")

    # Depth breakdown
    all_depths = sorted(set(r for split in ds.values() for r in split["depth"]))
    print("\n=== Rows per (split, depth) ===")
    cell_w = 10
    header2 = f"{'Split':<12}" + "".join(f"{'d=' + str(d):>{cell_w}}" for d in all_depths) + f"{'Total':>{cell_w}}"
    print(header2)
    print("-" * len(header2))
    for name, split in ds.items():
        depth_counts: Counter = Counter(split["depth"])
        line = f"{name:<12}" + "".join(f"{depth_counts.get(d, 0):>{cell_w},}" for d in all_depths)
        line += f"{len(split):>{cell_w},}"
        print(line)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-depth", type=int, default=3,
        help="max all-CP nesting depth (CP_sent + CP_rel count)",
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help=(
            "lexical samples per frame (upper bound; small frames may yield fewer). "
            "Required for random_frame; for held_out_depth it is the fallback when "
            "--k-train / --k-eval are not given."
        ),
    )
    parser.add_argument(
        "--k-train", type=int, default=None,
        help="held_out_depth only: lexical samples per train-depth frame (falls back to --k).",
    )
    parser.add_argument(
        "--k-eval", type=int, default=None,
        help="held_out_depth only: lexical samples per eval-depth frame (falls back to --k).",
    )
    parser.add_argument(
        "--train-frame-cap", type=int, default=None,
        help=(
            "held_out_depth only: randomly keep at most this many train-depth frames "
            "(all kept if fewer). Uses --frame-seed."
        ),
    )
    parser.add_argument(
        "--eval-frame-cap", type=int, default=None,
        help=(
            "held_out_depth only: randomly keep at most this many eval-depth frames "
            "(all kept if fewer). Uses --frame-seed."
        ),
    )
    parser.add_argument(
        "--n-frames", type=int, default=None,
        help=(
            "random_frame only: reservoir-sample this many frames (across all depths) "
            "instead of enumerating all. Useful for depth >= 2 where full enumeration "
            "is infeasible. Ignored for held_out_depth (use --train/--eval-frame-cap)."
        ),
    )
    parser.add_argument(
        "--frame-seed", type=int, default=0,
        help="RNG seed for --n-frames reservoir sampling (independent of --split-seed)",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--split-policy",
        choices=["random_frame", "held_out_depth"],
        default="random_frame",
        help=(
            "random_frame: 80/10/10 over frames (default, original behavior). "
            "held_out_depth: depth-generalization split — train on --train-depths, "
            "hold out --eval-depths split 50/50 (by frame) into validation + test."
        ),
    )
    parser.add_argument(
        "--train-depths", type=str, default=None,
        help=(
            'held_out_depth only: comma-separated training depths, e.g. "0,1". '
            "Defaults to all depths < --max-depth."
        ),
    )
    parser.add_argument(
        "--eval-depths", type=str, default=None,
        help=(
            'held_out_depth only: comma-separated held-out depths, e.g. "2". '
            "Defaults to {--max-depth}."
        ),
    )
    parser.add_argument(
        "--out", type=str,
        default=str(EXP_DIR / "artifacts" / "datasets" / "frames"),
        help="local path for save_to_disk",
    )
    parser.add_argument(
        "--push", type=str, default=None,
        help="Hub repo id, e.g. your-username/hi-hf-v2-frames-d3-k100",
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    # Keep held-out-depth datasets in a separate subdir so they never overwrite
    # a random_frame run at the same --max-depth.
    if args.split_policy == "held_out_depth":
        subdir = f"heldout_depth_{args.max_depth}"
    else:
        subdir = f"depth_{args.max_depth}"
    depth_dir = Path(args.out) / subdir
    debug_dir = depth_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    DECISIONS_DIR.mkdir(parents=True, exist_ok=True)

    train_depths = _parse_depths(args.train_depths)
    eval_depths = _parse_depths(args.eval_depths)

    if args.split_policy == "held_out_depth":
        # Resolve depth sets (defaults mirror experiment 05) and per-group k.
        if train_depths is None:
            train_depths = set(range(args.max_depth))
        if eval_depths is None:
            eval_depths = {args.max_depth}
        if not train_depths.isdisjoint(eval_depths):
            parser.error(
                f"--train-depths {sorted(train_depths)} and --eval-depths "
                f"{sorted(eval_depths)} must be disjoint"
            )
        k_train = args.k_train if args.k_train is not None else args.k
        k_eval = args.k_eval if args.k_eval is not None else args.k
        if k_train is None or k_eval is None:
            parser.error("held_out_depth needs --k-train and --k-eval (or a shared --k).")
        if args.n_frames is not None:
            print(
                "  NOTE: --n-frames is ignored for held_out_depth; "
                "use --train-frame-cap / --eval-frame-cap instead."
            )

        # Full enumeration, then cap each depth group independently.
        frames = stage1_enumerate(
            args.max_depth, debug_dir, n_frames=None, frame_seed=args.frame_seed,
        )
        frame_rng = random.Random(args.frame_seed)
        train_frames = _select_frames_by_depth(
            frames, train_depths, args.train_frame_cap, frame_rng
        )
        eval_frames = _select_frames_by_depth(
            frames, eval_depths, args.eval_frame_cap, frame_rng
        )
        print(
            f"  selected {len(train_frames)} train-depth frames "
            f"(cap={args.train_frame_cap}) + {len(eval_frames)} eval-depth frames "
            f"(cap={args.eval_frame_cap})"
        )
        selected = train_frames + eval_frames
        rows = stage2_lexicalize_grouped(
            train_frames, eval_frames, k_train, k_eval, args.split_seed, debug_dir
        )
        rows = stage3_ambiguity(rows, debug_dir)
        train_rows, val_rows, test_rows = stage4_split(
            rows, selected, args.split_seed,
            policy="held_out_depth", max_depth=args.max_depth,
            train_depths=train_depths, eval_depths=eval_depths,
        )
    else:  # random_frame
        if args.k is None:
            parser.error("random_frame needs --k.")
        frames = stage1_enumerate(
            args.max_depth, debug_dir,
            n_frames=args.n_frames, frame_seed=args.frame_seed,
        )
        rows = stage2_lexicalize(frames, args.k, args.split_seed, debug_dir)
        rows = stage3_ambiguity(rows, debug_dir)
        train_rows, val_rows, test_rows = stage4_split(
            rows, frames, args.split_seed,
            policy="random_frame", max_depth=args.max_depth,
            train_depths=train_depths, eval_depths=eval_depths,
        )

    stage5_write(train_rows, val_rows, test_rows, depth_dir, args.push, args.private)


if __name__ == "__main__":
    main()

"""Build a HI/HF dataset that HOLDS OUT subject-modified frames (experiment 17).

Same frame-based pipeline as experiment 13, but with one new generalization
split. In training, every SUBJECT is atomic -- a proper name or a bare `D N`
-- while OBJECTS may freely carry adjectives and relative clauses. The held-out
val/test sets are exactly the frames in which some subject IS modified (by an
adjective, a relative clause, or both). The scientific question: having only
ever seen adjunction / relativization on OBJECTS, can the model generalize the
same operation to the SUBJECT position?

------------------------------------------------------------------------------
Where is the grammar modified?  --  Nowhere.
------------------------------------------------------------------------------
`src/grammar/v2/` is imported UNCHANGED. The "restricted grammar" (atomic
subjects) is realized as a *split predicate* over the existing v2 frame
universe: `frame_has_branching_subject()` below. Training frames are the ones
for which it is False; held-out frames are the ones for which it is True. This
keeps v2 stable (experiments 13-16 stay reproducible) and guarantees train and
eval are drawn from the *same* generative process, differing only in the
held-out construction -- exactly the spirit of `held_out_depth` in exp 13.

Schema (per unique lexicalization of a frame) -- adds two columns vs exp 13:
    subject_branching bool  True iff some subject in the frame is adj/RC-modified
    subject_mod       str   "" (train) | "adj" | "rel" | "adj+rel"  (eval slice)
(all other columns are identical to experiment 13's builder)

Usage:
    # Quick inspection of the whole distribution (no holdout), small k:
    uv run python experiments/17_relative_clause_generalization/run.py \\
        --max-depth 2 --k 5 --split-policy random_frame \\
        --out experiments/17_relative_clause_generalization/artifacts/datasets/frames

    # The real generalization split: train on atomic-subject frames, hold out
    # subject-modified frames (val/test 50/50 by frame). k_train=100, k_eval=50,
    # cap the held-out set at 500 frames.
    uv run python experiments/17_relative_clause_generalization/run.py \\
        --max-depth 2 \\
        --split-policy held_out_subject_branching \\
        --k-train 100 --k-eval 50 --eval-frame-cap 500 \\
        --out experiments/17_relative_clause_generalization/artifacts/datasets/frames \\
        [--push kylelovesllms/hi-hf-v2-frames-subjholdout-d2-ktrain100-keval50]

Output lands in --out/<subdir>/ where subdir is:
    depth_<N>                       for random_frame
    heldout_subj_branching_<N>      for held_out_subject_branching

To install: copy this file to
    experiments/17_relative_clause_generalization/run.py
"""

import argparse
import csv
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
    # frame classes used by the subject-branching predicate (read-only):
    FrameDPCommon,
    FrameDPProper,
    FrameNPsBar,
    FrameNPsAdj,
    FrameNPsRel,
    FrameSsubjGap,
    FrameSobjGap,
    FrameVPdp,
    FrameVPdpAdv,
    FrameVPcp,
    FrameVPcpAdv,
)

EXP_DIR = Path(__file__).parent
DECISIONS_DIR = EXP_DIR / "artifacts" / "decisions"


# ===========================================================================
# === NEW (exp 17): subject-branching predicate over UNMODIFIED v2 frames ===
# ===========================================================================
#
# A frame is a *training* frame iff every subject (matrix, embedded-clause, and
# the residual subject inside an object's relative clause) is atomic: a proper
# name (`FrameDPProper`) or a bare `D N` (`FrameDPCommon` whose NP is
# `FrameNPsBar`). Any adjective (`FrameNPsAdj`) or relative clause
# (`FrameNPsRel`) on ANY subject sends the frame to the held-out eval set.
# Objects are never inspected for their own modifiers -- only walked through to
# reach subjects that live inside their relative clauses.
#
# The walk returns the SET of modification kinds it finds on subjects
# (subset of {"adj", "rel"}) so eval rows can later be sliced by whether the
# held-out subject carried an adjective, a relative clause, or both. The boolean
# predicate is just "is that set non-empty".

def _subject_dp_kinds(dp) -> set[str]:
    """Modification kinds on a DP sitting in SUBJECT position, plus any nested
    subjects reachable through a relative clause on this subject."""
    if isinstance(dp, FrameDPProper):
        return set()
    nps = dp.nps_frame  # FrameDPCommon
    if isinstance(nps, FrameNPsBar):
        return set()
    if isinstance(nps, FrameNPsAdj):
        return {"adj"}
    # FrameNPsRel: this subject carries a relative clause (and maybe an adj core),
    # and the RC may itself contain further subjects to inspect.
    kinds = {"rel"}
    if isinstance(nps.core, FrameNPsAdj):
        kinds.add("adj")
    return kinds | _cp_rel_subject_kinds(nps.cp_rel)


def _object_dp_kinds(dp) -> set[str]:
    """Subject-modification kinds reachable through an OBJECT DP -- only via a
    relative clause on that object (the object's own adj/RC is allowed and
    contributes nothing here)."""
    if isinstance(dp, FrameDPCommon) and isinstance(dp.nps_frame, FrameNPsRel):
        return _cp_rel_subject_kinds(dp.nps_frame.cp_rel)
    return set()


def _cp_rel_subject_kinds(cp_rel) -> set[str]:
    s_gap = cp_rel.s_gap
    if isinstance(s_gap, FrameSsubjGap):
        # subject relativized out (gap == the modified noun): no overt subject
        # here, so descend into the relative clause's VP.
        return _vp_subject_kinds(s_gap.vp)
    # FrameSobjGap -> DP VP_obj_gap : the DP is the residual SUBJECT of the RC.
    return _subject_dp_kinds(s_gap.dp)


def _vp_subject_kinds(vp) -> set[str]:
    if isinstance(vp, (FrameVPdp, FrameVPdpAdv)):
        return _object_dp_kinds(vp.dp)
    if isinstance(vp, (FrameVPcp, FrameVPcpAdv)):
        return _frame_s_subject_kinds(vp.cp.s)  # CP_sent -> C S (embedded clause)
    return set()  # intransitive (+/- adverb): no complement


def _frame_s_subject_kinds(s) -> set[str]:
    return _subject_dp_kinds(s.dp) | _vp_subject_kinds(s.vp)


def subject_mod_kinds(frame: FrameS) -> frozenset[str]:
    """Set of subject-modification kinds (subset of {'adj','rel'}) anywhere in
    the frame. Empty => every subject is atomic (a training frame)."""
    return frozenset(_frame_s_subject_kinds(frame))


def frame_has_branching_subject(frame: FrameS) -> bool:
    """True iff some subject anywhere in the frame is adj- or RC-modified."""
    return bool(subject_mod_kinds(frame))


def _subject_mod_str(frame: FrameS) -> str:
    """'' | 'adj' | 'rel' | 'adj+rel' -- the eval analysis slice for a frame."""
    return "+".join(sorted(subject_mod_kinds(frame)))


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
    """Algorithm R (Vitter 1985); returns all items if fewer than k."""
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
            # NEW (exp 17): tag the frame inventory so the split is auditable.
            "subject_branching": frame_has_branching_subject(f),
            "subject_mod": _subject_mod_str(f),
        }
        for f in frames
    ]
    _write_csv(frame_rows, debug_dir / "01_frames.csv")
    return frames


# ---------------------------------------------------------------------------
# Stage 2: lexicalize frames
# ---------------------------------------------------------------------------

def _lexicalize_frames(frames: list[FrameS], k: int, rng: random.Random) -> list[dict]:
    """Sample up to k unique lexicalizations per frame. No I/O; rng advances in place."""
    rows: list[dict] = []
    for frame in tqdm(frames, desc="frames", unit="frame"):
        # NEW (exp 17): one predicate eval per frame, stamped onto every row.
        branching = frame_has_branching_subject(frame)
        mod = _subject_mod_str(frame)
        for pair in sample_pairs_from_frame(frame, k, rng):
            row = _pair_to_row(pair)
            row["subject_branching"] = branching
            row["subject_mod"] = mod
            rows.append(row)
    return rows


# === NEW (exp 17): frame selection by the subject-branching predicate ========
def _select_frames_by_branching(
    frames: list[FrameS], want_branching: bool, cap: int | None, rng: random.Random
) -> list[FrameS]:
    """Keep frames whose `frame_has_branching_subject` matches `want_branching`;
    if `cap` is set and there are more than `cap`, reservoir-sample that many."""
    subset = [f for f in frames if frame_has_branching_subject(f) == want_branching]
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
    """Lexicalize the two groups with separate k and return combined rows (one
    shared rng so the run is deterministic)."""
    print(
        f"\n[Stage 2] Lexicalizing {len(train_frames)} atomic-subject frames "
        f"(k_train={k_train}) + {len(eval_frames)} subject-modified frames (k_eval={k_eval})..."
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

def stage4_split(
    rows: list[dict],
    frames: list[FrameS],
    split_seed: int,
    policy: str = "held_out_subject_branching",
) -> tuple[list[dict], list[dict], list[dict]]:
    """Partition rows into (train, val, test).

    policy="random_frame": 80/10/10 over frame_ids (sanity / inspection only).
    policy="held_out_subject_branching": train = rows with subject_branching
        False; the subject-modified rows are split frame-disjoint 50/50 into
        validation + test (all lexicalizations of a held-out frame stay in one
        eval split, and val/test share no frame).
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

    # === NEW (exp 17) ======================================================
    if policy == "held_out_subject_branching":
        print(
            "\n[Stage 4] Splitting (held_out_subject_branching: "
            "train=atomic subjects, eval=subject-modified, val/test 50/50 by frame)..."
        )
        train_rows = [r for r in rows if not r["subject_branching"]]

        # Split the held-out frames (not rows) so val/test are frame-disjoint.
        # Sort before shuffling so the partition is reproducible under split_seed.
        eval_ids = sorted({r["frame_id"] for r in rows if r["subject_branching"]})
        rng = random.Random(split_seed)
        rng.shuffle(eval_ids)
        n_val = len(eval_ids) // 2
        val_ids = set(eval_ids[:n_val])
        test_ids = set(eval_ids[n_val:])
        val_rows = [r for r in rows if r["frame_id"] in val_ids]
        test_rows = [r for r in rows if r["frame_id"] in test_ids]

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

    csv_dir = depth_dir / "frames_csv"
    _write_csv(train_rows, csv_dir / "train.csv")
    _write_csv(val_rows, csv_dir / "val.csv")
    _write_csv(test_rows, csv_dir / "test.csv")

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

    # NEW (exp 17): rows per (split, subject_mod) -- confirms train is all "" and
    # exposes how eval splits across adj / rel / adj+rel.
    all_mods = sorted(set(r for split in ds.values() for r in split["subject_mod"]))
    print("\n=== Rows per (split, subject_mod) ===")
    cell_w = 12
    header2 = f"{'Split':<12}" + "".join(f"{repr(m or '<none>'):>{cell_w}}" for m in all_mods)
    print(header2)
    print("-" * len(header2))
    for name, split in ds.items():
        mod_counts: Counter = Counter(split["subject_mod"])
        line = f"{name:<12}" + "".join(f"{mod_counts.get(m, 0):>{cell_w},}" for m in all_mods)
        print(line)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-depth", type=int, default=2,
        help="max all-CP nesting depth (CP_sent + CP_rel count)",
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="random_frame: lexical samples per frame (required for random_frame).",
    )
    parser.add_argument(
        "--k-train", type=int, default=None,
        help="held_out_subject_branching: samples per atomic-subject frame (falls back to --k).",
    )
    parser.add_argument(
        "--k-eval", type=int, default=None,
        help="held_out_subject_branching: samples per subject-modified frame (falls back to --k).",
    )
    parser.add_argument(
        "--train-frame-cap", type=int, default=None,
        help="held_out_subject_branching: keep at most this many atomic-subject frames.",
    )
    parser.add_argument(
        "--eval-frame-cap", type=int, default=None,
        help="held_out_subject_branching: keep at most this many subject-modified frames.",
    )
    parser.add_argument(
        "--n-frames", type=int, default=None,
        help="random_frame only: reservoir-sample this many frames instead of enumerating all.",
    )
    parser.add_argument(
        "--frame-seed", type=int, default=0,
        help="RNG seed for frame caps / --n-frames (independent of --split-seed).",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--split-policy",
        choices=["random_frame", "held_out_subject_branching"],
        default="held_out_subject_branching",
    )
    parser.add_argument(
        "--out", type=str,
        default=str(EXP_DIR / "artifacts" / "datasets" / "frames"),
    )
    parser.add_argument("--push", type=str, default=None, help="Hub repo id")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if args.split_policy == "held_out_subject_branching":
        subdir = f"heldout_subj_branching_{args.max_depth}"
    else:
        subdir = f"depth_{args.max_depth}"
    depth_dir = Path(args.out) / subdir
    debug_dir = depth_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    DECISIONS_DIR.mkdir(parents=True, exist_ok=True)

    if args.split_policy == "held_out_subject_branching":
        k_train = args.k_train if args.k_train is not None else args.k
        k_eval = args.k_eval if args.k_eval is not None else args.k
        if k_train is None or k_eval is None:
            parser.error("held_out_subject_branching needs --k-train and --k-eval (or a shared --k).")

        # Full enumeration, then partition by the subject-branching predicate.
        frames = stage1_enumerate(args.max_depth, debug_dir, n_frames=None, frame_seed=args.frame_seed)
        frame_rng = random.Random(args.frame_seed)
        train_frames = _select_frames_by_branching(frames, False, args.train_frame_cap, frame_rng)
        eval_frames = _select_frames_by_branching(frames, True, args.eval_frame_cap, frame_rng)
        print(
            f"  selected {len(train_frames)} atomic-subject frames "
            f"(cap={args.train_frame_cap}) + {len(eval_frames)} subject-modified frames "
            f"(cap={args.eval_frame_cap})"
        )
        if not eval_frames:
            parser.error(
                "no subject-modified frames at this --max-depth; nothing to hold out "
                "(adjective holdout appears from depth 0, RC holdout from depth 1)."
            )
        selected = train_frames + eval_frames
        rows = stage2_lexicalize_grouped(
            train_frames, eval_frames, k_train, k_eval, args.split_seed, debug_dir
        )
        rows = stage3_ambiguity(rows, debug_dir)
        train_rows, val_rows, test_rows = stage4_split(
            rows, selected, args.split_seed, policy="held_out_subject_branching",
        )
    else:  # random_frame
        if args.k is None:
            parser.error("random_frame needs --k.")
        frames = stage1_enumerate(
            args.max_depth, debug_dir, n_frames=args.n_frames, frame_seed=args.frame_seed,
        )
        rows = stage2_lexicalize(frames, args.k, args.split_seed, debug_dir)
        rows = stage3_ambiguity(rows, debug_dir)
        train_rows, val_rows, test_rows = stage4_split(
            rows, frames, args.split_seed, policy="random_frame",
        )

    stage5_write(train_rows, val_rows, test_rows, depth_dir, args.push, args.private)


if __name__ == "__main__":
    main()

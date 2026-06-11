"""Interactive ambiguity resolution for experiment 13's v2 grammar dataset.

Reads the cross-frame ambiguity report produced by run.py
(artifacts/datasets/debug/03_ambiguities.csv), shows competing bracketed parses
for each ambiguous HI surface string, and lets you decide per-group what to keep.

Decisions are saved to a versioned file:
    artifacts/decisions/ambiguity_decisions_v<N>.csv
where N auto-increments so existing decision files are never overwritten.

A --apply mode replays a saved decisions file against the full sentence CSV
(02_sentences.csv) to emit a cleaned Arrow dataset — no prompts, fully
reproducible.

Usage — interactive:
    uv run python experiments/13_adjunction_rel_clause_grammar/postprocess.py \\
        --ambiguities artifacts/datasets/debug/03_ambiguities.csv

Usage — replay saved decisions:
    uv run python experiments/13_adjunction_rel_clause_grammar/postprocess.py \\
        --apply artifacts/decisions/ambiguity_decisions_v1.csv \\
        --sentences artifacts/datasets/debug/02_sentences.csv \\
        --out artifacts/datasets/frames_cleaned

Decision values in the CSV:
    keep_both   — retain rows from all frames (no change)
    keep_1      — retain only rows whose frame_id == the first frame seen for this hi
    keep_2      — retain only rows whose frame_id == the second frame seen for this hi
    drop_both   — remove all rows with this hi string
    (missing)   — hi string was skipped; treated as keep_both when applying

To install: copy this file to
    experiments/13_adjunction_rel_clause_grammar/postprocess.py
"""

import argparse
import csv
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path

from datasets import Dataset, DatasetDict

EXP_DIR = Path(__file__).parent
DECISIONS_DIR = EXP_DIR / "artifacts" / "decisions"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows):,} rows -> {path}")


def _next_decisions_path() -> Path:
    """Return artifacts/decisions/ambiguity_decisions_v<N>.csv with N auto-incremented."""
    DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        p = DECISIONS_DIR / f"ambiguity_decisions_v{n}.csv"
        if not p.exists():
            return p
        n += 1


# ---------------------------------------------------------------------------
# Group ambiguous rows by HI string
# ---------------------------------------------------------------------------

def _group_by_hi(rows: list[dict]) -> "OrderedDict[str, list[dict]]":
    """Return an ordered dict mapping hi_string -> list of rows (across all frames)."""
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for row in rows:
        groups.setdefault(row["hi"], []).append(row)
    return groups


# ---------------------------------------------------------------------------
# Interactive session
# ---------------------------------------------------------------------------

PROMPT = (
    "\n  [K]eep both  [1] keep frame 1  [2] keep frame 2  "
    "[D]rop both  [S]kip (decide later)\n  > "
)

VALID_CHOICES = {"k": "keep_both", "1": "keep_1", "2": "keep_2", "d": "drop_both", "s": None}


def _display_group(hi: str, rows: list[dict]) -> None:
    """Print the competing parses for one ambiguous HI string."""
    # Collect distinct (frame_id, hi_bracketed) pairs
    frames_seen: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for row in rows:
        fid = row["frame_id"]
        if fid not in seen_ids:
            seen_ids.add(fid)
            frames_seen.append((fid, row.get("hi_bracketed", ""), row.get("hf_bracketed", "")))

    print(f"\n{'='*72}")
    print(f"  HI string: \"{hi}\"")
    print(f"  {len(frames_seen)} competing frames:\n")
    for i, (fid, hi_b, hf_b) in enumerate(frames_seen, start=1):
        print(f"  Frame {i}: {fid}")
        print(f"    HI bracketed: {hi_b}")
        print(f"    HF bracketed: {hf_b}")


def run_interactive(ambiguities_path: Path) -> None:
    rows = _read_csv(ambiguities_path)
    if not rows:
        print("No ambiguities found — nothing to resolve.")
        return

    groups = _group_by_hi(rows)
    print(f"\nFound {len(groups)} ambiguous surface strings ({len(rows):,} rows total).")
    print("For each, choose what to keep in the final dataset.\n")

    # Per-group, collect the distinct frame_ids in encounter order
    group_frames: dict[str, list[str]] = {}
    for hi, group_rows in groups.items():
        seen: list[str] = []
        for r in group_rows:
            if r["frame_id"] not in seen:
                seen.append(r["frame_id"])
        group_frames[hi] = seen

    decisions: list[dict] = []  # [{hi, decision, frame_1, frame_2}]
    n_total = len(groups)
    n_done = 0

    for i, (hi, group_rows) in enumerate(groups.items(), start=1):
        _display_group(hi, group_rows)
        print(f"  ({i}/{n_total})", end="")

        while True:
            raw = input(PROMPT).strip().lower()
            if raw in VALID_CHOICES:
                break
            print("  Unrecognised — please type K, 1, 2, D, or S.")

        decision = VALID_CHOICES[raw]
        if decision is None:
            print("  Skipped.")
            continue

        frames = group_frames[hi]
        decisions.append({
            "hi": hi,
            "decision": decision,
            "frame_1": frames[0] if len(frames) > 0 else "",
            "frame_2": frames[1] if len(frames) > 1 else "",
        })
        n_done += 1
        print(f"  -> {decision}")

    # Save decisions
    if decisions:
        out_path = _next_decisions_path()
        _write_csv(decisions, out_path)
        print(f"\nSaved {len(decisions)} decisions -> {out_path}")
        print(f"Skipped {n_total - n_done} groups (treated as keep_both when applying).")
    else:
        print("\nNo decisions recorded.")


# ---------------------------------------------------------------------------
# Apply mode: replay decisions -> cleaned dataset
# ---------------------------------------------------------------------------

def run_apply(
    decisions_path: Path,
    sentences_path: Path,
    out_dir: Path,
) -> None:
    print(f"Reading decisions from {decisions_path}...")
    decisions_rows = _read_csv(decisions_path)
    print(f"Reading sentences from {sentences_path}...")
    all_rows = _read_csv(sentences_path)

    # Build decision map: hi -> (decision, frame_1, frame_2)
    decision_map: dict[str, tuple[str, str, str]] = {}
    for d in decisions_rows:
        decision_map[d["hi"]] = (d["decision"], d["frame_1"], d["frame_2"])

    kept: list[dict] = []
    dropped = 0
    for row in all_rows:
        hi = row["hi"]
        if hi not in decision_map:
            kept.append(row)
            continue
        decision, frame_1, frame_2 = decision_map[hi]
        if decision == "keep_both":
            kept.append(row)
        elif decision == "keep_1":
            if row["frame_id"] == frame_1:
                kept.append(row)
            else:
                dropped += 1
        elif decision == "keep_2":
            if row["frame_id"] == frame_2:
                kept.append(row)
            else:
                dropped += 1
        elif decision == "drop_both":
            dropped += 1
        else:
            kept.append(row)  # unknown decision: keep

    print(f"Kept {len(kept):,} rows, dropped {dropped:,} rows.")

    # Re-split using the same frame_id partition as in the original
    # (read frame_id -> split from the original splits if available,
    #  otherwise fall back to partitioning by frame_id membership)
    # For simplicity: rows already carry frame_id; split from kept by
    # which train/val/test CSV they'd belong to (read from frames_csv/).
    frames_csv_dir = sentences_path.parent.parent / "frames_csv"
    split_map: dict[str, str] = {}  # frame_id -> split
    for split_name in ("train", "val", "test"):
        csv_path = frames_csv_dir / f"{split_name}.csv"
        if csv_path.exists():
            for r in _read_csv(csv_path):
                split_map[r["frame_id"]] = split_name

    if not split_map:
        print(
            "WARNING: could not find frames_csv/ splits. "
            "Writing all kept rows to a single 'cleaned' split."
        )
        ds = DatasetDict({"cleaned": Dataset.from_list(kept)})
    else:
        train_kept = [r for r in kept if split_map.get(r["frame_id"]) == "train"]
        val_kept = [r for r in kept if split_map.get(r["frame_id"]) == "val"]
        test_kept = [r for r in kept if split_map.get(r["frame_id"]) == "test"]
        ds = DatasetDict({
            "train": Dataset.from_list(train_kept),
            "validation": Dataset.from_list(val_kept),
            "test": Dataset.from_list(test_kept),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    print(f"Cleaned dataset saved -> {out_dir}")
    print(ds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ambiguities",
        type=Path,
        default=EXP_DIR / "artifacts" / "datasets" / "debug" / "03_ambiguities.csv",
        help="path to 03_ambiguities.csv (produced by run.py)",
    )
    parser.add_argument(
        "--apply",
        type=Path,
        default=None,
        help="replay a saved decisions CSV (non-interactive)",
    )
    parser.add_argument(
        "--sentences",
        type=Path,
        default=EXP_DIR / "artifacts" / "datasets" / "debug" / "02_sentences.csv",
        help="full sentences CSV (needed with --apply)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=EXP_DIR / "artifacts" / "datasets" / "frames_cleaned",
        help="output directory for cleaned Arrow dataset (used with --apply)",
    )
    args = parser.parse_args()

    if args.apply:
        run_apply(args.apply, args.sentences, args.out)
    else:
        run_interactive(args.ambiguities)


if __name__ == "__main__":
    main()

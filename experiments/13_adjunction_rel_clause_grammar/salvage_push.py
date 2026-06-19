"""Salvage: build the HF dataset from already-written CSVs and push to the Hub.

Why this exists:
    run.py Stage 5 calls Dataset.from_list over the full ~2.79M-row Python list,
    which transposes it into columns AND builds an Arrow copy while the list is
    still alive -> ~54-62 GB peak, which OOMs even at --mem=64G.

    The train/val/test CSVs are already written and complete, so here we stream
    them CSV -> Arrow with load_dataset (batched, bounded memory) and push. Peak
    RAM stays well under 16 GB.

Schema is pinned explicitly so the CSV round-trip can't silently mistype a column
(e.g. is_ambiguous -> string, depth -> float).

Usage:
    uv run python experiments/13_adjunction_rel_clause_grammar/salvage_push.py \\
        --csv-dir .../depth_2_frames_1.3M_k2/depth_2/frames_csv \\
        --push kylelovesllms/hi-hf-v2-frames-1.3M-d2-k2-random
"""

import argparse
from pathlib import Path

from datasets import Features, Value, load_dataset

STR = Value("string")
INT = Value("int64")

# Mirrors _pair_to_row() in run.py.
FEATURES = Features({
    "hi_structure": STR,
    "hf_structure": STR,
    "frame_id": STR,
    "depth": INT,
    "tree_height": INT,
    "hi": STR,
    "hf": STR,
    "hi_bracketed": STR,
    "hf_bracketed": STR,
    "hi_frame_tagged": STR,
    "hf_frame_tagged": STR,
    "is_ambiguous": Value("bool"),
    "n_tokens": INT,
})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-dir", required=True,
                   help="dir containing train.csv / val.csv / test.csv")
    p.add_argument("--push", required=True, help="Hub repo id")
    p.add_argument("--private", action="store_true")
    p.add_argument("--save-to-disk", default=None,
                   help="optional local Arrow output dir (save_to_disk)")
    p.add_argument("--no-push", action="store_true",
                   help="load + report only; skip the Hub push (dry run)")
    args = p.parse_args()

    csv_dir = Path(args.csv_dir)
    data_files = {
        "train": str(csv_dir / "train.csv"),
        "validation": str(csv_dir / "val.csv"),
        "test": str(csv_dir / "test.csv"),
    }
    for split, path in data_files.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"missing {split} CSV: {path}")

    print(f"Loading CSVs from {csv_dir} (streaming CSV -> Arrow)...")
    # keep_default_na=False so string fields never get turned into None by
    # pandas' default NA tokens; bool/int columns still infer + cast via FEATURES.
    ds = load_dataset(
        "csv",
        data_files=data_files,
        features=FEATURES,
        keep_default_na=False,
    )
    print(ds)
    print("\nfeatures:", ds["train"].features)
    print("sample row:", ds["train"][0])
    for name, split in ds.items():
        print(f"  {name}: {len(split):,} rows")

    if args.save_to_disk:
        ds.save_to_disk(args.save_to_disk)
        print(f"  saved Arrow -> {args.save_to_disk}")

    if args.no_push:
        print("--no-push set; skipping Hub push.")
        return

    print(f"\nPushing -> {args.push} ...")
    ds.push_to_hub(args.push, private=args.private)
    print(f"  pushed -> https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()

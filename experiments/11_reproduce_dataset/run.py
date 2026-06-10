"""Reproduce + verify the published `hi_hf_frames_d3_random_100` splits.

WHY
---
Future analyses (multi-seed work, golden/hypothesis attention templates) need the
EXACT train/val/test split that trained 06-09, plus confidence that a locally
regenerated copy matches the published Hub dataset
`kylelovesllms/hi_hf_frames_d3_random_100`.

The builder is fully deterministic, so the splits are reproducible:
  * frames are enumerated in fixed order (no RNG),
  * partitioned by a seeded shuffle `random.Random(--split-seed)` (default 0),
  * materialized with per-split seeds `split_seed+1/+2/+3`.
There is no PYTHONHASHSEED / multiprocessing nondeterminism, and `random.Random`
(Mersenne Twister) is stable across CPython versions. The published dataset does
NOT store the seed, but every row carries `frame_id`, which lets us verify.

WHAT THIS DOES (no model/build logic duplicated)
------------------------------------------------
1. (default) Regenerate locally by shelling out to the EXISTING builder
   experiments/05_build_dataset_with_frames/run.py with the original args and
   NO --push (so the Hub repo is never overwritten).
2. Load the local copy (`load_from_disk`) and the published one (`load_dataset`).
3. Verify per split: row counts, column set, order-robust content equality, and
   the frame_id partition. Print PASS/FAIL.
4. Write results/split_manifest.json = {split: sorted(frame_ids)} (+ depth counts)
   as the human-readable record of which frames are in which split.

USAGE
-----
    uv run python experiments/11_reproduce_dataset/run.py                 # regenerate + verify
    uv run python experiments/11_reproduce_dataset/run.py --no-regenerate # verify existing local copy
    uv run python experiments/11_reproduce_dataset/run.py --no-verify     # just (re)build locally
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
RESULTS_DIR = EXP_DIR / "results"
DEFAULT_OUT = EXP_DIR / "artifacts" / "hi_hf_frames_d3_random_100_repro"

BUILDER = REPO_ROOT / "experiments/05_build_dataset_with_frames/run.py"

# The exact knobs that produced the published dataset (defaults of the builder,
# made explicit here so the reproduction is self-documenting).
BUILD_ARGS = dict(max_depth=3, k=100, split_policy="random_frame", split_seed=0)
HUB_ID = "kylelovesllms/hi_hf_frames_d3_random_100"
SPLITS = ["train", "validation", "test"]
# Columns to compare; frame_id + the surface/parse fields fully identify a row.
KEY_COLS = ["frame_id", "depth", "hi", "hf", "hi_bracketed", "hf_bracketed", "n_tokens"]


def regenerate(out_path: Path) -> None:
    """Shell out to the existing builder with the original args and NO --push."""
    cmd = [
        sys.executable, str(BUILDER),
        "--max-depth", str(BUILD_ARGS["max_depth"]),
        "--k", str(BUILD_ARGS["k"]),
        "--split-policy", BUILD_ARGS["split_policy"],
        "--split-seed", str(BUILD_ARGS["split_seed"]),
        "--out", str(out_path),
        # deliberately NO --push: never overwrite the Hub repo
    ]
    print(f"[regenerate] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _sorted_rows(ds, cols):
    """All rows as a sorted list of column-tuples (order-robust comparison)."""
    cols_present = [c for c in cols if c in ds.column_names]
    table = {c: ds[c] for c in cols_present}
    rows = [tuple(table[c][i] for c in cols_present) for i in range(len(ds))]
    rows.sort()
    return cols_present, rows


def verify(local_path: Path) -> bool:
    """Compare the local copy against the published Hub dataset. Returns all-pass."""
    from datasets import load_dataset, load_from_disk

    if not local_path.exists():
        sys.exit(f"local copy not found at {local_path}; run without --no-regenerate first.")
    print(f"[verify] local : {local_path}")
    print(f"[verify] hub   : {HUB_ID}")
    local = load_from_disk(str(local_path))
    hub = load_dataset(HUB_ID)

    all_pass = True
    manifest: dict[str, dict] = {}
    for split in SPLITS:
        if split not in local or split not in hub:
            print(f"  [{split}] FAIL: split missing (local={split in local}, hub={split in hub})")
            all_pass = False
            continue
        lds, hds = local[split], hub[split]

        # 1) row counts
        count_ok = len(lds) == len(hds)
        # 2) column sets
        lcols, hcols = set(lds.column_names), set(hds.column_names)
        cols_ok = lcols == hcols
        # 3) order-robust content equality on shared key columns
        lcols_used, lrows = _sorted_rows(lds, KEY_COLS)
        _, hrows = _sorted_rows(hds, KEY_COLS)
        content_ok = lrows == hrows
        n_diff = sum(1 for a, b in zip(lrows, hrows) if a != b) + abs(len(lrows) - len(hrows))
        # 4) frame_id partition
        lframes, hframes = set(lds["frame_id"]), set(hds["frame_id"])
        frames_ok = lframes == hframes

        ok = count_ok and cols_ok and content_ok and frames_ok
        all_pass &= ok
        print(f"  [{split}] {'PASS' if ok else 'FAIL'}: "
              f"rows {len(lds)} vs {len(hds)} ({'ok' if count_ok else 'MISMATCH'}); "
              f"cols {'ok' if cols_ok else f'local-only={lcols - hcols} hub-only={hcols - lcols}'}; "
              f"content {'ok' if content_ok else f'{n_diff} differing rows'}; "
              f"frames {len(lframes)} ({'ok' if frames_ok else 'MISMATCH'})")

        depth_frames = Counter(d for d in lds["depth"])
        manifest[split] = {
            "n_rows": len(lds),
            "n_frames": len(lframes),
            "frame_ids": sorted(lframes),
            "rows_per_depth": dict(sorted(depth_frames.items())),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS_DIR / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[verify] wrote manifest -> {manifest_path.relative_to(EXP_DIR)}")
    print(f"\n{'ALL SPLITS IDENTICAL ✅' if all_pass else 'MISMATCH DETECTED ❌ (see above)'}")
    return all_pass


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--regenerate", action=argparse.BooleanOptionalAction, default=True,
                   help="Rebuild the local copy via the exp-05 builder (no --push). "
                        "Use --no-regenerate to verify an existing local copy.")
    p.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True,
                   help="Compare the local copy against the published Hub dataset.")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help="Local save_to_disk path for the regenerated dataset.")
    args = p.parse_args()

    out_path = Path(args.out)
    if args.regenerate:
        regenerate(out_path)
    if args.verify:
        ok = verify(out_path)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

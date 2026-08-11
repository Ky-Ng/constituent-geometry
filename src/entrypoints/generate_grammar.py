"""Generate a parallel HI/HF corpus from the toy CFG.

Writes:
    data/grammar_samples.csv  (machine-readable: example_number, hi_tree, hf_tree, hi_surface, hf_surface)
    data/grammar_samples.txt  (side-by-side eyeballing view)

Trees are stored in NLTK Penn-Treebank bracket notation so they parse directly
via nltk.Tree.fromstring(...).

Usage:
    uv run python src/entrypoints/generate_grammar.py --n_samples 100 --seed 0
"""
import argparse
import csv
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Ensure `grammar` is importable when the script is run directly as a file path
# (the project does not currently ship an installed package).
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from grammar import sample_tree  # noqa: E402

FIELDS = ["example_number", "hi_tree", "hf_tree", "hi_surface", "hf_surface"]


def build_rows(n_samples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n_samples):
        tree = sample_tree(rng)
        rows.append({
            "example_number": i + 1,
            "hi_tree": tree.to_nltk_str("HI"),
            "hf_tree": tree.to_nltk_str("HF"),
            "hi_surface": " ".join(tree.linearize("HI")),
            "hf_surface": " ".join(tree.linearize("HF")),
        })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_pretty(rows: list[dict], path: Path) -> None:
    hi_w = max((len(r["hi_surface"]) for r in rows), default=2)
    hf_w = max((len(r["hf_surface"]) for r in rows), default=2)
    idx_w = max(len(str(rows[-1]["example_number"])) if rows else 1, 1)
    sep = "  |  "
    header = f"{'#':>{idx_w}}{sep}{'HI':<{hi_w}}{sep}{'HF':<{hf_w}}"
    rule = "-" * len(header)
    with path.open("w") as f:
        f.write(header + "\n")
        f.write(rule + "\n")
        for r in rows:
            f.write(
                f"{r['example_number']:>{idx_w}}{sep}"
                f"{r['hi_surface']:<{hi_w}}{sep}"
                f"{r['hf_surface']:<{hf_w}}\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()

    rows = build_rows(args.n_samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "grammar_samples.csv"
    txt_path = args.output_dir / "grammar_samples.txt"
    write_csv(rows, csv_path)
    write_pretty(rows, txt_path)

    print(f"Wrote {len(rows)} samples (seed={args.seed}):")
    print(f"  {csv_path}")
    print(f"  {txt_path}")


if __name__ == "__main__":
    main()

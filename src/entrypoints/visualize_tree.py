"""Visualize a parse tree from data/grammar_samples.csv.

Examples:
    # ASCII pretty-print of example 1 (both HI and HF):
    uv run --no-sync python src/entrypoints/visualize_tree.py --example 1

    # Only HI, saved as SVG:
    uv run --no-sync python src/entrypoints/visualize_tree.py --example 42 --which HI --format svg --save tree.svg

    # LaTeX qtree string (print to stdout):
    uv run --no-sync python src/entrypoints/visualize_tree.py --example 7 --which HF --format latex
"""
import argparse
import csv
import sys
from pathlib import Path

import nltk

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_row(csv_path: Path, example_number: int) -> dict:
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if int(row["example_number"]) == example_number:
                return row
    raise SystemExit(f"example_number={example_number} not found in {csv_path}")


def render(tree_str: str, fmt: str) -> str | None:
    tree = nltk.Tree.fromstring(tree_str)
    if fmt == "ascii":
        tree.pretty_print()
        return None
    if fmt == "latex":
        return tree.pformat_latex_qtree()
    if fmt == "svg":
        import svgling
        return svgling.draw_tree(tree).get_svg().tostring()
    raise ValueError(f"unknown format: {fmt}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=REPO_ROOT / "data" / "grammar_samples.csv")
    parser.add_argument("--example", type=int, required=True, help="example_number column value")
    parser.add_argument("--which", choices=["HI", "HF", "both"], default="both")
    parser.add_argument("--format", choices=["ascii", "svg", "latex"], default="ascii")
    parser.add_argument("--save", type=Path, default=None, help="Output file (svg/latex only)")
    args = parser.parse_args()

    row = load_row(args.csv, args.example)
    panels = [("HI", row["hi_tree"], row["hi_surface"])] if args.which == "HI" else \
             [("HF", row["hf_tree"], row["hf_surface"])] if args.which == "HF" else \
             [("HI", row["hi_tree"], row["hi_surface"]), ("HF", row["hf_tree"], row["hf_surface"])]

    outputs = []
    for label, tree_str, surface in panels:
        header = f"=== example {args.example} — {label} ==="
        print(header)
        print(f"surface: {surface}")
        out = render(tree_str, args.format)
        if out is not None:
            outputs.append((label, out))
            if args.format != "svg":
                print(out)
        print()

    if args.save is not None:
        if args.which == "both" and len(outputs) == 2:
            # write two files side by side
            stem, suffix = args.save.stem, args.save.suffix
            for label, out in outputs:
                path = args.save.with_name(f"{stem}_{label}{suffix}")
                path.write_text(out)
                print(f"wrote {path}", file=sys.stderr)
        elif len(outputs) == 1:
            args.save.write_text(outputs[0][1])
            print(f"wrote {args.save}", file=sys.stderr)


if __name__ == "__main__":
    main()

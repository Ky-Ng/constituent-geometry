"""Lexical sweep of the per-head patching analysis over *sets* of prompt pairs.

`run.py` patches one (original, counterfactual) pair. This runs the same five-site sweep
(`head_patching.sweep_site`) over a set of pairs that vary one lexical dimension at a time and
aggregates every cell across pairs:

    effect_mean / effect_std : mean and sample std (ddof=1) of normalized_effect
    hit_rate                 : fraction of pairs whose patched top-1 at the critical position is the
                               counterfactual answer (an IIA-style hit rate)

Sets (--set):
  01_nouns           transitive frame, V=chases, D1=the, D2=this; ordered (N1, N1') pairs sampled without
                     replacement from N_singular, N2/N2' drawn from the remaining nouns
  02_verbs           nouns fixed (dog/cat vs researcher/dancer), D1=the, D2=this; one pair per transitive verb
  03_determiners     nouns fixed, V=chases; one pair per (D1, D2) in {the, a, this}^2
  04_intransitive    <bos> D1 N1 V <sep> D1 N1 V <eos>, D1=the; (N1, N1') pairs x intransitive verbs
  05_proper_subject  <bos> N1 V D2 N2 <sep> N1 D2 N2 V <eos>, N1 from N_proper (no determiner), V=chases,
                     D2=this; the critical position is <sep> itself (it predicts N1)

The critical position is index(<sep>) + 1 (first head-final token) except for 05, where it is <sep>.
The answer is the token after the critical position (N1 vs N1'). Pairs whose clean top-1 at the
critical position is not the answer, or whose counterfactual clean top-1 is not its own answer, are
skipped and counted.

Outputs (under --out):
  results/<set>/<pair_id>/{descrip.json, <site>.json, whole_head_summary.csv}   same shape as run.py
  results/<set>/{aggregate.json, aggregate.csv, pairs.json}
  figures/<set>/{<site>.png, whole_head_summary.png}   cell value = mean effect, text = mean±std
  logs/<set>/sweep.log                                  console output
"""
import argparse
import csv
import itertools
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate
from transformer_lens.model_bridge import TransformerBridge

from head_patching import SITES, sweep_site
from metrics import logit_diff, normalized_effect
from plot import plot_site_heatmap, plot_whole_head_summary

# ----------------------------------------------------------------------------- vocabulary
DETERMINERS = ["the", "a", "this"]
N_SINGULAR = ["dog", "cat", "boy", "girl", "teacher", "student", "friend", "researcher", "dancer", "artist",
              "musician", "engineer", "father", "mother", "sister", "brother"]
N_PROPER = ["John", "Mary", "Iskarous", "Jia", "James", "Hamilton", "Betty", "Shri"]
# unambiguous transitive verbs only (likes/believes/hates/knows also take clauses; thinks/assumes/claims excluded)
V_TRANSITIVE = ["faces", "kisses", "chases", "pursues", "loves", "soothes", "hugs", "consoles", "tickles", "bedazzles", "vexes"]
V_INTRANSITIVE = ["swims", "dances", "sings", "laughs", "smiles", "claps", "jeers", "applauds"]
SPECIALS = ["<bos>", "<sep>", "<eos>"]

# ----------------------------------------------------------------------------- frames
# roles: one entry per token; slot names (D1, N1, V, D2, N2) are filled from a pair's slot dict,
# anything else is a literal token. critical: "after_sep" -> index(<sep>) + 1, "sep" -> index(<sep>).
FRAMES: dict[str, dict] = {
    "transitive": {"roles": ["<bos>", "D1", "N1", "V", "D2", "N2", "<sep>", "D1", "N1", "D2", "N2", "V", "<eos>"],
                   "critical": "after_sep"},
    "intransitive": {"roles": ["<bos>", "D1", "N1", "V", "<sep>", "D1", "N1", "V", "<eos>"],
                     "critical": "after_sep"},
    "proper_subject": {"roles": ["<bos>", "N1", "V", "D2", "N2", "<sep>", "N1", "D2", "N2", "V", "<eos>"],
                       "critical": "sep"},
}
SET_FRAME = {"01_nouns": "transitive", "02_verbs": "transitive", "03_determiners": "transitive",
             "04_intransitive": "intransitive", "05_proper_subject": "proper_subject"}
SET_DEFAULT_N = {"01_nouns": 30, "02_verbs": len(V_TRANSITIVE), "03_determiners": len(DETERMINERS) ** 2,
                 "04_intransitive": 16, "05_proper_subject": 16}


@dataclass
class Pair:
    pair_id: str
    frame: str
    original_slots: dict[str, str]
    counterfactual_slots: dict[str, str]

    def _render(self, slots: dict[str, str]) -> str:
        return " ".join(slots.get(role, role) for role in FRAMES[self.frame]["roles"])

    @property
    def original(self) -> str:
        return self._render(self.original_slots)

    @property
    def counterfactual(self) -> str:
        return self._render(self.counterfactual_slots)


def build_pairs(set_name: str, rng: random.Random, n_pairs: int) -> list[Pair]:
    frame = SET_FRAME[set_name]
    pairs: list[Pair] = []
    if set_name == "01_nouns":
        ordered = [(a, b) for a in N_SINGULAR for b in N_SINGULAR if a != b]
        for i, (n1, n1p) in enumerate(rng.sample(ordered, n_pairs)):
            n2, n2p = rng.sample([n for n in N_SINGULAR if n not in (n1, n1p)], 2)
            pairs.append(Pair(f"{i:02d}_{n1}-{n1p}", frame,
                              dict(D1="the", N1=n1, V="chases", D2="this", N2=n2),
                              dict(D1="the", N1=n1p, V="chases", D2="this", N2=n2p)))
    elif set_name == "02_verbs":
        for i, v in enumerate(V_TRANSITIVE[:n_pairs]):
            pairs.append(Pair(f"{i:02d}_{v}", frame,
                              dict(D1="the", N1="dog", V=v, D2="this", N2="cat"),
                              dict(D1="the", N1="researcher", V=v, D2="this", N2="dancer")))
    elif set_name == "03_determiners":
        for i, (d1, d2) in enumerate(list(itertools.product(DETERMINERS, repeat=2))[:n_pairs]):
            pairs.append(Pair(f"{i:02d}_{d1}-{d2}", frame,
                              dict(D1=d1, N1="dog", V="chases", D2=d2, N2="cat"),
                              dict(D1=d1, N1="researcher", V="chases", D2=d2, N2="dancer")))
    elif set_name == "04_intransitive":
        ordered = [(a, b) for a in N_SINGULAR for b in N_SINGULAR if a != b]
        verbs = list(itertools.islice(itertools.cycle(V_INTRANSITIVE), n_pairs))  # each verb ~equally often
        rng.shuffle(verbs)
        for i, ((n1, n1p), v) in enumerate(zip(rng.sample(ordered, n_pairs), verbs)):
            pairs.append(Pair(f"{i:02d}_{n1}-{n1p}_{v}", frame,
                              dict(D1="the", N1=n1, V=v), dict(D1="the", N1=n1p, V=v)))
    elif set_name == "05_proper_subject":
        ordered = [(a, b) for a in N_PROPER for b in N_PROPER if a != b]
        for i, (n1, n1p) in enumerate(rng.sample(ordered, n_pairs)):
            n2, n2p = rng.sample(N_SINGULAR, 2)
            pairs.append(Pair(f"{i:02d}_{n1}-{n1p}", frame,
                              dict(N1=n1, V="chases", D2="this", N2=n2),
                              dict(N1=n1p, V="chases", D2="this", N2=n2p)))
    else:
        raise ValueError(set_name)
    return pairs


def check_vocab(model: TransformerBridge) -> None:
    """Every word used here must be exactly one token (the tokenizer is word-level)."""
    words = DETERMINERS + N_SINGULAR + N_PROPER + V_TRANSITIVE + V_INTRANSITIVE + SPECIALS
    bad = {w: s for w in words if (s := model.to_str_tokens(model.to_tokens(w, prepend_bos=False))) != [w]}
    if bad:
        raise SystemExit(f"vocabulary words that are not single tokens: {bad}")


class Tee:
    """Mirror stdout into a log file."""
    def __init__(self, path: Path):
        self.file = open(path, "w")
        self.stdout = sys.stdout

    def write(self, s: str) -> None:
        self.stdout.write(s)
        self.file.write(s)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()


def cell_key(r: dict) -> tuple:
    return (r["site"], r["layer"], tuple(r["heads"]), None if r["positions"] is None else tuple(r["positions"]))


def describe_cell(c: dict, position_labels: list[str]) -> str:
    pos = "ALL" if c["positions"] is None else position_labels[c["positions"][0]]
    return f"{c['site']} L{c['layer']} H{'+'.join(map(str, c['heads']))} @ {pos}"


def key_cell_rows(cells: list[dict], roles: list[str], position_labels: list[str]) -> list[list[str]]:
    """The cells the single-pair analysis singled out, addressed by role so they transfer across frames."""
    n1, v = roles.index("N1"), roles.index("V")
    spec = [
        ("L1 H1 z @ ALL", ("z", 1, (1,), None)),
        ("L1 H0 z @ ALL", ("z", 1, (0,), None)),
        ("L1 H0+1 z @ ALL", ("z", 1, (0, 1), None)),
        ("L0 H1 z @ ALL", ("z", 0, (1,), None)),
        (f"L1 H1 v @ {position_labels[n1]}", ("v", 1, (1,), (n1,))),
        (f"L1 H0 v @ {position_labels[v]}", ("v", 1, (0,), (v,))),
        (f"L0 H1 v @ {position_labels[n1]}", ("v", 0, (1,), (n1,))),
        (f"L0 H1 z @ {position_labels[v]}", ("z", 0, (1,), (v,))),
    ]
    lookup = {cell_key(c): c for c in cells}
    rows = []
    for label, key in spec:
        c = lookup[key]
        rows.append([label, f"{c['effect_mean']:.2f} ± {c['effect_std']:.2f}", f"{c['hit_rate']:.2f}"])
    routing = [c for c in cells if c["site"] in ("q", "k", "pattern")]
    worst = max(routing, key=lambda c: abs(c["effect_mean"]))
    rows.append([f"max |mean| over q/k/pattern: {describe_cell(worst, position_labels)}",
                 f"{worst['effect_mean']:.2f} ± {worst['effect_std']:.2f}", f"{worst['hit_rate']:.2f}"])
    return rows


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--set", required=True, choices=list(SET_FRAME))
parser.add_argument("--model_name", default="kylelovesllms/gpt2-2l2h128d10ep3lr01drop-shift")
parser.add_argument("--n_pairs", type=int, default=None, help="Pairs to draw (default per set: see SET_DEFAULT_N)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--sites", nargs="+", default=list(SITES), choices=SITES)
parser.add_argument("--out", default=str(Path(__file__).parent), help="Experiment dir; writes results/<set>, figures/<set>, logs/<set>")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()


@torch.inference_mode()
def main() -> None:
    out = Path(args.out)
    results_dir, figures_dir, logs_dir = out / "results" / args.set, out / "figures" / args.set, out / "logs" / args.set
    for d in (results_dir, figures_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(logs_dir / "sweep.log")
    print(vars(args))

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    n_pairs = args.n_pairs if args.n_pairs is not None else SET_DEFAULT_N[args.set]
    pairs = build_pairs(args.set, rng, n_pairs)
    frame = FRAMES[SET_FRAME[args.set]]
    roles = frame["roles"]

    model = TransformerBridge.boot_transformers(args.model_name, device=args.device)
    check_vocab(model)
    attn_filter = lambda name: ".attn.hook_" in name  # noqa: E731

    used: list[dict] = []      # per pair: descrip + rows per site
    skipped: list[dict] = []
    print(f"\n{len(pairs)} pairs in set {args.set} (frame: {' '.join(roles)})")
    for pair in pairs:
        tokens_o = model.to_tokens(pair.original, prepend_bos=False)
        tokens_c = model.to_tokens(pair.counterfactual, prepend_bos=False)
        str_o, str_c = model.to_str_tokens(tokens_o), model.to_str_tokens(tokens_c)
        assert len(str_o) == len(str_c) == len(roles), (str_o, str_c, roles)
        critical_pos = str_o.index("<sep>") + (1 if frame["critical"] == "after_sep" else 0)
        answer_pos = critical_pos + 1
        answer_tok, cf_tok = str_o[answer_pos], str_c[answer_pos]
        assert answer_tok != cf_tok, pair
        answer_id, cf_id = int(tokens_o[0, answer_pos]), int(tokens_c[0, answer_pos])

        logits_o, _ = model.run_with_cache(tokens_o, names_filter=attn_filter)
        logits_c, cache_c = model.run_with_cache(tokens_c, names_filter=attn_filter)
        pred_o = model.to_str_tokens(logits_o.argmax(-1))
        pred_c = model.to_str_tokens(logits_c.argmax(-1))
        ld_clean = logit_diff(logits_o, critical_pos, answer_id, cf_id)
        ld_cf = logit_diff(logits_c, critical_pos, answer_id, cf_id)
        record = {
            "pair_id": pair.pair_id, "set": args.set, "frame": SET_FRAME[args.set], "roles": roles, "seed": args.seed,
            "model": args.model_name, "original": pair.original, "counterfactual": pair.counterfactual,
            "slots_original": pair.original_slots, "slots_counterfactual": pair.counterfactual_slots,
            "tokens_original": str_o, "tokens_counterfactual": str_c,
            "critical_pos": critical_pos, "answer_token": answer_tok, "counterfactual_token": cf_tok,
            "clean_top_token": pred_o[critical_pos], "counterfactual_top_token": pred_c[critical_pos],
            "logit_diff_clean": ld_clean, "logit_diff_counterfactual": ld_cf, "sites": args.sites,
        }
        status = []
        if pred_o[critical_pos] != answer_tok:
            status.append(f"clean top-1 is {pred_o[critical_pos]!r}, not {answer_tok!r}")
        if pred_c[critical_pos] != cf_tok:
            status.append(f"counterfactual top-1 is {pred_c[critical_pos]!r}, not {cf_tok!r}")
        print(f"[{pair.pair_id}] ld_clean={ld_clean:7.2f} ld_cf={ld_cf:7.2f}  {answer_tok} vs {cf_tok}  "
              + ("SKIP: " + "; ".join(status) if status else "ok"))
        if status:
            skipped.append(record | {"skip_reason": "; ".join(status)})
            continue

        pair_dir = results_dir / pair.pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        (pair_dir / "descrip.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
        record["rows"] = {}
        whole_head_rows = []
        for site in args.sites:
            cells = sweep_site(model, tokens_o, cache_c, site=site, critical_pos=critical_pos,
                               answer_id=answer_id, counterfactual_id=cf_id)
            rows = [c.to_json() | {"normalized_effect": normalized_effect(c.logit_diff, ld_clean, ld_cf)} for c in cells]
            (pair_dir / f"{site}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
            record["rows"][site] = rows
            whole_head_rows += [r for r in rows if r["positions"] is None]
        with open(pair_dir / "whole_head_summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["site", "layer", "heads", "logit_diff", "normalized_effect", "top_token"])
            for r in whole_head_rows:
                w.writerow([r["site"], r["layer"], "+".join(map(str, r["heads"])), f"{r['logit_diff']:.4f}",
                            f"{r['normalized_effect']:.4f}", r["top_token"]])
        used.append(record)

    print(f"\nused {len(used)} / {len(pairs)} pairs; skipped {len(skipped)}")
    if not used:
        raise SystemExit("no usable pairs")
    critical_positions = {p["critical_pos"] for p in used}
    assert len(critical_positions) == 1, critical_positions
    critical_pos = critical_positions.pop()

    # position labels: the literal token if the slot is constant across the set, else the role name
    position_labels = []
    for i, role in enumerate(roles):
        seen = {p["tokens_original"][i] for p in used} | {p["tokens_counterfactual"][i] for p in used}
        label = next(iter(seen)) if len(seen) == 1 else role
        position_labels.append(f"{i}_{'*' if i == critical_pos else ''}{label}")

    # ------------------------------------------------------------------ aggregate across pairs
    acc: dict[tuple, dict] = {}
    for p in used:
        for site in args.sites:
            for r in p["rows"][site]:
                a = acc.setdefault(cell_key(r), {"site": r["site"], "layer": r["layer"], "heads": r["heads"],
                                                 "positions": r["positions"], "effects": [], "hits": [], "lds": []})
                a["effects"].append(r["normalized_effect"])
                a["hits"].append(r["top_token"] == p["counterfactual_token"])
                a["lds"].append(r["logit_diff"])
    cells = []
    for a in acc.values():
        eff = np.array(a["effects"])
        cells.append({"site": a["site"], "layer": a["layer"], "heads": a["heads"], "positions": a["positions"],
                      "n": len(eff), "effect_mean": float(eff.mean()),
                      "effect_std": float(eff.std(ddof=1)) if len(eff) > 1 else 0.0,
                      "hit_rate": float(np.mean(a["hits"])), "logit_diff_mean": float(np.mean(a["lds"]))})

    ld_clean_all = np.array([p["logit_diff_clean"] for p in used])
    ld_cf_all = np.array([p["logit_diff_counterfactual"] for p in used])
    aggregate = {
        "set": args.set, "seed": args.seed, "model": args.model_name, "frame": SET_FRAME[args.set], "roles": roles,
        "n_pairs_requested": len(pairs), "n_pairs_used": len(used), "n_pairs_skipped": len(skipped),
        "skipped": [{"pair_id": s["pair_id"], "original": s["original"], "counterfactual": s["counterfactual"],
                     "reason": s["skip_reason"]} for s in skipped],
        "critical_pos": critical_pos, "position_labels": position_labels, "sites": args.sites,
        "logit_diff_clean": {"mean": float(ld_clean_all.mean()), "std": float(ld_clean_all.std(ddof=1)) if len(used) > 1 else 0.0,
                             "min": float(ld_clean_all.min()), "max": float(ld_clean_all.max())},
        "logit_diff_counterfactual": {"mean": float(ld_cf_all.mean()), "std": float(ld_cf_all.std(ddof=1)) if len(used) > 1 else 0.0,
                                      "min": float(ld_cf_all.min()), "max": float(ld_cf_all.max())},
        "effect_std_is": "sample std (ddof=1) over pairs",
        "hit_rate_is": "fraction of pairs whose patched top-1 at the critical position equals the counterfactual answer",
        "cells": cells,
    }
    (results_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False))
    with open(results_dir / "aggregate.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site", "layer", "heads", "position", "position_label", "n", "effect_mean", "effect_std", "hit_rate", "logit_diff_mean"])
        for c in cells:
            pos = "ALL" if c["positions"] is None else c["positions"][0]
            w.writerow([c["site"], c["layer"], "+".join(map(str, c["heads"])), pos,
                        "ALL" if c["positions"] is None else position_labels[pos], c["n"],
                        f"{c['effect_mean']:.4f}", f"{c['effect_std']:.4f}", f"{c['hit_rate']:.4f}", f"{c['logit_diff_mean']:.4f}"])
    (results_dir / "pairs.json").write_text(json.dumps(
        [{k: v for k, v in p.items() if k != "rows"} | {"used": True} for p in used]
        + [{k: v for k, v in s.items() if k != "rows"} | {"used": False} for s in skipped], indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------ figures (mean effect, text = mean±std)
    n_used = len(used)
    for site in args.sites:
        site_rows = [c | {"normalized_effect": c["effect_mean"], "top_token": ""} for c in cells if c["site"] == site]
        plot_site_heatmap(
            rows=site_rows, site=site, position_labels=position_labels, clean_top_token="",
            title=f"{args.set}: site={site}, mean normalized effect over n={n_used} pairs on logit(N1) - logit(N1') at pos {critical_pos}",
            out_path=figures_dir / f"{site}.png",
            cell_text=lambda r: f"{r['normalized_effect']:.2f}±{r['effect_std']:.2f}",
        )
    whole_rows = [c | {"normalized_effect": c["effect_mean"], "top_token": ""} for c in cells if c["positions"] is None]
    plot_whole_head_summary(
        whole_rows, sites=args.sites,
        title=f"{args.set}: whole-head patches, mean normalized effect over n={n_used} pairs at pos {critical_pos} (N1 vs N1')",
        out_path=figures_dir / "whole_head_summary.png",
        cell_text=lambda r: f"{r['normalized_effect']:.2f}±{r['effect_std']:.2f}\nhit {r['hit_rate']:.2f}",
    )

    # ------------------------------------------------------------------ console summary
    print(f"\nclean logit_diff: mean {ld_clean_all.mean():.2f} (min {ld_clean_all.min():.2f}, max {ld_clean_all.max():.2f});  "
          f"counterfactual: mean {ld_cf_all.mean():.2f} (min {ld_cf_all.min():.2f}, max {ld_cf_all.max():.2f})")
    print(f"position labels: {position_labels}")
    print(f"\nKey cells for {args.set} (n={n_used} pairs; mean ± std of normalized effect, hit rate):")
    print(tabulate(key_cell_rows(cells, roles, position_labels), headers=["cell", "effect", "hit rate"], tablefmt="github"))
    print("\nWhole-head patches (all positions), aggregated:")
    print(tabulate([[c["site"], f"L{c['layer']}", "H" + "+".join(map(str, c["heads"])),
                     f"{c['effect_mean']:.3f} ± {c['effect_std']:.3f}", f"{c['hit_rate']:.2f}"]
                    for c in cells if c["positions"] is None],
                   headers=["site", "layer", "heads", "norm. effect", "hit rate"]))
    print("\nresults ->", results_dir, "\nfigures ->", figures_dir, "\nlog ->", logs_dir / "sweep.log")


if __name__ == "__main__":
    main()

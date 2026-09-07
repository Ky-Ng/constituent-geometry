"""Per-head activation patching on the toy HI->HF model.

Question: which attention head(s) migrate the head-initial noun into the head-final slot?

For an (original, counterfactual) pair that differ only in nouns, e.g.

    original       <bos> the dog chases this cat <sep> the dog this cat chases <eos>
    counterfactual <bos> the researcher chases this dancer <sep> the researcher this dancer chases <eos>

the critical position is the first head-final `the` (index of <sep> + 1), where the model must
predict `dog` (original) vs `researcher` (counterfactual). For every site in {z, q, k, v, pattern},
every (layer, head) and every position (plus "all positions"), we patch the counterfactual
activation into the original run and read

    logit_diff        = logit(dog) - logit(researcher) at the critical position
    normalized_effect = (ld_patched - ld_clean) / (ld_counterfactual - ld_clean)   # 0 = no effect, 1 = fully flipped
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from tabulate import tabulate
from transformer_lens.model_bridge import TransformerBridge

from head_patching import SITES, sweep_site
from metrics import logit_diff, normalized_effect
from plot import plot_site_heatmap, plot_whole_head_summary

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model_name", default="kylelovesllms/gpt2-2l2h128d10ep3lr01drop-shift")
parser.add_argument("--original", required=True, help="Original prompt (activations get overwritten)")
parser.add_argument("--counterfactual", required=True, help="Counterfactual prompt (source of patched activations)")
parser.add_argument("--critical_pos", type=int, default=None,
                    help="Readout position. Default: index of <sep> + 1 (first head-final token)")
parser.add_argument("--sites", nargs="+", default=list(SITES), choices=SITES)
parser.add_argument("--out", default=str(Path(__file__).parent), help="Experiment dir; writes results/<id> and figures/<id>")
parser.add_argument("--tag", default="", help="Readable suffix for the run id")
parser.add_argument("--group", default="",
                    help="Subfolder under results/ and figures/ (e.g. 00_single_pair); empty = directly under results/")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()
print(vars(args))


@torch.inference_mode()
def main() -> None:
    model = TransformerBridge.boot_transformers(args.model_name, device=args.device)

    tokens_o = model.to_tokens(args.original, prepend_bos=False)
    tokens_c = model.to_tokens(args.counterfactual, prepend_bos=False)
    str_o = model.to_str_tokens(tokens_o)
    str_c = model.to_str_tokens(tokens_c)
    if len(str_o) != len(str_c):
        raise SystemExit(f"prompts must tokenize to the same length: {len(str_o)} vs {len(str_c)}\n{str_o}\n{str_c}")

    critical_pos = args.critical_pos if args.critical_pos is not None else str_o.index("<sep>") + 1
    answer_pos = critical_pos + 1
    answer_tok, cf_tok = str_o[answer_pos], str_c[answer_pos]
    if answer_tok == cf_tok:
        raise SystemExit(f"original and counterfactual agree at answer position {answer_pos} ({answer_tok!r}); nothing to measure")
    answer_id, cf_id = int(tokens_o[0, answer_pos]), int(tokens_c[0, answer_pos])

    attn_filter = lambda name: ".attn.hook_" in name  # noqa: E731
    logits_o, _ = model.run_with_cache(tokens_o, names_filter=attn_filter)
    logits_c, cache_c = model.run_with_cache(tokens_c, names_filter=attn_filter)

    pred_o = model.to_str_tokens(logits_o.argmax(-1))
    pred_c = model.to_str_tokens(logits_c.argmax(-1))
    print(tabulate([["pos"] + list(range(len(str_o))),
                    ["original"] + str_o, ["pred"] + pred_o,
                    ["counterfactual"] + str_c, ["pred"] + pred_c]))

    ld_clean = logit_diff(logits_o, critical_pos, answer_id, cf_id)
    ld_cf = logit_diff(logits_c, critical_pos, answer_id, cf_id)
    print(f"critical_pos={critical_pos} ({str_o[critical_pos]!r})  answer={answer_tok!r} vs counterfactual={cf_tok!r}")
    print(f"logit_diff clean={ld_clean:.3f}  counterfactual={ld_cf:.3f}  (normalized effect: 0 -> clean, 1 -> counterfactual)")

    descrip = {
        "model": args.model_name, "original": args.original, "counterfactual": args.counterfactual,
        "tokens_original": str_o, "tokens_counterfactual": str_c,
        "critical_pos": critical_pos, "answer_token": answer_tok, "counterfactual_token": cf_tok,
        "clean_top_token": pred_o[critical_pos], "counterfactual_top_token": pred_c[critical_pos],
        "logit_diff_clean": ld_clean, "logit_diff_counterfactual": ld_cf, "sites": args.sites,
    }
    run_id = hashlib.sha256(json.dumps(descrip, sort_keys=True).encode()).hexdigest()[:8]
    if args.tag:
        run_id += f"_{args.tag}"
    results_dir = Path(args.out) / "results" / args.group / run_id
    figures_dir = Path(args.out) / "figures" / args.group / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "descrip.json").write_text(json.dumps(descrip, indent=2, ensure_ascii=False))

    position_labels = [f"{i}_{t}" for i, t in enumerate(str_o)]
    for i, (o, c) in enumerate(zip(str_o, str_c)):
        if o != c:
            position_labels[i] = f"{i}_{c}->{o}"
    position_labels[critical_pos] = "*" + position_labels[critical_pos]

    whole_head_rows: list[dict] = []
    for site in args.sites:
        cells = sweep_site(model, tokens_o, cache_c, site=site, critical_pos=critical_pos,
                           answer_id=answer_id, counterfactual_id=cf_id)
        rows = [c.to_json() | {"normalized_effect": normalized_effect(c.logit_diff, ld_clean, ld_cf)} for c in cells]
        (results_dir / f"{site}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

        plot_site_heatmap(
            rows=rows, site=site, position_labels=position_labels, clean_top_token=pred_o[critical_pos],
            title=f"site={site}: normalized effect on logit({answer_tok}) - logit({cf_tok}) at pos {critical_pos}",
            out_path=figures_dir / f"{site}.png",
        )
        for r in rows:
            if r["positions"] is None:
                whole_head_rows.append({"site": site, "layer": r["layer"], "heads": r["heads"],
                                        "logit_diff": r["logit_diff"], "normalized_effect": r["normalized_effect"],
                                        "top_token": r["top_token"]})

    with open(results_dir / "whole_head_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site", "layer", "heads", "logit_diff", "normalized_effect", "top_token"])
        for r in whole_head_rows:
            w.writerow([r["site"], r["layer"], "+".join(map(str, r["heads"])), f"{r['logit_diff']:.4f}",
                        f"{r['normalized_effect']:.4f}", r["top_token"]])
    print("\nWhole-head patches (all positions):")
    print(tabulate([[r["site"], f"L{r['layer']}", "H" + "+".join(map(str, r["heads"])), f"{r['logit_diff']:.3f}",
                     f"{r['normalized_effect']:.3f}", r["top_token"]] for r in whole_head_rows],
                   headers=["site", "layer", "heads", "logit_diff", "norm. effect", "top token"]))

    plot_whole_head_summary(whole_head_rows, sites=args.sites,
                            title=f"whole-head patches: normalized effect at pos {critical_pos} ({answer_tok} vs {cf_tok})",
                            out_path=figures_dir / "whole_head_summary.png")
    print("\nresults ->", results_dir, "\nfigures ->", figures_dir)


if __name__ == "__main__":
    main()

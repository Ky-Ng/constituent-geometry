import hashlib
import json
from pathlib import Path
from jaxtyping import Float
from torch import Tensor
import torch
from transformer_lens.model_bridge import TransformerBridge

from causal_mediation_helpers import causal_intervention, get_logits_cache, print_aligned_predictions, plot_top_tok_predicted

def residual_stream_causal_mediation(*, 
    model_name: str, 
    original: str, 
    counterfactual: str, 
    out: str
) -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerBridge.boot_transformers(model_name, device=device)

    logits_original, cache_original, tokens_original = get_logits_cache(
        model=model,
        prompt=original,
        names_filter=(lambda name: "resid" in name)
    )

    print_aligned_predictions(
        logits=logits_original,
        model=model,
        input_tokens=model.to_tokens(original, prepend_bos=False)
    )

    logits_counterfactual, cache_counterfactual, tokens_counterfactual = get_logits_cache(
        model=model,
        prompt=counterfactual,
        names_filter=(lambda name: "resid" in name)
    )

    print_aligned_predictions(
        logits=logits_counterfactual,
        model=model,
        input_tokens=model.to_tokens(counterfactual, prepend_bos=False)
    )

    # Activation Patching
    new_logits_per_intervention: dict[tuple[str, int],
                                      Float[Tensor, "batch seq vocab"]] = {}
    _, num_tokens = tokens_original.shape

    for hook_name, act_cache in cache_counterfactual.items():
        for tok_pos in range(num_tokens):  # token position for better cache locality
            print(f"patching pos {tok_pos} layer {hook_name}")
            intervened_logits = causal_intervention(
                model=model,
                original_tokens=tokens_original,
                hook_pos=hook_name,
                token_pos=tok_pos,
                counterfactual_tensor=act_cache
            )

            new_logits_per_intervention[(
                hook_name, tok_pos)] = intervened_logits

    top_tok_per_pos_intervened: dict[tuple[str, int], list[str]] = {}

    top_tok_per_pos_original: list[str] = model.to_str_tokens(
        logits_original.argmax(dim=-1))  # type: ignore

    for (hook_name, layer), logits in new_logits_per_intervention.items():
        top_token_ids = logits.argmax(dim=-1)  # B, seq
        top_tok_per_pos_intervened[(hook_name, layer)] = model.to_str_tokens(  # type: ignore
            top_token_ids
        )

    descrip = {
        "original": original,
        "counterfactual": counterfactual,
        "model": model_name,
        "intervention_hooks": list(cache_counterfactual.keys())
    }

    # sort_keys → stable across dict orderings
    payload = json.dumps(descrip, sort_keys=True)
    run_id = hashlib.sha256(payload.encode()).hexdigest()[:8]  # Grab 8 chars
    out_dir = Path(out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "descrip.json", "w", encoding="utf-8") as f:
        json.dump(descrip, f, indent=2)

    for tok_pos, tok_name in enumerate(original.split()):
        plot_top_tok_predicted(
            target_pos=tok_pos,
            original_tokens=model.to_str_tokens(
                tokens_original),  # type: ignore
            counterfactual_tokens=model.to_str_tokens(
                tokens_counterfactual),  # type: ignore
            predicted_original_tokens=top_tok_per_pos_original,
            predicted_intervened_tokens_per_pos=top_tok_per_pos_intervened,
            out=out_dir / f"causal_mediation_{tok_pos}_{tok_name}.png"
        )

    print("outputs written to ", out_dir)
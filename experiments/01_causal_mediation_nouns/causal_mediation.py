from typing import List

import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens.model_bridge import TransformerBridge

from src.entrypoints.causal_mediation_helpers import causal_intervention, get_logits_cache, print_aligned_predictions


original = "<bos> the dog chases this cat <sep> the dog this cat chases <eos>"
counterfactual = "<bos> the researcher chases this dancer <sep> the researcher this dancer chases <eos>"

model = TransformerBridge.boot_transformers(
    "kylelovesllms/gpt2-2l2h128d10ep3lr01drop-shift", device="cpu")

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
    for tok_pos in range(num_tokens): # token position for better cache locality
        print(f"patching pos {tok_pos} layer {hook_name}")
        intervened_logits = causal_intervention(
            model=model,
            original_tokens=tokens_original,
            hook_pos=hook_name,
            token_pos=tok_pos,
            counterfactual_tensor=act_cache
        )

        new_logits_per_intervention[(hook_name, tok_pos)] = intervened_logits

intervened_output_prediction: dict[tuple[str, int], List[str] | List[List[str]]] = {}

for (hook_name, layer), logits in new_logits_per_intervention.items():
    top_token_ids = logits.argmax(dim=-1) # B, seq
    token_list = model.to_str_tokens(top_token_ids)
    intervened_output_prediction[(hook_name, layer)] = token_list

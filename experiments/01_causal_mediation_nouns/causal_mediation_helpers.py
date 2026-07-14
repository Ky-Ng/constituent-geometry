from typing import Callable
from jaxtyping import Float, Int

from tabulate import tabulate
from torch import Tensor
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge


def get_logits_cache(
    model: TransformerBridge,
    prompt: str,
    names_filter: Callable | None = None
) -> tuple[Float[Tensor, "batch seq vocab"], ActivationCache, Int[Tensor, "seq"]]:
    """
    Returns logits, cache
    """

    tokens = model.to_tokens(prompt, prepend_bos=False)
    print("Input Tokens", model.to_str_tokens(tokens))

    logits, cache = model.run_with_cache(
        input=tokens,
        names_filter=names_filter
    )

    return logits, cache, tokens


def print_aligned_predictions(
        logits: Float[Tensor, "batch seq vocab"],
        model: TransformerBridge,
        input_tokens: Int[Tensor, "batch seq"]
) -> None:
    print("Align predictions")
    ground_truth_vs_predicted = [
        ["Grn Truth: "] + model.to_str_tokens(input_tokens)[1:],
        ["Predicted: "] + model.to_str_tokens(logits.argmax(dim=-1))[:-1]
    ]
    print(tabulate(ground_truth_vs_predicted))


def causal_intervention(
    model: TransformerBridge,
    original_tokens: Int[Tensor, "batch seq"],
    hook_pos: str,
    token_pos: int,
    counterfactual_tensor: Float[Tensor, "batch seq d_model"]
) -> Float[Tensor, "batch seq vocab"]:
    """
    Patch the model's output from counterfactual's `token_pos` and component `hook_pos`
    into model on the original prompt and forward a natural generation

    Currently only patches the residual stream

    Returns logits for each token position
    """

    def patch_residual_hook_fn(
        activation: Float[Tensor, "batch seq d_model"],
        hook: HookPoint
    ) -> Float[Tensor, "batch seq d_model"]:
        activation[:, token_pos, :] = counterfactual_tensor[:, token_pos, :]
        return activation

    logits = model.run_with_hooks(
        input=original_tokens,
        fwd_hooks=[(hook_pos, patch_residual_hook_fn)]
    )

    return logits

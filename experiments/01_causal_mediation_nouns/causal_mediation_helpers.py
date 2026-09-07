from pathlib import Path
from typing import Callable, List
from jaxtyping import Float, Int

from tabulate import tabulate
from torch import Tensor
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge
from plot_helpers import plot_heatmap


def get_logits_cache(
    model: TransformerBridge,
    prompt: str,
    names_filter: Callable | None = None
) -> tuple[Float[Tensor, "batch seq vocab"], ActivationCache, Int[Tensor, "seq"]]:
    """
    Returns logits, cache, tokens
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
    """
    Sanity check a set of logits vs. the ground truth
    """
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


def plot_top_tok_predicted(
    target_pos: int,
    original_tokens: list[str],
    counterfactual_tokens: list[str],
    predicted_original_tokens: list[str],
    predicted_intervened_tokens_per_pos: dict[tuple[str, int], list[str]],
    out: str | Path
) -> None:
    """
    Plots 

                   target_pos top token at Interchange Intervention per layer per tok pos 
    layer 3     |
    layer 2     |
    layer 1     |
    layer 0     |
                    _   _   _   _  _
                    0   1   ... 8  
                    t   c       ^   
                    h   a       s  
                    e   t       e
                                p
                                v   

        In practice we will put an arrow that says 1_cat -> 1_dog when we replace the original tokens with the counterfactual tokens

    x-axis: Token position; intervened token -> original token
    y-axis: Layer; e.g. "blocks.1.resid_pre"
    z-axis: top token predicted at position `target_pos` after patching in the activation from the counterfactual token as x,y position
    """
    # Step 1) Get label of each token as either `original` or `counterfactual -> original`
    original_token_labels = [
        f"{i}_{token}" for i, token in enumerate(original_tokens)
    ]

    counterfactual_token_labels = [
        f"{i}_{token}" for i, token in enumerate(counterfactual_tokens)
    ]

    is_token_different = [
        orig != counter for orig, counter in zip(original_token_labels, counterfactual_token_labels)
    ]

    token_labels = [
        f"{counterfactual} -> {original}" if dif
        else original
        for original, counterfactual, dif
        in zip(original_token_labels, counterfactual_token_labels, is_token_different)
    ]

    # Step 2) Get labels in order and preserve order from dictionary
    layer_labels: list[str] = list(
        dict.fromkeys(hook_name for hook_name,
                      _ in predicted_intervened_tokens_per_pos.keys()).keys()
    )

    # Step 3) Prepare Z-axis with the predicted word
    # Case 1) Predicted == Intervened, heatmap[layer][token] = prediction
    # Case 2) Predicted == Intervened, heatmap[layer][token] = original -> intervened
    original_top = predicted_original_tokens[target_pos]
    top_tokens_grid: list[list[str]] = []
    is_flipped_grid: list[list[float]] = []

    for hook_name in layer_labels:
        top_tokens_row, is_flipped_row = [], []

        for tok_idx in range(len(token_labels)):
            intervened_top = predicted_intervened_tokens_per_pos[(
                hook_name, tok_idx)][target_pos]
            matched = (original_top == intervened_top)
            top_tokens_row.append(
                intervened_top if matched else f"{original_top} -> {intervened_top}")
            is_flipped_row.append(0.0 if matched else 1.0)

        top_tokens_grid.append(top_tokens_row)
        is_flipped_grid.append(is_flipped_row)

    token_labels[target_pos] = "*" + token_labels[target_pos]

    plot_heatmap(
        Z=is_flipped_grid,
        Z_text=top_tokens_grid,
        x_labels=token_labels,
        y_labels=layer_labels,
        x_axis_label="token position (counterfactual -> original)",
        y_axis_label="layer",
        title=f"Top Token at pos `{token_labels[target_pos]}` with Intervention",
        out_path=out,
    )

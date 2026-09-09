"""Runner for Generalized Activation Patching Experiment"""

import json
import logging
from tabulate import tabulate
import torch
from torch import Tensor
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.hook_points import HookPoint
from log_helper import init_logger
from prompt_helper import apply_template_toy_model
from plot_helpers import plot_heatmap
from metrics import logit_diff, normalized_effect

# Load JSON Configurations
JSON_FILE_ARG = "experiments/04_generalized_activation_patching/artifacts/run_configs/run_patching_test.json"
with open(JSON_FILE_ARG, "r") as config_file:
    configs = json.load(config_file)

with open(configs["prompts_file"], "r") as prompts_file:
    prompts = json.load(prompts_file)

init_logger(out_dir=configs["log_dir"])

# Load Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TransformerBridge.boot_transformers(
    configs["src_model_path"], device=device)

# Construct and Tokenize prompt: TODO implement separate tokenization functions per model type
PROMPT_IDX = 0

original_prompt = apply_template_toy_model(
    source_text=prompts['prompts'][PROMPT_IDX]['original'],
    translated_text=prompts['prompts'][PROMPT_IDX]['original_translated']
)

counterfactual_prompt = apply_template_toy_model(
    source_text=prompts['prompts'][PROMPT_IDX]['counterfactual'],
    translated_text=prompts['prompts'][PROMPT_IDX]['counterfactual_translated']
)

tokens_o = model.to_tokens(original_prompt, prepend_bos=False)
tokens_c = model.to_tokens(counterfactual_prompt, prepend_bos=False)

logging.info(
    f"original token str {model.to_str_tokens(tokens_o, prepend_bos=False)}")
logging.info(
    f"counterfactual token str {model.to_str_tokens(tokens_c, prepend_bos=False)}")
logging.info("Original vs. Counterfactual prompt\n"
             + tabulate([
                 ["original:"] +
                 model.to_str_tokens(tokens_o, prepend_bos=False),
                 ["counterfactual:"] +
                 model.to_str_tokens(tokens_c, prepend_bos=False)
             ])
             )

# Run + Cache counterfactual hookpoints
logits_c, cache_c = model.run_with_cache(
    tokens_c,
    names_filter=lambda hook_name: configs["target_hookpoint_filter"] in hook_name
)

# Run + Cache original hookpoints to calc Normalized Logit Diff
logits_o, _ = model.run_with_cache(
    tokens_o,
    names_filter=lambda _: False  # run with logits only
)

critical_pos = configs["critical_pos"]
critical_o_token_id, critical_c_token_id = tokens_o.squeeze(
    0)[critical_pos+1].item(), tokens_c.squeeze(0)[critical_pos+1].item()

critical_o_token_str, critical_c_token_str = model.to_single_str_token(
    critical_o_token_id), model.to_single_str_token(critical_c_token_id)

logging.info(
    f"Critical token at prediction pos {critical_pos}: \n" +
    tabulate(
        [["original:"] + [critical_o_token_id] + [critical_o_token_str],
         ["counterfactual:"] + [critical_c_token_id] + [critical_c_token_str]]
    )
)

logit_diff_o = logit_diff(logits=logits_o, pos=critical_pos, answer_id=critical_o_token_id, counterfactual_id=critical_c_token_id)
logit_diff_c = logit_diff(logits=logits_c, pos=critical_pos, answer_id=critical_o_token_id, counterfactual_id=critical_c_token_id)

# Run + Cache intervened logits
# Just batch the residual stream over token positions first
# SEQ_LEN = 13
# for hook_point in ["blocks.0.hook_resid_pre"]:
#     for pos in range(SEQ_LEN):
#         def my_hook(act: Tensor, hook: HookPoint):
#             act[:, pos, :] = cache_c[hook_point][:, pos, :]
#             return act

#         model.run_with_hooks(
#             tokens_o,
#             fwd_hooks=[(hook_point, my_hook)]
#         )

# Attempt at the batched version
# SEQ_LEN = 13
if tokens_o.shape != tokens_c.shape:
    raise RuntimeError(
        f"Only supporting same length counterfactual/original: shape mismatch, tokens_o shape {tokens_o.shape} != tokens_c shape {tokens_c.shape}")

seq_len = tokens_o.shape[-1]
rows = torch.arange(seq_len)
index_mask = torch.arange(seq_len)

hookpoints = list(cache_c.keys())
logit_diffs_i = []
top_logit = []
for hook_point in hookpoints:
    def my_hook(act: Tensor, hook: HookPoint):
        # Cache has shape [B=1, S, D] since run with a single prompt
        act[rows, index_mask, :] = cache_c[hook_point][0, index_mask, :]
        return act

    tokens_o_batch = tokens_o.expand(seq_len, -1)  # [1, B] -> [B, B]

    logits_i = model.run_with_hooks(
        tokens_o_batch,
        fwd_hooks=[(hook_point, my_hook)]
    )

    log_dif_i = logits_i.detach()[:, critical_pos, critical_o_token_id] - logits_i.detach()[:, critical_pos, critical_c_token_id]
    log_dif_i /= (logit_diff_c - logit_diff_o)
    breakpoint()
    logit_diffs_i.append(
       log_dif_i.cpu().tolist() 
    )

# TODO: Calculate Metric (e.g. IIA) over all prompts

# TODO: Plot first k=3 intervened prompts target logit diff

# Plot first k=1 intervened prompts target logit diff

# Plotting
position_labels = [f"{i}_{t}" for i, t in enumerate(model.to_str_tokens(tokens_o))]
for i, (o, c) in enumerate(zip(model.to_str_tokens(tokens_o), model.to_str_tokens(tokens_c))):
    if o != c:
        position_labels[i] = f"{i}_{c}->{o}"
position_labels[critical_pos] = "*" + position_labels[critical_pos]

plot_heatmap(
    Z=logit_diffs_i,
    Z_text=logit_diffs_i,
    x_labels=position_labels,
    y_labels=hookpoints,
    x_axis_label="token position (counterfactual -> original)",
    y_axis_label="layer",
    title=f"Top Token at pos `{critical_pos}` Intervention",
    out_path="foo.png",
)

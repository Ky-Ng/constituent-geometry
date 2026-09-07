"""Run Causal Mediation Experiment"""
import argparse
from causal_mediation import residual_stream_causal_mediation

# Step 1: Arg Parse with Help Message
parser = argparse.ArgumentParser(
    description="Run residual-stream causal mediation: patch activations from a "
                "counterfactual prompt into the original and measure the logit effect."
)
parser.add_argument(
    "--model_name",
    help="HuggingFace model name to load via TransformerBridge",
)
parser.add_argument(
    "--original",
    required=True,
    help="Original prompt whose activations get patched",
)
parser.add_argument(
    "--counterfactual",
    required=True,
    help="Counterfactual prompt providing the patched-in activations",
)
parser.add_argument(
    "--out",
    required=True,
    help="Output path for results/figure",
)
args = parser.parse_args()

print(vars(args))

print("Running Causal Mediation Experiment with following arguments")

# Step 2: Run Causal Mediation 
residual_stream_causal_mediation(
    model_name=args.model_name,
    original=args.original,
    counterfactual=args.counterfactual,
    out=args.out
)

print("#"*20, "finished", "#"*20)

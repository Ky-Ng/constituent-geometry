"""Pure metric functions for head patching. No model access, so these are unit-testable."""
from jaxtyping import Float
from torch import Tensor


def logit_diff(
    logits: Float[Tensor, "batch seq vocab"],
    pos: int,
    answer_id: int,
    counterfactual_id: int,
) -> float:
    """logit(answer) - logit(counterfactual) at one position (batch row 0).

    Positive means the model prefers the original answer; negative means it prefers the
    counterfactual answer.
    """
    return (logits[0, pos, answer_id] - logits[0, pos, counterfactual_id]).item()


def normalized_effect(ld_patched: float, ld_clean: float, ld_counterfactual: float) -> float:
    """Where the patched logit diff sits between the clean run (0.0) and the counterfactual run (1.0).

    0.0 -> the patch did nothing; 1.0 -> the patch moved the model all the way to the
    counterfactual answer; values outside [0, 1] are possible (overshoot / backfire).
    """
    denom = ld_counterfactual - ld_clean
    if denom == 0.0:
        raise ValueError("clean and counterfactual logit diffs are equal; the pair does not separate the answers")
    return (ld_patched - ld_clean) / denom

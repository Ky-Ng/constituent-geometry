"""Semantics checks for head_patching.py against the toy model.

Run:  HF_HOME=/scratch1/kgng/hf_cache uv run pytest experiments/04_ai_migration_head_analysis/test_head_patching.py -q
"""
import pytest
import torch

from head_patching import hook_name, make_patch_hook, run_patched
from metrics import logit_diff, normalized_effect

MODEL = "kylelovesllms/gpt2-2l2h128d10ep3lr01drop-shift"
ORIGINAL = "<bos> the dog chases this cat <sep> the dog this cat chases <eos>"
COUNTERFACTUAL = "<bos> the researcher chases this dancer <sep> the researcher this dancer chases <eos>"


def test_normalized_effect_endpoints():
    assert normalized_effect(2.0, 2.0, -3.0) == 0.0
    assert normalized_effect(-3.0, 2.0, -3.0) == 1.0
    assert normalized_effect(-0.5, 2.0, -3.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        normalized_effect(1.0, 2.0, 2.0)


def test_logit_diff_sign():
    logits = torch.zeros(1, 3, 5)
    logits[0, 2, 1] = 4.0
    logits[0, 2, 3] = 1.0
    assert logit_diff(logits, pos=2, answer_id=1, counterfactual_id=3) == pytest.approx(3.0)
    assert logit_diff(logits, pos=2, answer_id=3, counterfactual_id=1) == pytest.approx(-3.0)


@pytest.fixture(scope="module")
def setup():
    from transformer_lens.model_bridge import TransformerBridge
    model = TransformerBridge.boot_transformers(MODEL, device="cpu")
    tok_o = model.to_tokens(ORIGINAL, prepend_bos=False)
    tok_c = model.to_tokens(COUNTERFACTUAL, prepend_bos=False)
    with torch.inference_mode():
        _, cache_o = model.run_with_cache(tok_o)
        _, cache_c = model.run_with_cache(tok_c)
    return model, tok_o, cache_o, cache_c


def _capture(model, tokens, name, extra_hooks):
    """Run with `extra_hooks` and return the activation at `name` (read after all patches)."""
    store = {}

    def grab(act, hook):
        store["x"] = act.clone()
        return act

    with torch.inference_mode():
        model.run_with_hooks(tokens, fwd_hooks=extra_hooks + [(name, grab)])
    return store["x"]


def test_z_all_heads_all_positions_reproduces_counterfactual_attn_out(setup):
    """attn_out = z @ W_O + b_O, so patching every head at every position at layer 0 must give the
    counterfactual's attn_out at layer 0 (upstream of layer 0 nothing else changed)."""
    model, tok_o, cache_o, cache_c = setup
    name = hook_name(0, "z")
    hook = make_patch_hook("z", heads=[0, 1], positions=None, source=cache_c[name])
    attn_out = _capture(model, tok_o, "blocks.0.hook_attn_out", [(name, hook)])
    assert torch.allclose(attn_out, cache_c["blocks.0.hook_attn_out"], atol=1e-5)
    assert not torch.allclose(attn_out, cache_o["blocks.0.hook_attn_out"], atol=1e-3)


def test_single_position_single_head_patch_is_local(setup):
    """Patching head 0's z at position 3 only may change attn_out at position 3."""
    model, tok_o, cache_o, cache_c = setup
    name = hook_name(0, "z")
    hook = make_patch_hook("z", heads=[0], positions=[3], source=cache_c[name])
    attn_out = _capture(model, tok_o, "blocks.0.hook_attn_out", [(name, hook)])
    diff = (attn_out - cache_o["blocks.0.hook_attn_out"]).abs().amax(dim=-1)[0]  # [pos]
    assert diff[3] > 1e-4
    assert torch.all(diff[torch.arange(len(diff)) != 3] < 1e-6)


def test_pattern_and_v_patch_reproduce_counterfactual_z(setup):
    """z = pattern @ v per head. Patching both (all heads, all rows/positions) at layer 0 must
    reproduce the counterfactual's z exactly - this pins down the index semantics of `pattern`."""
    model, tok_o, cache_o, cache_c = setup
    p_name, v_name, z_name = hook_name(0, "pattern"), hook_name(0, "v"), hook_name(0, "z")
    hooks = [
        (p_name, make_patch_hook("pattern", heads=[0, 1], positions=None, source=cache_c[p_name])),
        (v_name, make_patch_hook("v", heads=[0, 1], positions=None, source=cache_c[v_name])),
    ]
    z = _capture(model, tok_o, z_name, hooks)
    assert torch.allclose(z, cache_c[z_name], atol=1e-5)


def test_run_patched_unchanged_when_source_is_original(setup):
    """Patching the original's own activations back in is a no-op (checks nothing leaks)."""
    model, tok_o, cache_o, _ = setup
    with torch.inference_mode():
        clean = model(tok_o)
    for site in ("z", "q", "k", "v", "pattern"):
        patched = run_patched(model, tok_o, layer=1, site=site, heads=[0, 1], positions=None,
                              cache_counterfactual=cache_o)
        assert torch.allclose(clean, patched, atol=1e-5), site

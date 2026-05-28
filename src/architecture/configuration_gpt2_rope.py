"""Configuration for the GPT2-style decoder-only Transformer with RoPE (experiment 08).

This is the HuggingFace "custom model" pattern: a ``PretrainedConfig`` subclass
holds every architectural hyperparameter and is what gets serialized to
``config.json``.

Why this is NOT just VaswaniConfig with ``is_encoder_decoder=False``:
* The tokenizer for experiment 08 adds a new ``<sot>`` special token at id=4,
  so the special-token-id defaults differ from VaswaniConfig (which derives
  them from ``cfg_vocab.token_to_id()``).
* We need a separate ``model_type`` string ("gpt2_rope") so ``AutoConfig``
  cannot accidentally load a Vaswani checkpoint as a GPT2-RoPE model and vice
  versa.
* ``activation`` defaults to GELU (GPT2 block), not ReLU (Vaswani FFN).

Special-token ids are passed in explicitly rather than derived from
``cfg_vocab.py`` because the SOT-augmented tokenizer for this experiment lives
in this experiment's ``artifacts/`` and does not share cfg_vocab's id layout.

To copy: rename this file to ``configuration_gpt2_rope.py``.
"""

from __future__ import annotations

from transformers import PretrainedConfig


class GPT2RoPEConfig(PretrainedConfig):
    """Hyperparameters for the GPT2-style decoder-only Transformer with RoPE.

    Defaults match experiments 06/07 width and shape (d_model=128, n_heads=4,
    n_layers=4, d_ff=512) so per-parameter capacity is comparable. The default
    vocab size is 59 = 5 specials (PAD, BOS, EOS, UNK, SOT) + 54 grammar
    terminals from ``grammar.cfg_vocab.grammar_words()``; the run script
    overrides this with ``tok.vocab_size`` at construction time.

    ``head_dim = d_model / n_heads`` must be even (RoPE constraint). For the
    default 128/4 = 32, that holds.
    """

    model_type = "gpt2_rope"

    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
    }

    def __init__(
        self,
        vocab_size: int = 59,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",            # GPT2 block uses GELU, not ReLU
        max_position_embeddings: int = 128,  # bumped from 64: concat seq is ~2x longer
        layer_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = True,
        # --- special token ids --------------------------------------------------
        # PAD=0, BOS=1, EOS=2, UNK=3, SOT=4 (matches build_tokenizer.py for exp 08).
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 3,
        sot_token_id: int = 4,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        # PretrainedConfig has no first-class slot for <sot>, so we attach it
        # ourselves. This lets the saved config.json round-trip the sot id, so a
        # reloaded model + tokenizer pair stays consistent without re-deriving it.
        self.unk_token_id = unk_token_id
        self.sot_token_id = sot_token_id

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )
        if (d_model // n_heads) % 2 != 0:
            raise ValueError(
                f"head_dim ({d_model // n_heads}) must be even for RoPE."
            )

        # Force is_encoder_decoder=False even if a stale config.json claims
        # otherwise -- decoder-only is part of this model's identity.
        kwargs.pop("is_encoder_decoder", None)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_encoder_decoder=False,
            **kwargs,
        )

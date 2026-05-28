"""Configuration for the Vaswani-style encoder-decoder Transformer.

This is the HuggingFace "custom model" pattern: a ``PretrainedConfig`` subclass
holds every architectural hyperparameter and is what gets serialized to
``config.json``. Because we inherit from ``PretrainedConfig``, we get
``save_pretrained`` / ``from_pretrained`` / ``to_json_file`` for free, and the
config round-trips through the exact same ``config.json`` format that every HF
model on the Hub uses (e.g. the gemma repo you looked at).

To copy: rename this file to ``configuration_vaswani.py``.
"""

from __future__ import annotations

from transformers import PretrainedConfig

# Pull special-token ids from the single source of truth (grammar/vocab.py) so
# the config and the tokenizer cannot disagree. These values get baked into
# config.json at save time, so a config reloaded via from_pretrained no longer
# needs the grammar module present.
from grammar.cfg_vocab import BOS, EOS, PAD, token_to_id

_TOKEN_IDS = token_to_id()


class VaswaniConfig(PretrainedConfig):
    """Hyperparameters for the original "Attention is All You Need" Transformer.

    The three knobs you asked to sweep are ``d_model``, ``n_heads`` and
    ``n_layers`` (a single depth shared by encoder and decoder, as in the paper's
    N=6). Everything else has a paper-faithful default but is overridable.

    ``model_type`` is the string HF writes into ``config.json`` under
    ``"model_type"``; it is how ``AutoConfig`` later recognizes the file.
    """

    model_type = "vaswani"

    # Maps our attribute names onto the names some generic HF code expects.
    # e.g. generation utilities sometimes read ``config.num_attention_heads``.
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
    }

    def __init__(
        self,
        vocab_size: int = len(_TOKEN_IDS),
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",          # paper uses ReLU in the FFN
        max_position_embeddings: int = 64,  # caps sequence length for the PE table
        layer_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = True,   # paper ties the three embedding matrices
        # --- special token ids: derived from grammar/vocab.py (cannot drift) --
        pad_token_id: int = _TOKEN_IDS[PAD],
        bos_token_id: int = _TOKEN_IDS[BOS],
        eos_token_id: int = _TOKEN_IDS[EOS],
        decoder_start_token_id: int = _TOKEN_IDS[BOS],  # decoder primed with <bos>
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

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )

        # ``is_encoder_decoder=True`` tells HF (generate(), Trainer, etc.) to treat
        # this as a seq2seq model: it expects an encoder, decoder, and cross-attn.
        # Pop it from kwargs first: on reload it arrives via config.json, and
        # passing it both there and explicitly would be a duplicate-keyword error.
        kwargs.pop("is_encoder_decoder", None)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_encoder_decoder=True,
            **kwargs,
        )

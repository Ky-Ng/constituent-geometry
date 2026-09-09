def apply_template_toy_model(
    source_text: str,
    translated_text: str
) -> str:
    """
    Apply translation template for GPT-2 toy models like `kylelovesllms/gpt2-2l2h128d10ep3lr01drop-shift`
    """
    return (
        "<bos> " +
        source_text +
        " <sep> " +
        translated_text +
        " <eos> "
    )

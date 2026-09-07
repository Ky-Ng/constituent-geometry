import torch
from transformer_lens.model_bridge import TransformerBridge

# model_name = "google/gemma-3-4b-it"
# to_translate = "this cat chases the dog"


def get_model_natural_answer(model_name: str, to_translate: str, target_language: str = "Japanese") -> str:

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": f"You are a helpful assistant. Respond only with the translation from English to {target_language}."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": to_translate},
            ]
        }
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerBridge.boot_transformers(model_name, device=device)

    text = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    tokens = model.to_tokens(text, prepend_bos=False)

    batch_size, prompt_len = tokens.shape
    assert batch_size == 1, "only allow runs with batch size 1"

    input_output_tokens: torch.Tensor = model.generate(
        tokens,
        max_new_tokens=64,
        do_sample=False,
        stop_at_eos=True,
        eos_token_id=model.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
    )

    print("Complete input output")
    print("#"*20)
    print(input_output_tokens.shape)
    print(model.to_str_tokens(input_output_tokens))
    print("#"*20)

    output_tokens = input_output_tokens[0, prompt_len:]  # type: ignore
    out_str = model.tokenizer.decode(
        output_tokens,
        skip_special_tokens=True
    ).strip()  # Remove trailing newline

    print("Extracted Output")
    print("#"*20)
    print(out_str)
    print(model.to_str_tokens(output_tokens))
    print("#"*20)
    return out_str


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate the model's natural (greedy) translation for a given English phrase."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="HuggingFace model name to load via TransformerBridge",
    )
    parser.add_argument(
        "--to-translate",
        required=True,
        help="English text to translate to Japanese/Head Final language",
    )
    parser.add_argument(
        "--target-language",
        required=True,
        help="Target language specified in System Prompt",
        default="Japanese"
    )
    args = parser.parse_args()

    get_model_natural_answer(
        args.model_name, args.to_translate, args.target_language)

import torch
from transformer_lens.model_bridge import TransformerBridge

# to_translate = "this cat chases the dog"
# translated = "この猫は犬を追いかける"
# model_name = "google/gemma-3-4b-it"


def visualize_tokenized(model_name: str, to_translate: str, translated: str, target_language: str = "Japanese") -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerBridge.boot_transformers(model_name, device=device)

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
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": translated}],
        }
    ]

    tokens = model.tokenizer.apply_chat_template(
        messages,
        tokenize=True,  # See in words what gets split
        # adding in the assistant response with the prefilled translation
        add_generation_prompt=False
    ).input_ids

    tokens_as_tensor = torch.tensor(tokens)

    print("#"*20)
    print(model.to_string(tokens_as_tensor))
    print("#"*20)

    print("#"*20)
    print(model.to_str_tokens(tokens_as_tensor))
    print("#"*20)

    print("#"*20)
    print("".join(model.to_str_tokens(tokens_as_tensor)))
    print("#"*20)

    print(model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # See in words what gets split
        # adding in the assistant response with the prefilled translation
        add_generation_prompt=False
    ))


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
        "--translated",
        required=True,
        help="Japanese text",
    )
    parser.add_argument(
        "--target-language",
        required=True,
        help="Target language specified in System Prompt",
        default="Japanese"
    )
    args = parser.parse_args()

    visualize_tokenized(
        model_name=args.model_name,
        to_translate=args.to_translate,
        translated=args.translated,
        target_language=args.target_language
    )

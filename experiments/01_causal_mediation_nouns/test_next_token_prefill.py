import torch
from transformer_lens.model_bridge import TransformerBridge

# prompt = "<bos> the dog that chases Betty pursues a cat <sep> a a a a"
prompt = "<bos> the dog that chases Betty pursues a cat <sep> boy boy boy boy"
model_name = "kylelovesllms/gpt2-2l2h128d10ep3lr01drop-shift"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TransformerBridge.boot_transformers(model_name, device=device)

tokens = model.to_tokens(prompt, prepend_bos=False)
print("Input Tokens", model.to_str_tokens(tokens))

logits = model(tokens)

next_token = logits.argmax(dim=-1).squeeze(dim=0)[-1].item()
tokens_list = tokens.squeeze(dim=0).tolist()

tokens_list.append(next_token)

tokens_with_new_token_as_str = model.to_str_tokens(torch.tensor(tokens_list))
print(tokens_with_new_token_as_str)
breakpoint()
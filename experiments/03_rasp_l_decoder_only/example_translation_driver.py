from hi_hf_translation_raspl import translate

prompt = "the dog chases a cat"
split_prompt: list[str] = prompt.split()
breakpoint()
translation = translate(hi_words=split_prompt, depth=1)

print(translation)
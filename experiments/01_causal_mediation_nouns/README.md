# 01_causal_mediation_nouns — Observations

## Goal
Set up Causal Mediation experiment by patching the entire residual stream

## Setup
- Model: `kylelovesllms/gpt2-2l2h128d10ep3lr01drop-shift` which is a toy trained model

## Results

## Notes

### Toy Model
```zsh
python experiments/01_causal_mediation_nouns/run.py \
--model_name kylelovesllms/gpt2-2l2h128d10ep3lr01drop-shift \
--original "<bos> the dog chases this cat <sep> the dog this cat chases <eos>" \
--counterfactual "<bos> the dog likes this cat <sep> the dog this cat likes <eos>" \
--out experiments/01_causal_mediation_nouns/figures/word_heatmaps 
```

### Gemma
- note I dropped the demonstrative from the cat one
```zsh
python experiments/01_causal_mediation_nouns/run.py \
--model_name google/gemma-3-4b-it \
--original "<bos><start_of_turn>user
You are a helpful assistant. Respond only with the translation from English to Japanese.

this cat chases the dog<end_of_turn>
<start_of_turn>model
猫は犬を追いかける<end_of_turn>" \
--counterfactual "<bos><start_of_turn>user
You are a helpful assistant. Respond only with the translation from English to Japanese.

this sister chases the brother<end_of_turn>
<start_of_turn>model
姉は兄を追いかける<end_of_turn>" \
--out experiments/01_causal_mediation_nouns/figures/word_heatmaps/gemma-3-4b-it
```

### Gemma:: `this cat chases the dog`
gemma3-4b-it
```zsh
python experiments/01_causal_mediation_nouns/model_natural_answer.py \
--model-name google/gemma-3-4b-it \
--to-translate "this cat chases the dog"
```

output: `この猫は犬を追いかける`

```zsh
python experiments/01_causal_mediation_nouns/tokenize_visualizer.py \
--model-name google/gemma-3-4b-it \
--to-translate "this cat chases the dog" \
--translated "この猫は犬を追いかける"
```
output tokenization: ['この', '猫', 'は', '犬', 'を', '追い', 'かける']

| この | 猫 | は | 犬 | を | 追い | かける |
|------|-----|------|-----|------|------------|-----|
| 0 | 1 | 2 | 3| 4 | 5 | 6|

| この | 猫 | は | 犬 | を | 追いかける | 。 |
|------|-----|------|-----|------|------------|-----|
| kono | neko | wa | inu | o | oikake-ru | . |
| this | cat | TOP | dog | ACC | chase-NPST | PUNC |

### Gemma:: `this sister chases the brother`

```zsh
python experiments/01_causal_mediation_nouns/model_natural_answer.py \
--model-name google/gemma-3-4b-it \
--to-translate "this sister chases the brother"
```
output: `姉は兄を追いかける`

'姉', 'は', '兄', 'を', '追い', 'かける',

```zsh
python experiments/01_causal_mediation_nouns/tokenize_visualizer.py \
--model-name google/gemma-3-4b-it \
--to-translate "this sister chases the brother" \
--translated "姉は兄を追いかける"
```

### Gemma:: `the sister chases this brother`

```zsh
python experiments/01_causal_mediation_nouns/model_natural_answer.py \
--model-name google/gemma-3-4b-it \
--to-translate "the sister chases this brother"
```

output: `姉は弟を追いかける`

| 姉 | は | 弟 | を | 追いかける |
|-----|------|--------|------|------------|
| ane | wa | otouto | o | oikake-ru |
| e.sister | TOP | y.brother | ACC | chase-NPST |
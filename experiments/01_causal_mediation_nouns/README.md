# 01_causal_mediation_nouns — Observations

## Goal
Set up Causal Mediation experiment by patching the entire residual stream

## Setup
- Model: `kylelovesllms/gpt2-2l2h128d10ep3lr01drop-shift` which is a toy trained model
- Model: `google/gemma-3-4b-it` used for real Japanese

## Results

## Experiments Run

### Toy Model
```zsh
python experiments/01_causal_mediation_nouns/run.py \
--model_name kylelovesllms/gpt2-2l2h128d10ep3lr01drop-shift \
--original "<bos> the dog chases this cat <sep> the dog this cat chases <eos>" \
--counterfactual "<bos> the dog likes this cat <sep> the dog this cat likes <eos>" \
--out experiments/01_causal_mediation_nouns/figures/word_heatmaps 
```

### Gemma
`that dog chases this cat` ~> `あの犬がこの猫を追いかける`

| idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| **token** | あの | 犬 | が | この | 猫 | を | 追い | かける |
| **romaji** | ano | inu | ga | kono | neko | o | oi | kakeru |
| **gloss** | that.DIST | dog | NOM | this.PROX | cat | ACC | chase | -NPST |

`that horse chases this cow` ~> `あの馬がこの牛を追いかける`

| idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| **token** | あの | 馬 | が | この | 牛 | を | 追い | かける |
| **romaji** | ano | uma | ga | kono | ushi | o | oi | kakeru |
| **gloss** | that.DIST | horse | NOM | this.PROX | cow | ACC | chase | -NPST |

## Alignment across positions

| idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| **role** | D_subj | N_subj | case | D_obj | N_obj | case | V | V |
| **constituent** | DP_subj | DP_subj | DP_subj | DP_obj | DP_obj | DP_obj | VP | VP |
| **differs across pair** | no | **yes** | no | no | **yes** | no | no | no |

- check tokenization matches between the two prompts
```zsh
python experiments/01_causal_mediation_nouns/tokenize_visualizer.py \
--model-name google/gemma-3-4b-it \
--to-translate "that dog chases this cat" \
--translated "あの犬がこの猫を追いかける"
```

```zsh
python experiments/01_causal_mediation_nouns/tokenize_visualizer.py \
--model-name google/gemma-3-4b-it \
--to-translate "that horse chases this cow" \
--translated "あの馬がこの牛を追いかける"
```

- causal intervention prompt
    - `--original` and `counterfactual` are taken from the `tokenize_visualizer.py` outputs from above

```zsh
python experiments/01_causal_mediation_nouns/run.py \
--model_name google/gemma-3-4b-it \
--original "<bos><start_of_turn>user
You are a helpful assistant. Respond only with the translation from English to Japanese.

that dog chases this cat<end_of_turn>
<start_of_turn>model
あの犬がこの猫を追いかける<end_of_turn>" \
--counterfactual "<bos><start_of_turn>user
You are a helpful assistant. Respond only with the translation from English to Japanese.

that horse chases this cow<end_of_turn>
<start_of_turn>model
あの馬がこの牛を追いかける<end_of_turn>" \
--out experiments/01_causal_mediation_nouns/figures/word_heatmaps/gemma-3-4b-it
```
# 01_causal_mediation_nouns — Observations

## Goal
Set up Causal Mediation experiment by patching the entire residual stream

## Setup
- Model: `kylelovesllms/gpt2-2l2h128d10ep3lr01drop-shift` which is a toy trained model
- Model: `google/gemma-3-4b-it` used for real Japanese

## Results

## Noun Swapping

### Toy Model
```zsh
python experiments/01_causal_mediation_nouns/run.py \
--model_name kylelovesllms/gpt2-2l2h128d10ep3lr01drop-shift \
--original "<bos> the dog chases this cat <sep> the dog this cat chases <eos>" \
--counterfactual "<bos> the dog likes this cat <sep> the dog this cat likes <eos>" \
--out experiments/01_causal_mediation_nouns/figures/word_heatmaps 
```

### Gemma Japanese
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

 Alignment across positions

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

`that dog chases this bird` ~> `저 개가 이 새를 쫓는다`

| idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| **token** | 저 | 개 | 가 | 이 | 새 | 를 | 쫓 | 는다 |
| **IPA** | tɕʌ | kɛ | ɡa | i | sɛ | ɾɯl | t͈ɕon | nɯn.da |
| **romaja** | jeo | gae | ga | i | sae | reul | jjot | neunda |
| **gloss** | that.DIST | dog | NOM | this.PROX | bird | ACC | chase | -NPST.DECL |

`that cow chases this sheep` ~> `저 소가 이 양을 쫓는다`

| idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| **token** | 저 | 소 | 가 | 이 | 양 | 을 | 쫓 | 는다 |
| **IPA** | tɕʌ | so | ɡa | i | jaŋ | ɯl | t͈ɕon | nɯn.da |
| **romaja** | jeo | so | ga | i | yang | eul | jjot | neunda |
| **gloss** | that.DIST | cow | NOM | this.PROX | sheep | ACC | chase | -NPST.DECL |

### Gemma Korean
- Ensure that tokenization is aligned between the two Korean tokenizations

```zsh
python experiments/01_causal_mediation_nouns/tokenize_visualizer.py \
--model-name google/gemma-3-4b-it \
--to-translate "that dog chases this mouse" \
--target-language "Korean" \
--translated "저 개가 이 쥐를 쫓는다"
```

```zsh
python experiments/01_causal_mediation_nouns/tokenize_visualizer.py \
--model-name google/gemma-3-4b-it \
--to-translate "that horse chases this cow" \
--target-language "Korean" \
--translated "저 말이 이 소를 쫓는다"
```

```zsh
python experiments/01_causal_mediation_nouns/tokenize_visualizer.py \
--model-name google/gemma-3-4b-it \
--to-translate "that cow chases this sheep" \
--target-language "Korean" \
--translated "저 소가 이 양을 쫓는다"
```

```zsh
python experiments/01_causal_mediation_nouns/tokenize_visualizer.py \
--model-name google/gemma-3-4b-it \
--to-translate "that cow chases this bird" \
--target-language "Korean" \
--translated "저 소가 이 새를 쫓는다"
```

```zsh
python experiments/01_causal_mediation_nouns/tokenize_visualizer.py \
--model-name google/gemma-3-4b-it \
--to-translate "that dog chases this bird" \
--target-language "Korean" \
--translated "저 개가 이 새를 쫓는다"
```

- Misalignment
dog mouse:  '저', ' 개',   '가',  ' 이', ' ', '쥐', '를', ' ', '쫓', '는다'
horse cow:  '저', ' 말이',        ' 이',     ' 소', '를'  ' ', '쫓', '는다'
cow sheep:  '저', ' 소',   '가',  ' 이',     ' 양', '을', ' ', '쫓', '는다'
cow bird:   '저', ' 소',   '가',  ' 이',     ' 새', '를', ' ', '쫓', '는다'
dog bird:   '저', ' 개',   '가',  ' 이',     ' 새', '를', ' ', '쫓', '는다'

- Use `cow + sheep` and `dog bird` prompts; notes the accusative marker changes based on the vowel

```zsh
python experiments/01_causal_mediation_nouns/run.py \
--model_name google/gemma-3-4b-it \
--original "<bos><start_of_turn>user
You are a helpful assistant. Respond only with the translation from English to Korean.

that dog chases this bird<end_of_turn>
<start_of_turn>model
저 개가 이 새를 쫓는다<end_of_turn>" \
--counterfactual "<bos><start_of_turn>user
You are a helpful assistant. Respond only with the translation from English to Korean.

that cow chases this sheep<end_of_turn>
<start_of_turn>model
저 소가 이 양을 쫓는다<end_of_turn>" \
--out experiments/01_causal_mediation_nouns/figures/word_heatmaps/gemma-3-4b-it/korean
```

## Determiner Swapping
### Toy Model
```zsh
python experiments/01_causal_mediation_nouns/run.py \
--model_name kylelovesllms/gpt2-2l2h128d10ep3lr01drop-shift \
--original "<bos> the dog chases this cat <sep> the dog this cat chases <eos>" \
--counterfactual "<bos> a dog chases this cat <sep> a dog this cat chases <eos>" \
--out experiments/01_causal_mediation_nouns/figures/word_heatmaps 
```

```zsh
python experiments/01_causal_mediation_nouns/run.py \
--model_name kylelovesllms/gpt2-2l2h128d10ep3lr01drop-shift \
--original "<bos> the dog chases this cat <sep> the dog this cat chases <eos>" \
--counterfactual "<bos> the dog chases a cat <sep> the dog a cat chases <eos>" \
--out experiments/01_causal_mediation_nouns/figures/word_heatmaps 
```

## Verb Swapping with Intransitives
### Toy Model
```zsh
python experiments/01_causal_mediation_nouns/run.py \
--model_name kylelovesllms/gpt2-2l2h128d10ep3lr01drop-shift \
--original "<bos> the dog swims <sep> the dog swims <eos>" \
--counterfactual "<bos> the dog dances <sep> the dog dances <eos>" \
--out experiments/01_causal_mediation_nouns/figures/word_heatmaps 
```

```zsh
python experiments/01_causal_mediation_nouns/run.py \
--model_name kylelovesllms/gpt2-2l2h128d10ep3lr01drop-shift \
--original "<bos> the dog that likes this cat swims <sep> the this cat likes that dog swims <eos>" \
--counterfactual "<bos> the dog that likes this cat dances <sep> the this cat likes that dog dances <eos>" \
--out experiments/01_causal_mediation_nouns/figures/word_heatmaps 
```

## Swap Relative Clauses
### Toy Model
```zsh
python experiments/01_causal_mediation_nouns/run.py \
--model_name kylelovesllms/gpt2-2l2h128d10ep3lr01drop-shift \
--original "<bos> the dog that chases a cat that pursues this researcher swims <sep> the a this researcher pursues that cat chases that dog swims <eos>" \
--counterfactual "<bos> the dog that pursues this researcher that chases a cat swims <sep> the this a cat chases that researcher pursues that dog swims <eos>" \
--out experiments/01_causal_mediation_nouns/figures/word_heatmaps 
```
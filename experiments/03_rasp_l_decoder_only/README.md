# 03_rasp_l_decoder — Observations

- Note: in Claude.md, I have stated that AI should not write code. This is the exception. `you` below refers to the AI agents who will write this RASP-L algorithm for me to evaluate.

## Experiment Design
The goal is to write a RASP-L program that uses only the [RASP-L](https://github.com/apple/ml-np-rasp) `lib.py` and `core.py` to solve translation from Head Initial English to Head Final English (Zapanese).

You may not view or modify files unless specified.

You may view but not modify:
1. `experiments/03_rasp_l_decoder_only/rasp_core/core.py`
2. `experiments/03_rasp_l_decoder_only/rasp_core/lib.py`
3. `experiments/03_rasp_l_decoder_only/grammar/cfg_vocab.py`
4. `experiments/03_rasp_l_decoder_only/grammar/generate_with_frames.py`
5. `experiments/03_rasp_l_decoder_only/references/_s2_closer.py`
    - Note: `_s2_closer.py` is a reference implementation for a bidirectional attention encoder-only, seq2seq algorithm using a modified RASP-L lib which has not yet been hand-verified, only AI verified

You may view and modify:
1. `experiments/03_rasp_l_decoder_only/hi_hf_translation_raspl.py`
2. `experiments/03_rasp_l_decoder_only/testing/translation_unit_tests.py`
3. `experiments/03_rasp_l_decoder_only/README.md`
4. `experiments/03_rasp_l_decoder_only/testing/validation.py`

If you are to use web search or other papers, you must cite your references in the references section.

## Approach
Dataset: [kylelovesllms/hi-hf-v2-frames-1.3M-d2-k2-random](https://huggingface.co/datasets/kylelovesllms/hi-hf-v2-frames-1.3M-d2-k2-random)

Translation pairs must use the following format

```
<bos> [hi column from dataset] <sep> [hf column from dataset] <eos>
```

## Goal
Write a RASP-L algorithm that uses causal masking, faithful to the GPT-2 Decoder only architecture, which when given head initial tokens (i.e. `<bos> [hi column from dataset] <sep>`), autoregressively generates the correct head final tokens (i.e. `[hf column from dataset] <eos>`). The RASP-L program may only receive information from the `hi` and `hf` column formatted with the `<bos>`, `<sep>`, and `<eos>` token.

## Engineering Approach
1. You should first start off with being able to translate minimal examples of specific types of phrases (e.g. verb phrases, noun phrases, single relative clauses) and document your progress in the unit tests file `experiments/03_rasp_l_decoder_only/translation_unit_tests.py`
    - Note: it is possible to disambiguate a words lexical category based on the words that follow it
        - `that` is used as both a relative clause head and a complementizer phrase head which can be disambiguated based on what type of clause follows
        - certain VP.CP and VP.DP ("like", "hates") may be ambiguous which type of verb it is. This can be resolved by looking one token (or two tokens if there is an adjective/adverb) over to see if the next word is `that` or is a `noun`
        - hardcoded POS lexicons are allowed for this exercise but must follow the decoder only architecture with causal attention (`causal` must always be true)

2. You should then extend the algorithm to handle arbitrary depth recursion of sentences (e.g. `<bos> Jia pursues the pitiful musician that knows happily a cat that sings eagerly <sep> Jia the a eagerly sings that cat happily knows that pitiful musician pursues <eos>`). 
    - The data provided in the HF dataset has at max 2 depth. However, I will verify later with deeper depth sentences
    - Note: a recursive depth d may require O(d) layers/length RASP-L program. The RASP-L program may take in `d` the maximum depth/recursion of a sentence and return a RASP program of `d` long (e.g. using a loop that repeats the same behavior)

3. The goal of this is a pedagogical exercise to explain potential algorithms a real decoder model employs to solve real-life translation. Thus, you should make the code as readable as possible, abstract into functions when needed, and provide a first principles writeup below with concrete minimal examples.

4. When developing the RASP-L algorithm, including writing your own unit tests, use the train split. When you have passed your own unit tests, you may test against the validation split; the validation split can only be used in `experiments/03_rasp_l_decoder_only/testing/validation.py`.

5. `run.py` will not be used

## Follow ups
- later follow ups will be on evaluating whether a SGD trained transformer implements this algorithm

## Algorithm

### DP Parsing

### VP Parsing (VP.DP, VP.CP, VP.Intrans)

... TODO fill in


## References
- Fill in any references you use
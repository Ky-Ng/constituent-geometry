## Motivation
- Linguistics have a priori assumptions that language is represented in hierarchical and compositional binary trees which are `headed` in *human language*. However, we should not have commitments to how language is represented in artificial language models like LLMs
- Investigate the geometry of LLMs, specifically the residual stream, to see if/how syntactic relationships are represented

## Relevant Papers
1. Hewitt and Manning 2019: [A Structural Probe for Finding Syntax in Word Representations](https://www-nlp.stanford.edu/pubs/hewitt2019structural.pdf)
2. Manning et al. 2020:  [Emergent linguistic structure in artificial neural networks trained by self-supervision](https://www.pnas.org/doi/10.1073/pnas.1907367117)
3. Gurnee et al. 2025: [When Models Manipulate Manifolds: The Geometry of a Counting Task](https://arxiv.org/abs/2601.04480)

## Proposed Experiment
1. Train a language model on a task with a toy Context Free Grammar (CFG) with a translation objective between:
	1. Head Initial CFG (HICFG): simple CFG for English sentences
	2. Head Final CFG (HFCFG): The exact same CFG as the HICFG except that each non-terminal symbol is reversed
2. Probe the QK and OV circuits of the model
3. Probe the geometry of the residual stream
### Hypothesis
1. In an transformers, token to token information flow is bottle-necked by the attention mechanism which makes this a key place to investigate information flow
2. In this toy task, a linguist could create an algorithm that (1) marks the constituency of the sentence (2) reverse the order of the constituents
	1. The goal is to test if models perform well on this task, do they represent the sentence's syntax that would allow for such (1) constituency marking and (2) rotation of the constituents in space, perhaps in the residual stream

### Implementation Details
1. Decoder Only
	1. Each sequence is broken up into `<HI><TRANSLATE_TOKEN><HF>`
		1. HI is Head Initial
		2. HF is Head Final
	2. Run a parallel Head Final to Head Initial translation`<HF><TRANSLATE_TOKEN><HI>`
2. The architecture should be faithful to GPT-2 to later generalize to non-toy models with a larger vocabulary and hidden dimension size
3. Dimensions/Sizes
	1. N Layers, start with N=2 for the most toy example
	2. n_heads = 1 to isolate the information flow across positions
	3. d_model=3
	4. Note: if the model performance suffers, this will be a case for increasing the dimensionality, layers, or sizes
4. Do not use tied weights to see cleanly separate information flow

## Probing Methods
1. QK and OV Circuits
	1. QK circuit: test if locally, each head and complement attend to each other, and that one daughter of each constituent (say the head) is looking at its neighboring constituent
	2. OV circuit: test if what is being written out is the logit aligned to the unembedding matrix for the translated token at that position 
2. Residual Stream: Plot L2 Distances of each token position at each layer to recover a Minimum Spanning Tree (MST)
3. Attention: Plot the QKV projections using the W_q W_k W_v matrices in space and view over time
## Implications
1. A future goal is to expand such experiments to (1) more complete syntactic descriptions of languages and (2) "in the wild" models 
2. Limitation: languages are not purely context sensitive, likely there are more complicated relationships that this experiment would overlook 

## CFG Rules
- CFG Rules are listed in `unambiguous_cfg_flat.md`
- For all terminal symbols, rather than using real words, please use `TerminalSymbol_N` as the input (for example)
	- `n1`, `n2`, `n3`, `n4` for nouns

## CFG Grammar 
### Head Initial Grammar
```python
# Non terminal Rules
S -> DP VP

DP -> D NP

NP -> N

VP -> V_trans DP | V_intrans

# Terminal symbols
D -> d1 | d2 | d3 | d4 | d5

N -> n1 | n2 | n3 | n4 | n5

V_trans -> V_trans1 | V_trans2 | V_trans3 | V_trans4

V_intrans -> V_intrans1 | V_intrans2 | V_intrans3 | V_intrans4
```

### Head Final Grammar
```python
# Non terminal Rules
S -> DP VP

DP -> NP D

NP -> N

VP -> DP V_trans | V_intrans

# Terminal symbols
D -> d1 | d2 | d3 | d4 | d5

N -> n1 | n2 | n3 | n4 | n5

V_trans -> V_trans1 | V_trans2 | V_trans3 | V_trans4

V_intrans -> V_intrans1 | V_intrans2 | V_intrans3 | V_intrans4
```

### Output CSV

Columns: `example_number, hi_tree, hf_tree, hi_surface, hf_surface`.

Trees are stored in NLTK / Penn Treebank bracket notation (parentheses, space-separated) so they can be parsed directly via `nltk.Tree.fromstring(...)` and visualized with `pretty_print`, `draw`, or `svgling`. Surface strings are the yielded terminals, space-separated. The skeleton (structure-without-terminals) is intentionally not stored .

```csv
example_number,hi_tree,hf_tree,hi_surface,hf_surface
1,(S (DP (D d1) (NP (N n1))) (VP (V_trans V_trans1) (DP (D d2) (NP (N n2))))),(S (DP (NP (N n1)) (D d1)) (VP (DP (NP (N n2)) (D d2)) (V_trans V_trans1))),d1 n1 V_trans1 d2 n2,n1 d1 n2 d2 V_trans1
```
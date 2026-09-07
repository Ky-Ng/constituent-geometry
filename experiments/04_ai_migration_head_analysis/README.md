# 04_ai_migration_head_analysis — Observations

Code and runs in this folder were written by an AI agent (see `rules.md`).

## Goal
Exp 01 patched the whole residual stream and saw noun identity "migrate" from the head-initial (HI) noun
position to the head-final (HF) slot inside layer 1. That cannot say *which head* does it. Here every
attention head is patched individually, at five sites (`z`, `q`, `k`, `v`, `pattern`), at every position
and at all positions, and the effect is read as a **logit difference** at a single critical position.

## Setup
- Model: `kylelovesllms/gpt2-2l2h128d10ep3lr01drop-shift` (2 layers x 2 heads, d_model 128, learned absolute positions, v2 vocab).
- Pair: original `<bos> the A chases D B <sep> the A D B chases <eos>` vs counterfactual with `A -> C`, `B -> D`.
- Critical position: first HF token `the` (index of `<sep>` + 1 = position 7), where the model predicts `A` vs `C`.
- Metric at the critical position:
  `logit_diff = logit(A) - logit(C)`;
  `normalized_effect = (ld_patched - ld_clean) / (ld_counterfactual - ld_clean)` (0 = clean run, 1 = fully counterfactual).
- Sites and what the position index means (`head_patching.py`):
  `z` head output at the query position, `q` query at the query position, `k`/`v` key/value at the **source** position,
  `pattern` the attention row of the query position.
- Files: `run.py` (single-pair CLI), `sweep.py` (lexical sweep over pair sets, see "Lexical sweep" below), `attention_patterns.py` (full attention matrices per head for a pair),
  `head_patching.py` (hooks + sweep), `metrics.py`, `plot.py`, `test_head_patching.py`
  (semantics checks: whole-head `z` patch reproduces the counterfactual attention output; single-position patch is local;
  `pattern` x `v` patch reproduces `z`; self-patching is a no-op).

```zsh
# tests (pytest is a dev extra; --with keeps .venv untouched)
HF_HOME=/scratch1/kgng/hf_cache HF_HUB_OFFLINE=1 uv run --with pytest pytest experiments/04_ai_migration_head_analysis/test_head_patching.py -q

# run 1: the user's spec, repeated `the`
HF_HOME=/scratch1/kgng/hf_cache HF_HUB_OFFLINE=1 uv run python experiments/04_ai_migration_head_analysis/run.py \
  --original "<bos> the dog chases the cat <sep> the dog the cat chases <eos>" \
  --counterfactual "<bos> the researcher chases the dancer <sep> the researcher the dancer chases <eos>" \
  --tag the_the_both_nouns --group 00_single_pair

# run 2: mixed determiners (exp 01 running example)
HF_HOME=/scratch1/kgng/hf_cache HF_HUB_OFFLINE=1 uv run python experiments/04_ai_migration_head_analysis/run.py \
  --original "<bos> the dog chases this cat <sep> the dog this cat chases <eos>" \
  --counterfactual "<bos> the researcher chases this dancer <sep> the researcher this dancer chases <eos>" \
  --tag the_this_both_nouns --group 00_single_pair

# run 3: subject noun only (control)
HF_HOME=/scratch1/kgng/hf_cache HF_HUB_OFFLINE=1 uv run python experiments/04_ai_migration_head_analysis/run.py \
  --original "<bos> the dog chases this cat <sep> the dog this cat chases <eos>" \
  --counterfactual "<bos> the researcher chases this cat <sep> the researcher this cat chases <eos>" \
  --tag the_this_subject_only --group 00_single_pair
```

Outputs: `results/<group>/<id>_<tag>/{descrip.json, <site>.json, whole_head_summary.csv}`,
`figures/<group>/<id>_<tag>/{<site>.png, whole_head_summary.png}`, logs in `logs/<group>/<tag>.log`.

## Results

All three runs give the same numbers to two decimals, so the determiner choice and the object noun do not
matter for this readout. Numbers below are run 2 (`e7a376e1_the_this_both_nouns`).
Clean `logit_diff = 18.8`, counterfactual `= -23.5`.

### Whole-head patches (all positions at once), normalized effect

| head | z | v | q | k | pattern |
|---|---|---|---|---|---|
| L0 H0 | -0.01 | -0.01 | 0.00 | 0.00 | 0.00 |
| L0 H1 | 0.24 | 0.24 | 0.01 | 0.00 | 0.00 |
| L1 H0 | 0.29 | 0.29 | 0.00 | 0.00 | 0.00 |
| L1 H1 | **0.61** (top-1 flips to `researcher`) | 0.60 | 0.00 | 0.00 | 0.00 |
| L1 H0+H1 | **1.00** | 1.00 | 0.00 | 0.00 | 0.00 |

### Per-position patches (`z` and `v`), normalized effect; only non-zero cells shown

| head | site | position | effect |
|---|---|---|---|
| L1 H1 | z | 7 `the` (critical) | 0.61 |
| L1 H1 | v | 2 `dog` (HI subject noun) | 0.62 |
| L1 H0 | z | 7 `the` (critical) | 0.29 |
| L1 H0 | v | 3 `chases` (HI verb) | 0.26 |
| L0 H1 | z | 3 `chases` (HI verb) | 0.26 |
| L0 H1 | v | 2 `dog` (HI subject noun) | 0.24 |

Every `q`, `k`, and `pattern` cell is within +/-0.01 of zero.

### Attention rows of the implicated heads (identical for both prompts)

Full matrices (every head, original vs counterfactual vs difference, critical query row outlined) are in
`figures/00_single_pair/e7a376e1_the_this_both_nouns/attention/attention_L{0,1}.png`, produced by
`attention_patterns.py`; the rows below are read off those figures (`attention_rows.json` alongside).


| head, query | top sources |
|---|---|
| L1 H1 @ 7 `the` | 2 noun 0.71, 4 `this` 0.15 |
| L1 H0 @ 7 `the` | 3 `chases` 0.81, 4 `this` 0.10 |
| L0 H1 @ 3 `chases` | 0 `<bos>` 0.40, 1 `the` 0.21, 2 noun 0.20, 3 `chases` 0.20 |

Same heads in the other two frames (one pair each; figures under `figures/04_intransitive/attention_dog_researcher_swims/`
and `figures/05_proper_subject/attention_John_Mary_chases/`):

| frame | head, query | top sources |
|---|---|---|
| intransitive `the dog swims` | L1 H1 @ 5 `the` | 2 noun 0.69, 3 `swims` 0.22 |
| intransitive `the dog swims` | L1 H0 @ 5 `the` | 3 `swims` 0.81, 2 noun 0.12 |
| proper `John chases this cat` | L1 H1 @ 5 `<sep>` | 1 name 0.89, 0 `<bos>` 0.09 |
| proper `John chases this cat` | L1 H0 @ 5 `<sep>` | 1 name 0.72, 0 `<bos>` 0.09, 4 `cat` 0.07 |
| proper `John chases this cat` | L0 H1 @ 2 `chases` | 0 `<bos>` 0.22, 3 `this` 0.21 ... (row at query 2 is 0.20 on the name) |

So L1 H0's attention really does move with the frame: onto the verb when the subject is a determiner phrase,
onto the name itself when the subject is a bare proper noun. It is not a fixed offset from the critical position.

## Observations

1. **Two paths carry the noun, and together they account for everything.**
   - Direct path: **L1 H1** at the critical position attends 0.71 to the HI noun and its *value* there carries 0.62 of the effect.
   - Indirect path: **L0 H1** at the verb position copies the noun into the verb's residual (its value at pos 2 -> its output at
     pos 3, 0.24-0.26); then **L1 H0** at the critical position attends 0.81 to the verb and reads the noun back out (0.26-0.29).
   - Patching both layer-1 heads together gives 1.00. Individually 0.61 + 0.29 = 0.90; the gap is the nonlinearity of the
     final LN/MLP, so "fractions" should be read loosely.
2. **Routing is content-independent here.** `q`, `k`, and `pattern` patches do nothing, and the attention rows are the same for
   both prompts. Noun identity moves through the value/output stream; the queries and keys implement a fixed
   position-to-position map (consistent with an offset-from-`<sep>` account). Caveat: both prompts have identical structure and
   length, so this says nothing about whether routing changes when structure changes (that needs a structural counterfactual,
   e.g. transitive vs intransitive, design doc 05).
3. **Why exp 01's residual heatmaps looked the way they did.** The noun-swap heatmap flipped when patching the HI noun position
   up to L1 resid_pre and then the HF slot from L1 resid_mid: that is L1 H1's hop. The verb-position copy made by L0 H1 is why
   a single-position residual patch at the verb position also mattered in the transitive verb-swap heatmap and why after layer 0
   the information looked "spread out".
4. **A top-1 flip metric would have missed L1 H0.** Patching it alone leaves the top-1 at `dog` while moving the logit diff by 12
   logits (0.29 normalized). The logit-diff readout is the right choice.

## Caveats / next steps
- Single noun pair (`dog/cat` vs `researcher/dancer`). The 0.6 / 0.3 split should be confirmed over the `N_singular` grid
  (16 x 15 ordered pairs) before it is quoted; the IIA/pair-sampling plan in design doc 04 does this.
- The per-position `z` rows for layer 1 are concentrated at the critical position *by construction* (last layer, readout at
  that position). The informative per-position rows are `v` (where the head reads from) and layer-0 `z` (where layer 0 writes).
- Per-cell forward passes (~330 per run) are fine here; Gemma needs the batching from design doc 04 or a two-stage sweep.

## Extension to Gemma-3-4B-IT (Japanese) - what changes
- **Critical position must be passed explicitly** (`--critical_pos`): there is no `<sep>`; it is the token before the first
  noun of the translation (e.g. `あの` predicting `犬`). Tokenization alignment between the pair must be checked first
  (exp 01's `tokenize_visualizer.py`), and the answer tokens must be single tokens.
- **Grouped-query attention.** Gemma-3-4B has 8 query heads but 4 key/value heads (`config.json` omits them; confirm with
  `model.cfg.n_heads` / `n_key_value_heads` on the GPU node). `hook_q`/`hook_z` index 8 heads, `hook_k`/`hook_v` index 4, and each
  k/v head feeds two query heads, so a per-head `k`/`v` patch is really a per-*pair* patch. `make_patch_hook` indexes the head
  dimension generically and will run, but the summary tables need the two head index spaces kept apart.
- **Scale.** 34 layers x 8 heads x ~45 positions x 5 sites is ~60k forward passes of a 4B model. Do whole-head `z` first
  (34 x 8 = 272 passes), then per-position `v`/`z` only for the top heads, and batch positions (doc 04).
- Needs a GPU job (`slurm/run_gpu.sbatch`); the toy runs above were CPU on a login node.
- `plot.py` sets a CJK-capable font first in the fallback list for Japanese tick labels.

## Lexical sweep

Is the 0.6 / 0.3 split above a property of `dog/cat` vs `researcher/dancer` with `chases`, or of the model?
`sweep.py --set <set>` (seed 0) runs the identical five-site sweep over a *set* of pairs that vary one lexical
dimension at a time and aggregates every cell across pairs: mean ± sample std (ddof=1) of the normalized effect,
and a **hit rate** = fraction of pairs whose patched top-1 at the critical position is the counterfactual answer
(IIA-style). A pair is skipped if the clean top-1 at the critical position is not the answer (or the counterfactual
clean top-1 is not its own answer); no pair was skipped in any set.

```zsh
for s in 01_nouns 02_verbs 03_determiners 04_intransitive 05_proper_subject; do
  HF_HOME=/scratch1/kgng/hf_cache HF_HUB_OFFLINE=1 uv run python experiments/04_ai_migration_head_analysis/sweep.py --set $s
done
```

Outputs: `results/<set>/{aggregate.json, aggregate.csv, pairs.json}` plus `results/<set>/<pair_id>/` in the `run.py`
layout, `figures/<set>/{<site>.png, whole_head_summary.png}` (cell colour = mean, text = mean±std; the summary also
prints the hit rate), `logs/<set>/sweep.log`. Position labels are role names (`N1`, `V`, `D1`, ...) where the slot
varies within the set and the literal token where it does not; `*` marks the critical position.

| set | frame (both prompts share it) | what varies across pairs | used / skipped | clean `ld` mean (min–max) | cf `ld` mean |
|---|---|---|---|---|---|
| `01_nouns` | transitive, `V=chases`, `D1=the`, `D2=this` | 30 ordered `(N1, N1')` pairs from `N_singular` (no replacement); `N2`, `N2'` drawn from the remaining nouns | 30 / 0 | 20.2 (15.1–33.5) | -20.8 |
| `02_verbs` | transitive, `dog/cat` vs `researcher/dancer`, `D1=the`, `D2=this` | `V` over the 11 unambiguous transitive verbs (same verb in both prompts) | 11 / 0 | 19.2 (17.9–20.4) | -24.1 |
| `03_determiners` | transitive, same nouns, `V=chases` | `(D1, D2)` over `{the, a, this}^2` (same in both prompts) | 9 / 0 | 18.7 (18.3–18.9) | -23.3 |
| `04_intransitive` | `<bos> the N1 V <sep> the N1 V <eos>`, critical = pos 5 (`the`) | 16 `(N1, N1')` pairs x 8 intransitive verbs (each verb twice) | 16 / 0 | 15.3 (9.6–27.3) | -15.3 |
| `05_proper_subject` | `<bos> N1 chases this N2 <sep> N1 this N2 chases <eos>`, critical = pos 5 (`<sep>` itself) | 16 `(N1, N1')` pairs from `N_proper`; `N2`, `N2'` from `N_singular` | 16 / 0 | 19.8 (14.2–36.3) | -20.1 |

### Key cells per set (mean ± std of normalized effect, hit rate)

`N1` = head-initial subject-noun position (2; 1 in set 05), `V` = head-initial verb position (3; 2 in set 05).
The last row is the largest |mean| over *all* `q`, `k`, `pattern` cells (every position and ALL).

**`01_nouns` (n = 30)**

| cell | effect | hit |
|---|---|---|
| L1 H1 z @ ALL | 0.67 ± 0.05 | 1.00 |
| L1 H0 z @ ALL | 0.32 ± 0.04 | 0.00 |
| L1 H0+1 z @ ALL | 1.00 ± 0.00 | 1.00 |
| L0 H1 z @ ALL | 0.27 ± 0.04 | 0.00 |
| L1 H1 v @ 2_N1 | 0.68 ± 0.05 | 1.00 |
| L1 H0 v @ 3_V (`chases`) | 0.28 ± 0.04 | 0.00 |
| L0 H1 v @ 2_N1 | 0.27 ± 0.04 | 0.00 |
| L0 H1 z @ 3_V (`chases`) | 0.29 ± 0.04 | 0.00 |
| max abs over q/k/pattern (q L0 H1 @ 2_N1) | 0.00 ± 0.00 | 0.00 |

**`02_verbs` (n = 11)**

| cell | effect | hit |
|---|---|---|
| L1 H1 z @ ALL | 0.61 ± 0.01 | 1.00 |
| L1 H0 z @ ALL | 0.30 ± 0.01 | 0.00 |
| L1 H0+1 z @ ALL | 1.00 ± 0.00 | 1.00 |
| L0 H1 z @ ALL | 0.25 ± 0.01 | 0.00 |
| L1 H1 v @ 2_N1 | 0.63 ± 0.01 | 1.00 |
| L1 H0 v @ 3_V | 0.26 ± 0.01 | 0.00 |
| L0 H1 v @ 2_N1 | 0.24 ± 0.01 | 0.00 |
| L0 H1 z @ 3_V | 0.27 ± 0.01 | 0.00 |
| max abs over q/k/pattern (q L0 H0+1 @ ALL) | 0.01 ± 0.00 | 0.00 |

**`03_determiners` (n = 9)**

| cell | effect | hit |
|---|---|---|
| L1 H1 z @ ALL | 0.60 ± 0.00 | 1.00 |
| L1 H0 z @ ALL | 0.29 ± 0.00 | 0.00 |
| L1 H0+1 z @ ALL | 1.00 ± 0.00 | 1.00 |
| L0 H1 z @ ALL | 0.25 ± 0.00 | 0.00 |
| L1 H1 v @ 2_N1 | 0.62 ± 0.00 | 1.00 |
| L1 H0 v @ 3_V (`chases`) | 0.26 ± 0.00 | 0.00 |
| L0 H1 v @ 2_N1 | 0.24 ± 0.00 | 0.00 |
| L0 H1 z @ 3_V (`chases`) | 0.27 ± 0.00 | 0.00 |
| max abs over q/k/pattern (q L0 H0+1 @ ALL) | 0.01 ± 0.00 | 0.00 |

**`04_intransitive` (n = 16)**

| cell | effect | hit |
|---|---|---|
| L1 H1 z @ ALL | 0.70 ± 0.06 | 0.81 |
| L1 H0 z @ ALL | 0.31 ± 0.05 | 0.00 |
| L1 H0+1 z @ ALL | 1.00 ± 0.00 | 1.00 |
| L0 H1 z @ ALL | 0.20 ± 0.03 | 0.00 |
| L1 H1 v @ 2_N1 | 0.72 ± 0.06 | 0.88 |
| L1 H0 v @ 3_V | 0.21 ± 0.04 | 0.00 |
| L0 H1 v @ 2_N1 | 0.20 ± 0.03 | 0.00 |
| L0 H1 z @ 3_V | 0.20 ± 0.03 | 0.00 |
| max abs over q/k/pattern (k L0 H0+1 @ ALL) | 0.01 ± 0.01 | 0.00 |
| *extra:* L1 H0 v @ 2_N1 | 0.07 ± 0.02 | 0.00 |

**`05_proper_subject` (n = 16)**

| cell | effect | hit |
|---|---|---|
| L1 H1 z @ ALL | 0.64 ± 0.03 | 1.00 |
| L1 H0 z @ ALL | 0.39 ± 0.06 | 0.00 |
| L1 H0+1 z @ ALL | 1.00 ± 0.01 | 1.00 |
| L0 H1 z @ ALL | 0.03 ± 0.01 | 0.00 |
| L1 H1 v @ 1_N1 | 0.64 ± 0.03 | 1.00 |
| L1 H0 v @ 2_V (`chases`) | 0.02 ± 0.00 | 0.00 |
| L0 H1 v @ 1_N1 | 0.03 ± 0.01 | 0.00 |
| L0 H1 z @ 2_V (`chases`) | 0.02 ± 0.00 | 0.00 |
| max abs over q/k/pattern (q L0 H1 @ ALL) | -0.01 ± 0.00 | 0.00 |
| *extra:* L1 H0 v @ 1_N1 | 0.34 ± 0.05 | 0.00 |

### Observations

1. **The split is a property of the model, not of the pair, in the transitive frame.** Over 30 noun pairs, 11 verbs and
   9 determiner patterns the same cells light up and nothing else does: L1 H1 direct path 0.60–0.68, L1 H0 via the verb
   0.29–0.32, L0 H1 verb-copy 0.25–0.27, both layer-1 heads together 1.00 ± 0.00. Only the nouns add variance
   (std 0.04–0.05); verbs and determiners move nothing by more than 0.01. The single-pair numbers (0.61 / 0.29) sit at the
   low end of the noun distribution (`01_nouns`: 0.67 ± 0.05 / 0.32 ± 0.04). Every `q`, `k`, `pattern` cell has |mean|
   <= 0.01 in every set.
2. **Hit rate is binary by head.** L1 H1 alone flips the top-1 in every pair of sets 01–03 and 05 (hit 1.00) while L1 H0
   alone never does (0 of 82 pairs) despite moving 0.3–0.4 of the logit difference - observation 4 above, now over 82 pairs.
3. **Intransitive frame (`04`): same heads, same two paths, weight shifted to the direct path.** L1 H1 carries 0.70 ± 0.06
   and the verb relay weakens (L0 H1 copy 0.20, L1 H0 read-out 0.21); L1 H0 now also reads a little straight from `N1`
   (v @ 2_N1 = 0.07). Clean margins are smaller (`ld` 9.6–27.3), and L1 H1 alone flips the top-1 in only 13/16 pairs.
4. **Proper-noun subject (`05`) is the one set where the mechanism changes.** The L0 H1 -> verb -> L1 H0 relay disappears
   (L0 H1 z @ ALL 0.03, L1 H0 v @ `chases` 0.02) and L1 H0 instead reads the noun *directly* from position 1
   (v @ 1_N1 = 0.34 ± 0.05). Both layer-1 heads are then direct readers of `N1` (0.64 + 0.39, together 1.00). So L1 H0 is
   not a "verb reader": the source it uses depends on the frame. It is not a fixed offset from the critical position
   either (it reads offset 4 in the transitive and proper-subject frames but offset 2 in the intransitive frame), so the
   routing rule is something structural that this sweep cannot resolve.
5. **Caveat unchanged.** Both prompts of every pair share the frame, so `q`/`k`/`pattern` ~ 0 still only says routing is
   independent of *lexical* content. The across-frame differences (3, 4) show it does depend on structure; attributing
   that to specific heads' queries/keys needs a structural counterfactual (patch across frames), as noted in observation 2.

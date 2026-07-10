# LOG — building a RASP-L program for HI → HF translation

A complete research/engineering log for `experiments/22_rasp_l_hi_hf`: how we went from
"translate the `hi` column to the `hf` column of `kylelovesllms/hi-hf-v2-frames-1.3M-d2-k2-random`"
to a verified **RASP-L** program, four independent implementations of its hard stage, and a
concrete theoretical result about why transformers can't depth-generalize on this task.

Read this alongside two companions:
* `displacement_handout.tex/.pdf` — a visual handout for the single most important idea
  (recursive reversal → additive displacement).
* the code: `rasp_l.py` (primitives), `grammar_oracle.py` (ground truth), `pos_tag.py`
  (stage 1), `_s2_{closer,interval,gaplca,twolevel}.py` (four stage-2 constructions).

---
---

# PART I — The research narrative (how the answer was found)

## 0. Framing and the one constraint that shaped everything

RASP-L (Zhou et al., *What Algorithms Can Transformers Learn?*, ICLR 2024; ref impl
`apple/ml-np-rasp`) is a tiny language where each operation is one transformer piece:

| RASP-L op | transformer piece |
|---|---|
| `tok_map` / `seq_map` (elementwise) | MLP / token embedding |
| `select` + `aggr` / `kqv` | one attention head |
| `sel_width` | one attention head (counting) |

The load-bearing constraint: `select` is **causal by default** (`kj ≤ qi`, a GPT-2 mask —
confirmed in `np_rasp/core.py`). RASP-L is the theory of *decoder-only* transformers. This
forced two decisions that recur throughout:

1. **Layout.** The task is a single stream `hi₁…hiₙ <sep> hf₁…hfₙ`. All HI features are
   computed while HI is entirely in the past of the output region, so a read that looks
   "forward" *within HI* (`causal=False` in the simulator) is realizable in a real causal
   decoder — it just happens at/after `<sep>`. Every `causal=False` in the code is one of these.
2. **What predicts length generalization** is *which* predicates a program uses: content +
   relative-position + counting generalize; unbounded bracket-matching does not. This is the
   thread that ends in Part III.

## 1. Reduction: HF is a permutation of HI

The first thing I checked in `src/grammar/v2/generate_with_frames.py`: every HI/HF rule pair
has the **same right-hand-side terminals**, only reordered. So **HF is a permutation of HI** —
same multiset, same length `n`. Translation collapses to "compute one permutation," and the
generator produces HF by reversing the two children of certain **flip** nodes, recursively:

* **flip** (children reverse HI→HF): `CP_sent=[C,S]`, `CP_rel=[C_rel,S_gap]`,
  `VP=[head-block, complement]`, the adverb blocks `VP_{intrans,dp,cp}_adv=[V,Adv]`, and
  `NP_singular=[coreNoun, CP_rel]`.
* **invariant** (keep order): `S=[subj,VP]`, `DP=[D,NP]`, `S_obj_gap=[DP,V]`,
  `NP_singular_adj=[Adj,N]`, all unary nodes.

## 2. Feasibility: does the flat HI string even determine HF? (yes, 100%)

RASP-L sees only the flat string, and it's lexically ambiguous: `that` is both `C` and
`C_rel`, and `believes/hates/knows/likes` are both `V_dp` and `V_cp`. Could two trees give
the same HI string? I wrote a one-token-lookahead recursive-descent parser
(`grammar_oracle.Parser`), flipped its tree, and compared to the generator's `hf`:
**100% exact on depths 0/1/2**, and the recovered HI tree matched the generator's bracketing
exactly. So HI is uniquely parseable → HF is a well-defined function of it → a RASP program
provably exists. `grammar_oracle.py` became our **ground truth** for everything after.

*(Side finding, verified separately: the dataset's `hf_frame_tagged` column is a valid
permutation but disagrees with the true node-identity permutation on ~all sentences with a
repeated token — it tags by `(word, occurrence)` independently per tree, which scrambles when
a flip reorders two identical tokens. Harmless for the HF **string**, but wrong if read as an
HI→HF index map, e.g. attention-arc plots.)*

## 3. The core trick: recursive reversal → **additive displacement**

Computing a full tree address per token would work but is heavy. The shortcut
(**this is the handout's subject**): nested reversals compose **additively**. A flip node lays
its children out as adjacent blocks `[H | C]` and swaps to `[C | H]`; so every head token
slides right by `|C|` and every comp token slides left by `|H|`. Nesting doesn't interfere —
an outer flip slides an inner span **rigidly**, adding a constant to every token inside — so:

```
hf_pos(x) = hi_pos(x) + Σ over enclosing flip-nodes F of
              ( +|comp(F)|  if x ∈ head child,   −|head(F)|  if x ∈ comp child )
```

This gives **exact integer target slots** (a true permutation, no sorting). **Verified 100%**
on 3,000 depth-2 frames. Everything downstream is: compute `disp`, then gather. (Full worked
examples with diagrams: `displacement_handout.pdf`.)

## 4. Program shape and the three stages

```
hi_ids ─stage1─▶ fine POS ─stage2─▶ depth & spans ─stage2─▶ hf_pos ─stage3─▶ hf_ids
```

I built stages 1 and 3 first (they're the easy, well-understood ends), which isolated the
one hard stage.

## 5. Stage 1 — POS is fully local (verified 100%)

Every "scary" disambiguation is decidable from a 3-token window:
* `that` → **C_rel** iff the previous token is an `N_singular`, else **C**.
* a verb → **V_cp** if the next non-adverb token is `that`; **V_dp** if it is a
  determiner/proper name; else **V_intrans** if the word is lexically intransitive; else
  **V_dp** — the leftover case is an *object-gap* verb (a V_dp-type verb with no complement,
  which can only be a gap: `the cat that the mouse fears`).

`pos_tag.py` = `tok_map` (word→lexical-class embedding) + three `read_rel` reads
(prev/next-1/next-2) + one `seq_map`. **100% on 3,000 depth-2 frames.** All ops are
content/relative-position → **length-generalizing.**

## 6. Stage 3 — the gather (verified 100% with oracle displacement)

`hf_pos = indices + disp`; invert the permutation and gather:
```python
src = kqv(hf_pos, indices, indices, equals, causal=False)   # slot j → the i with hf_pos[i]=j
out = index_select(hi_ids, src, causal=False)               # place HI tokens into HF order
```
Non-causal, but only over the HI block (past of the output region). Fed the **oracle's**
`hf_pos`, this reproduced HF **2000/2000**, so stage 3 was proven correct *independently of
stage 2*. That left exactly one unknown.

## 7. Stage 2 — the hard part: unmarked, coincident clause closes

`depth[i]` (number of enclosing CPs) is what tells each token which flip nodes enclose it and
how big their complements are. It is **not** a naive marked-bracket count:
* clause closings are **unmarked** in the surface string (no `)`), and
* an interior close can have **multiplicity 2** — a `CP_sent` nested at the right edge of a
  relative clause closes *together* with it (`the cat [that claims that the dog swims] likes X`:
  both close after `swims`).

Recovering these is **Dyck bracket-matching**. I couldn't be sure of nailing it in one
attempt, so I used the following methodology.

## 8. Methodology: verify-first + a parallel race of strategies

Two habits did the heavy lifting:
* **Ground-truth oracle.** Every intermediate s-op (POS, depth, displacement) was checked
  token-by-token against `grammar_oracle.oracle_features`, not just the final string.
* **Parallel strategy race.** For stage 2 I launched a background *workflow* of four
  independent agents, each attacking span-recovery with a different algorithm
  (closer-marking / interval-matching / pairwise-LCA / bounded-recursion), each required to
  verify to 100% against the oracle. I then **independently re-verified** all four myself
  (never trusting self-reports) on fresh samples and OOD depths. Three hit 100% on depth-2;
  see Part II. A second workflow adversarially reviewed the survivors (Part IV).

This is the general shape worth reusing: *reduce to a checkable target, build the easy ends
first to isolate the hard core, then attack the hard core several ways in parallel against a
ground truth.*

---
---

# PART II — The four stage-2 constructions

**All four compute the *same* additive displacement of §3** — `hf_pos = i + disp`, `disp` a
sum of `±|sibling|` over enclosing flip nodes. They are **not** different ideas from the
handout; they are four different ways to (a) recover the unmarked clause **spans** and (b)
express the displacement **sum**. Independently re-verified results:

| module | strategy | depth 0 | depth 1 | depth 2 | depth 3 | depth 4 | depth 5 |
|---|---|---|---|---|---|---|---|
| `_s2_closer`   | closer-marking + relaxation (K=10) | ✓ | ✓ | **✓** | ✓ | ✓ | ✓ |
| `_s2_interval` | that/verb balance + pointer-jump (6) | ✓ | ✓ | **✓** | ✓ | ✓ | ✓ |
| `_s2_twolevel` | 4 additive families + fix-point (40) | ✓ | ✓ | **✓** | ✓ | ✓ | ✓ |
| `_s2_gaplca`   | pairwise flip-LCA counting (K=6) | ✓ | ✓ | **✓** | ✗ (43%) | ✗ | ✗ |

(✓ = 100% exact-match on the sampled frames of that depth. depth-2 is the target regime; 3–5
are out-of-distribution stress probes.) The depth-3+ columns are the interesting part and are
explained in Part III.

Below, each construction: the idea, the key code, where/why it breaks, a minimal example.

## 2.1 `_s2_closer` — closer-marking (the chosen winner)

**Idea.** `depth[i] = #('that' openers q≤i) − #(closings < i)`, but recover the *close* of
each `that` first. Mutually-recursive END pointers `dpfull` (DP end incl. any relative
clause), `vpe` (VP end), `rce`/`cpe` (CP_rel / CP_sent end) are resolved by a **fixed number
of relaxation sweeps**; each sweep propagates one nesting level. Then `depth` is one *stabbing*
count, and `disp` is split into four additive flip-type pieces (`d_adv`, `d_cp`, `d_nprel`,
`d_vp`), each a stabbing sum or a local read.

**Key code.**
```python
ROUNDS = 10            # <-- the depth budget (see Part III)
...
K = ROUNDS
for _ in range(K):     # each sweep = one more resolved nesting level
    dpfull = where(dp_has_rc, rce_at_rc, dp_end_flat)     # DP end jumps over its RC
    vpe    = where(role∈{intrans,objgap}, hb_end, ...dpfull_at.../...cpe_at...)  # VP end
    rce    = where(is_crel, subj?vpe_at_q1 : dpfull_at_q1+1, idx)  # CP_rel end
    cpe    = where(is_c, vpe_at_vpos, idx)                # CP_sent end
e     = where(is_crel, rce, where(is_c, cpe, idx))        # clause end e(q) per 'that'
depth = _stab_ge(where(isthat, e, -1))                    # #intervals [q,e(q)] covering i
disp  = d_adv + d_cp + d_nprel + d_vp                     # four additive flip-type pieces
```

**Where it breaks.** At CP-nesting `depth > ROUNDS − 1`. With `ROUNDS=10` it is correct to
depth 9. Dial `ROUNDS` down and the wall moves in lockstep (Part III). This is the *clean*
failure mode: a bounded-round program is exactly a bounded-depth program.

**Why it's the winner.** Cleanest mapping to §3 (the four pieces *are* the four flip types), an
honest single knob (`ROUNDS`) that makes the depth-limit explicit and testable, and it
re-verified fastest.

## 2.2 `_s2_interval` — opener→closer interval matching

**Idea.** A cute observation: inside any clause `#that == #verb`, so the *balance*
`B[i] = (#that ≤ i) − (#verb ≤ i)` makes each `that` and its clause's **main verb** a matched
open/close pair. Match every verb leftward to the `that` it closes with **one** attention head
(no iteration!); invert to get each clause's main verb. The clause's *end* then extends past
that main verb by the verb's complement tail — and THAT tail resolution still needs a **bounded
number of pointer-jump rounds** (`for _ in range(6)`).

**Key code.**
```python
B    = cumsum(that) − cumsum(verb)                 # O(1): that/verb balance
moc  = kqv(B_at_thats, B+1, indices, equals, reduction='max')  # verb → the 'that' it closes
mv   = invert(moc)                                 # 'that' → its clause's main verb
for _ in range(6):                                 # <-- still bounded rounds for the tail
    vend  = where(jump, close_at_target, local_end)          # verb end (inherit deeper close)
    close = where(isthat, vend_at_mainverb, INF)             # that's clause end
```

**Where it breaks.** The `that`↔verb match is O(1) layers and depth-independent — but the
**complement-tail resolution** (`range(6)`) is again a fixed round count, so the wall returns:
correct to depth ≈ 5 with 6 rounds. Same *layers ∝ depth* wall as `closer`, just with a
cheaper matching front-end. Lesson: a clever O(1) sub-step doesn't remove the wall if *any*
part still iterates per nesting level.

**Minimal example of the wall.** Any depth-6 sentence (six nested CPs) with these 6 rounds:
the sixth clause's tail is never resolved.

## 2.3 `_s2_twolevel` — explicit bounded recursion (four families)

**Idea.** Same four additive families as `closer` (A: CP, B: VP-main, C: adverb, D: NP-rel),
but the span pointers `DPEND/VPEND/CLEND/CPEND` are resolved by a **bottom-up fix-point** with
a large fixed iteration count (`ITERS = 40`). Because `40 ≫` any tested depth, it looked like
it "generalized" — but that is only the round budget being generous.

**Key code.**
```python
ITERS = 40
for _ in range(ITERS):     # bottom-up fix-point: innermost clauses resolve first
    DPEND = ...; VPEND = ...; CLEND = ...; CPEND = ...
depth = _cover(CPEND, isThat)          # enclosing-CP count via a stabbing head
disp  = capA + advC + plusD + minusD + plusB + minusB   # four additive families
```

**Where it breaks.** At depth `> ~ITERS`. Identical wall to `closer`; `ITERS=40` merely hides
it until depth ~39. This is the module that most clearly shows the "generalizes to depth 5!"
headline is an artifact of the constant, not real generalization.

## 2.4 `_s2_gaplca` — pairwise flip-LCA counting (correct only to depth 2)

**Idea.** Express `disp` as the "who crosses over me" count of §3:
`disp[i] = #{j>i : LCA(i,j) is a flip} − #{j<i : LCA(i,j) is a flip} = Term1 − Term2`.
`Term1` (i heads a flip) is a small local sum of comp-sizes; `Term2` (i sits in a flip's comp)
is a sum of interval-stabbing counts. It *also* recovers spans with a `K=6` relaxation.

**Where it breaks — and why it's different.** It fails at **depth 3 (43%)** *even though `K=6`
is more than enough rounds*. So this is **not** the layers-wall — it's a **latent correctness
bug**: at depth ≥ 3 the `Term1/Term2` decomposition produces target slots that are **not a
permutation** (collisions), so the gather duplicates and drops tokens. It was only ever
verified on depth ≤ 2, where the bug is dormant. This is the cautionary tale of the batch:
"100% on the training distribution" (depth ≤ 2) does **not** imply the algorithm is correct —
`gaplca` is a genuinely incomplete program that the in-distribution test could never catch.

**Minimal failure (len 13, depth 3):**
```
HI : Betty hates this girl that believes a engineer that knows that Mary dances
exp: Betty this a Mary dances that knows that engineer believes that girl hates
got: Betty this a Betty Mary Mary knows that engineer believes this girl Betty   # not a permutation
```
Note the repeated `Betty`/`Mary` and dropped tokens — a signature of an invalid `hf_pos`.

## 2.5 So: same idea or different?

**Same core, always.** Every module is the handout's *recursive-reversal → additive
displacement*: `hf_pos = i + Σ ±|sibling|`. What differs:

| axis | closer | interval | twolevel | gaplca |
|---|---|---|---|---|
| span recovery | relaxation ×K | balance-match + jump ×6 | fix-point ×40 | relaxation ×6 |
| displacement expressed as | 4 flip-type pieces | head-local + comp-stab | 4 families | Term1−Term2 (pairwise LCA) |
| failure | layers-wall (depth ≤ K−1) | layers-wall (≤ ~5) | layers-wall (≤ ~39) | **bug** at depth ≥ 3 |

The displacement *sum* has two equivalent guises — a **per-flip-node** sum (closer, twolevel,
interval's head-part) and a **pairwise-LCA crossing** count (gaplca, and the handout's "who
crosses over me?" reframing). They're the same quantity; §3 proves it.

---
---

# PART III — Layers vs depth: the theoretical wall

This is the payoff for the depth-generalization question behind exps 10/16/18/19.

## 3.1 The demonstrated law

`_s2_closer` exposes `ROUNDS` = number of relaxation sweeps = a fixed block of transformer
layers. Sweeping it:

```
 ROUNDS (≈ layers) |  depth0  depth1  depth2  depth3  depth4
        1          |   ✓        ✗       ✗       ✗       ✗
        2          |   ✓        ✓       ✗       ✗       ✗
        3          |   ✓        ✓       ✓       ✗       ✗
        4          |   ✓        ✓       ✓       ✓       ✗
       10          |   ✓        ✓       ✓       ✓       ✓
```

**`R` rounds solve CP-nesting depth `≤ R−1`, and fail *completely* one level deeper.** The
relationship is exactly linear: **layers needed = depth + 1** (up to the constant number of
s-ops per sweep).

## 3.2 Why — the mechanism

A clause's end depends on its complement's end, which (for `V_cp`) is another clause's end,
which depends on *its* complement… The END-pointer recursion has **dependency chains as long as
the nesting is deep**. One relaxation sweep = one attention "hop" = it can extend every
resolved pointer by exactly **one** level. So resolving depth-`d` nesting needs **Ω(d)
sequential sweeps**, i.e. Ω(d) layers. A transformer with a **fixed** number of layers `L`
therefore represents the HI→HF function only for depth `≤ f(L)` and **cannot** extend to deeper
inputs — no amount of data helps, because the deeper input needs more sequential composition
than the architecture physically has.

This is not an artifact of one construction: it recurs in `closer`, `interval`, and `twolevel`
(three independent algorithms), and it is exactly the RASP-L thesis — the task's only programs
use **unbounded-depth bracket matching**, which is outside the length-generalizing primitive
set (content + relative-position + counting). Copy/reverse and Dyck-to-unbounded-depth are the
canonical members of this "does-not-length-generalize-without-hints" family, and structured
reversal is precisely what HI→HF is.

## 3.3 Connection to known transformer expressivity theory

* Bounded-depth hierarchical languages (bounded Dyck) *are* recognizable by transformers, but
  the required **depth (layers) grows with the nesting depth** (Yao et al. 2021, *Self-Attention
  Networks Can Process Bounded Hierarchical Languages*). Our linear `layers = depth + 1` is a
  concrete instance.
* Fixed-depth transformers live in a shallow, uniform-`TC⁰`-ish circuit class (Merrill &
  Sabharwal); genuinely unbounded recursion/state (an unbounded stack for unbounded Dyck) is
  outside it. Depth generalization to *arbitrary* nesting is thus a hard expressivity barrier,
  not merely an optimization one.

## 3.4 Predictions for the dataset splits (the point of all this)

* **depth {0,1} → depth 2** (`…depth_withhold2…`): the model trained on depth ≤ 1 is never
  forced to *use* the extra sweep, so at depth 2 it hits the wall → **predict failure** — which
  matches the empirical negatives in exps 10/16.
* **object-gap → subject-gap relative clauses** (`…object-rc-only-depth2`): both gap types are
  the *same* `CP_rel` flip and need **no extra nesting depth** — the object-gap only changes
  what sits *inside* the clause, handled by the same rules at the same depth → **predict
  transfer succeeds**; a failure there would be optimization, not expressivity.

The two axes are predicted to behave **oppositely**, and Part III explains why: one asks for a
layer the model never had to learn to use; the other reuses machinery it already has.

## 3.5 Caveat — could a smarter program beat the wall?

The wall is about *fixed layers vs unbounded depth*. Two honest escape hatches, both matching
the theory: (a) give the model **depth as an input hint** (index/scratchpad tricks, à la Zhou
et al.) so it doesn't have to *derive* nesting — this is exactly "provide the non-length-gen
part"; (b) allow **layers to scale with input** (universal/looped transformers) — then depth
generalization is back on the table because you've restored the Ω(depth) sequential budget. A
plain fixed-depth decoder trained on shallow data gets neither, so the prediction stands.

---
---

# PART IV — Adversarial review

A second workflow reviewed the winner (`_s2_closer`) + `rasp_l.py` + `pos_tag.py` along four
dimensions. Two agents finished cleanly; two hit an API session limit mid-run (not a finding),
and I covered their dimensions directly. **Net verdict: no defects found; the winner is
verified correct on depth ≤ 2, is legal RASP-L, and the displacement formula is an independently
confirmed tree identity.**

**1. Claim audit — PASS (agent).** Independently re-derived everything *without* reusing the
oracle's `disp`: it built a **gold node-identity permutation** by relabeling every leaf with a
unique word and rendering HI/HF through the generator's own `get_head_initial/get_head_final`
(this is correct even when tokens repeat, unlike `hf_frame_tagged`). Results over ~2,472
sentences (depth 0 exhaustive; depths 1–3 sampled):
* Claim 1 (HF is a permutation of HI): **confirmed**, 0 failures.
* Claim 2 (the additive-displacement formula reproduces the true reordering): **confirmed**, and
  — importantly — **still exact at depth 3**, so it is a *mathematical identity about the tree*,
  not a depth-2 coincidence.
* Claim 3 (`_s2_closer`'s four pieces map to the four flip types and sum to the oracle `disp`):
  **confirmed** — `d_adv`↔adverb blocks, `d_cp`↔CP_sent+CP_rel, `d_nprel`↔NP_singular(NPsRel),
  `d_vp`↔transitive/clausal VP.

**2. RASP-L legality — PASS (agent).** Audited every cross-position callsite and every
`tok_map/seq_map/where` lambda:
* No predicate uses an **absolute index or sequence length** — every `indices` use inside a
  predicate is relative-offset, interval-containment (`geq(end[j], i)`), or a pointer /
  inverse-permutation equality. `n = len(...)` is used only for the empty-input guard.
* The `for _ in range(K)` loop only **re-composes s-ops** (`index_select`+`where`) — legal
  layer-stacking, empirically a depth budget (re-confirmed the wall: ROUNDS=4 → depth-3 40/40,
  ROUNDS=3 → 0/40).
* Every `causal=False` read is over the **HI block**; legality rests on the documented
  "realized in the post-`<sep>` output region" convention — flagged as the load-bearing
  assumption (minor). One harmless **nit**: `_read_off` passes `causal=False` even for backward
  offsets where `causal=True` would do; it still only reaches the past.

**3. Depth-wall — covered directly** (agent hit the limit). The `ROUNDS`-sweep table in §3.1,
independently reproduced by the legality agent, establishes `layers = depth + 1`.

**4. Adversarial correctness — covered directly** (agent hit the limit). I ran the winner
(ROUNDS=10) against hard depth-2 families, all **clean**: broad 3000/3000; **multiplicity-2
close** (CP_sent nested inside a relative clause) 800/800; **relative clause on the subject**
800/800; tall trees 400/400; and **all 24 depth-0 frames × 5 lexicalizations** 120/120. (Raw
output: `artifacts/adversarial_sweep.txt`.)

**Only honest caveat** (raised by the claim-audit agent, not a defect): depths 1–2 were sampled,
not exhaustively enumerated — the depth-2 frame space is ~1.39M frames, depth-3 astronomically
larger — so this is very strong evidence, not a formal proof, for depths ≥ 1. No counterexample
was found in any regime, including the specifically-targeted hard families.

---

## Appendix — file map

| file | role | length-gen status |
|---|---|---|
| `rasp_l.py` | clean-room RASP-L primitives (integer s-ops, causal by default) | — |
| `grammar_oracle.py` | recursive-descent parser + ground-truth features | (not a RASP program) |
| `pos_tag.py` | stage 1: local POS | length-generalizing |
| `_s2_closer.py` | **winner**: stage 2 via closer-marking (`ROUNDS` knob) | depth ≤ ROUNDS−1 |
| `_s2_interval.py` | stage 2 via that/verb balance + pointer-jump | depth ≤ ~5 |
| `_s2_twolevel.py` | stage 2 via 4 families + fix-point | depth ≤ ~39 |
| `_s2_gaplca.py` | stage 2 via pairwise flip-LCA (buggy ≥ depth 3) | correct only ≤ depth 2 |
| `displacement_handout.tex/.pdf` | visual handout for §3 | — |

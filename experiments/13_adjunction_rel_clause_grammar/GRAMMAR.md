# CFG Grammar

- The context free grammar used to generate the Head Initial and Head Final Sentence pairs

## Context Free Grammar
- A simple toy Context Free Grammar (CFG) is used to produce and translate Head Initial (HI) and Head Final (HF) sentences

- Note: at this time the grammar overgenerates but is a good first pass approximation; also bare nouns are currently not supported
- Care is taken to make the Head Initial Grammar unambiguous. Note that the Head Final grammar has the same linearization for Subject and Object Relative clauses since Subject DPs do not swap with the VPs

### Non-Recursive Rules (DP, VP)
| Head Initial Rule | Head Final Rule |
| --- | --- |
| S -> DP VP | S -> DP VP |
| DP -> NP_proper | DP -> NP_proper |
| NP_proper -> N_proper | NP_proper -> N_proper |
| DP -> D NP_singular | DP -> D NP_singular |
| NP_singular -> N_singular | NP_singular -> N_singular |
| VP -> V_intrans | VP -> V_intrans |
| VP -> V_dp DP | VP -> DP V_dp |

### Recursive Rules (Complementizers CP_sent, CP_rel)
- CP_sent is a sentential clause: "that" followed by a S
    - e.g. "that the researcher likes the dog"
- CP_rel is a clause which modifies a noun 
    - e.g. "the cat that likes the dog"
    - In this toy grammar, we restrict relative clauses to modify singular common nouns

| Head Initial Rule | Head Final Rule |
| --- | --- |
| VP -> V_cp CP_sent | VP -> CP_sent V_cp |
| CP_sent -> C S | CP_sent -> S C |
| NP_singular -> NP_singular CP_rel | NP_singular -> CP_rel NP_singular | 
| CP_rel -> C S_subj_gap | CP_rel -> S_subj_gap C |
| S_subj_gap -> VP | S_subj_gap -> VP |
| CP_rel -> C S_obj_gap | CP_rel -> S_obj_gap C |
| S_obj_gap -> DP VP_obj_gap | S_obj_gap -> DP VP_obj_gap |
| VP_obj_gap -> V_dp | VP_obj_gap -> V_dp |

### Adjunct Rules (AdjP, AdvP)
Adjectives and Adverbs are both consider `adjuncts` which adjoin to a phrase and project the same phrase. 

In Head Initial Rule, AdjP left adjoin while AdvP right adjoin. (e.g. "the boy chases the dog carefully". While this grammar can generate "the boy likes the dog loudly", we accept these overgenerations for sake of simplicity.)

In Head Final Rule, both AdjP and AdvP left adjoin. We introduce an additional `NP_singular_adj` to avoid ambiguity with the adjective modifying the relative clause rather than the noun. Note: this ambiguity is allowed in natural language but we disallow for simplicity.

In the case of ambiguity of adjoining CPs to the top most NPs (high attachment) versus embedded into a CP noun

Adverbs and Adjectives are allowed only a recursion of depth 1 (i.e. no recursion).

| Head Initial Rule | Head Final Rule |
| --- | --- |
| NP_singular -> NP_singular_adj | NP_singular -> NP_singular_adj |
| NP_singular_adj -> AdjP N_singular | NP_singular_adj -> AdjP N_singular |
| AdjP -> Adj | AdjP -> Adj |
| VP -> VP_intrans_adv | VP -> VP_intrans_adv |
| VP_intrans_adv -> V_intrans AdvP | VP_intrans_adv -> AdvP V_intrans |
| VP -> VP_dp_adv DP | VP -> DP VP_dp_adv |
| VP_dp_adv -> V_dp AdvP | VP_dp_adv -> AdvP V_dp |
| VP -> VP_cp_adv CP_sent | VP -> CP_sent VP_cp_adv |
| VP_cp_adv -> V_cp AdvP | VP_cp_adv -> AdvP V_cp|
| AdvP -> Adv | AdvP -> Adv |

### Out of Scope

Adding support for prepositional phrases is difficult because 

1. PPs can modify sentence level or at the VP level

2. PP restrictions are highly arbitrary/lexicalized aka which PPs can go with which VPs would take many "memorized" rules in the CFG

3. The VP and CP flipping already capture this flipping property

| VP -> VP PP | VP -> PP VP |
| PP -> P DP | PP -> DP P |

## Terminal (English) Words Used

### D

```
the
a
this
```

### N_singular
```
dog
cat
boy
girl
teacher
student
friend
researcher
dancer
artist
musician
engineer
father
mother
sister
brother
```

### N_proper
```
John
Mary
Iskarous
Jia
James
Hamilton
Betty
Shri
```

### V_dp
```
likes
believes
hates
knows

faces
kisses
chases
pursues
loves
soothes
hugs
consoles
tickles
bedazzles
vexes
```

### V_cp
```
likes
believes
hates
thinks
knows
assumes
claims
```

### V_intrans
```
swims
dances
sings
laughs
smiles
claps
jeers
applauds
```

### C
```
that
```

### Adj
```
attractive
bald
beautiful
chubby
clean
dazzling
drab
elegant
fancy
fit
flabby
glamorous
gorgeous
handsome
magnificent
muscular
plain
plump
scruffy
shapely
skinny
stocky
unkempt
unsightly
agreeable
ambitious
brave
calm
delightful
eager
faithful
gentle
happy
jolly
kind
lively
nice
obedient
polite
proud
silly
thankful
victorious
witty
wonderful
zealous
angry
bewildered
clumsy
defeated
embarrassed
fierce
grumpy
helpless
itchy
jealous
lazy
mysterious
nervous
obnoxious
panicky
pitiful
repulsive
scary
thoughtless
uptight
worried
```

### Adv
```
quickly
slowly
quietly
loudly
happily
sadly
eagerly
gently
fiercely
gracefully
carefully
clumsily
```



## Examples

1. John chases a cat
2. Mary swims
3. Iskarous likes this dog
3. Jia knows that Iskarous likes this dog
4. Jia knows that Iskarous likes that this dog dances

---

### 1. John chases a cat

| Japanese | John-wa | neko-o | oikakeru |
|---|---|---|---|
| Gloss | John-TOP | cat-ACC | chase.PRS |
| English | "John chases a cat." | | |

**Head Initial:**

Bracketed:

```
[S [DP John] [VP [V chases] [DP [D a] [NP cat]]]]
```

Tree:

```
S
|-- DP: John
`-- VP
    |-- V: chases
    `-- DP
        |-- D: a
        `-- NP: cat
```

**Head Final:**

Bracketed:

```
[S [DP John] [VP [DP [D a] [NP cat]] [V chases]]]
```

Tree:

```
S
|-- DP: John
`-- VP
    |-- DP
    |   |-- D: a
    |   `-- NP: cat
    `-- V: chases
```

---

### 2. Mary swims

| Japanese | Mary-wa | oyogu |
|---|---|---|
| Gloss | Mary-TOP | swim.PRS |
| English | "Mary swims." | |

Note: Because `VP -> V_intrans` has no sister to reorder, HI and HF are identical for intransitive sentences.

**Head Initial:**

Bracketed:

```
[S [DP Mary] [VP [V swims]]]
```

Tree:

```
S
|-- DP: Mary
`-- VP
    `-- V: swims
```

**Head Final:**

Bracketed:

```
[S [DP Mary] [VP [V swims]]]
```

Tree:

```
S
|-- DP: Mary
`-- VP
    `-- V: swims
```

---

### 3. Iskarous likes this dog

| Japanese | Iskarous-wa | kono | inu-o | konomu |
|---|---|---|---|---|
| Gloss | Iskarous-TOP | this | dog-ACC | like.PRS |
| English | "Iskarous likes this dog." | | | |

**Head Initial:**

Bracketed:

```
[S [DP Iskarous] [VP [V likes] [DP [D this] [NP dog]]]]
```

Tree:

```
S
|-- DP: Iskarous
`-- VP
    |-- V: likes
    `-- DP
        |-- D: this
        `-- NP: dog
```

**Head Final:**

Bracketed:

```
[S [DP Iskarous] [VP [DP [D this] [NP dog]] [V likes]]]
```

Tree:

```
S
|-- DP: Iskarous
`-- VP
    |-- DP
    |   |-- D: this
    |   `-- NP: dog
    `-- V: likes
```

---

### 4. Jia knows that Iskarous likes this dog

| Japanese | Jia-wa | Iskarous-ga | kono | inu-o | konomu | to | shitteiru |
|---|---|---|---|---|---|---|---|
| Gloss | Jia-TOP | Iskarous-NOM | this | dog-ACC | like.PRS | COMP | know.PROG |
| English | "Jia knows that Iskarous likes this dog." | | | | | | |

**Head Initial:**

Bracketed:

```
[S [DP Jia] [VP [V knows] [CP [C that] [S [DP Iskarous] [VP [V likes] [DP [D this] [NP dog]]]]]]]
```

Tree:

```
S
|-- DP: Jia
`-- VP
    |-- V: knows
    `-- CP
        |-- C: that
        `-- S
            |-- DP: Iskarous
            `-- VP
                |-- V: likes
                `-- DP
                    |-- D: this
                    `-- NP: dog
```

**Head Final:**

Bracketed:

```
[S [DP Jia] [VP [CP [S [DP Iskarous] [VP [DP [D this] [NP dog]] [V likes]]] [C that]] [V knows]]]
```

Tree:

```
S
|-- DP: Jia
`-- VP
    |-- CP
    |   |-- S
    |   |   |-- DP: Iskarous
    |   |   `-- VP
    |   |       |-- DP
    |   |       |   |-- D: this
    |   |       |   `-- NP: dog
    |   |       `-- V: likes
    |   `-- C: that
    `-- V: knows
```

---

### 5. Jia knows that Iskarous likes that this dog dances

| Japanese | Jia-wa | Iskarous-ga | kono | inu-ga | odoru | to | konomu | to | shitteiru |
|---|---|---|---|---|---|---|---|---|---|
| Gloss | Jia-TOP | Iskarous-NOM | this | dog-NOM | dance.PRS | COMP | like.PRS | COMP | know.PROG |
| English | "Jia knows that Iskarous likes that this dog dances." | | | | | | | | |

Note: here `likes` is parsed under `V_cp` (it appears in both `V_dp` and `V_cp` in the lexicon).

**Head Initial:**

Bracketed:

```
[S [DP Jia] [VP [V knows] [CP [C that] [S [DP Iskarous] [VP [V likes] [CP [C that] [S [DP [D this] [NP dog]] [VP [V dances]]]]]]]]]
```

Tree:

```
S
|-- DP: Jia
`-- VP
    |-- V: knows
    `-- CP
        |-- C: that
        `-- S
            |-- DP: Iskarous
            `-- VP
                |-- V: likes
                `-- CP
                    |-- C: that
                    `-- S
                        |-- DP
                        |   |-- D: this
                        |   `-- NP: dog
                        `-- VP
                            `-- V: dances
```

**Head Final:**

Bracketed:

```
[S [DP Jia] [VP [CP [S [DP Iskarous] [VP [CP [S [DP [D this] [NP dog]] [VP [V dances]]] [C that]] [V likes]]] [C that]] [V knows]]]
```

Tree:

```
S
|-- DP: Jia
`-- VP
    |-- CP
    |   |-- S
    |   |   |-- DP: Iskarous
    |   |   `-- VP
    |   |       |-- CP
    |   |       |   |-- S
    |   |       |   |   |-- DP
    |   |       |   |   |   |-- D: this
    |   |       |   |   |   `-- NP: dog
    |   |       |   |   `-- VP
    |   |       |   |       `-- V: dances
    |   |       |   `-- C: that
    |   |       `-- V: likes
    |   `-- C: that
    `-- V: knows
```

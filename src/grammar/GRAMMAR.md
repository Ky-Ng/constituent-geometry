# CFG Grammar

- The context free grammar used to generate the Head Initial and Head Final Sentence pairs

## Context Free Grammar
- A simple toy Context Free Grammar (CFG) is used to produce and translate Head Initial (HI) and Head Final (HF) sentences

- Note: at this time the grammar overgenerates but is a good first pass approximation; also bare nouns are currently not supported

| Head Initial Rule | Head Final Rule |
| --- | --- |
| S -> DP VP | S -> DP VP |
| DP -> NP_proper | DP -> NP_proper |
| DP -> D NP_singular | DP -> D NP_singular |
| VP -> V_dp DP | VP -> DP V_dp |
| VP -> V_cp CP | VP -> CP V_cp |
| VP -> V_intrans | VP -> V_intrans |
| CP -> C S | CP -> S C |

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

### NP_singular
```
dog
cat
boy
girl
```

### NP_proper
```
John
Mary
Iskarous
Jia
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
```

### C
```
that
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

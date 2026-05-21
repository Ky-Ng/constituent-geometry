# CFG Grammar

- The context free grammar used to generate the Head Initial and Head Final Sentence pairs

## Context Free Grammar
- A simple toy Context Free Grammar (CFG) is used to produce and translate Head Initial (HI) and Head Final (HF) sentences

- Note: at this time the grammar overgenerates but is a good first pass approximation; also bare nouns are currently not supported

| Head Initial Rule | Head Final Rule |
| --- | --- |
| S -> DP VP | S -> DP VP |
| DP -> D NP | DP -> D NP |
| VP -> V DP | VP -> DP V |
| VP -> V PP | VP -> PP V |
| VP -> V CP | VP -> CP V |
| PP -> P DP | PP -> DP P |
| CP -> C S | CP -> S C |

## Terminal (English) Words Used

### DP

```
the
a
```

### NP
```
dog
cat
boy
girl
```

### V

```
likes
believes

```

### C
```
that
```

### P
```
to
at
```

## Examples

1. The professor likes the class
2. The doctor teaches at a school
3. Jia knows that Iskarous likes the class
4. Iskarous believes that Jia teaches at a school

---

### 1. The professor likes the class

| Japanese | kyouju-wa | jugyou-o | konomu |
|---|---|---|---|
| Gloss | professor-TOP | class-ACC | like.PRS |
| English | "The professor likes the class." | | |

**Head Initial:**

Bracketed:

```
[S [DP [D the] [NP professor]] [VP [V likes] [DP [D the] [NP class]]]]
```

Tree:

```
S
|-- DP
|   |-- D: the
|   `-- NP: professor
`-- VP
    |-- V: likes
    `-- DP
        |-- D: the
        `-- NP: class
```

**Head Final:**

Bracketed:

```
[S [DP [D the] [NP professor]] [VP [DP [D the] [NP class]] [V likes]]]
```

Tree:

```
S
|-- DP
|   |-- D: the
|   `-- NP: professor
`-- VP
    |-- DP
    |   |-- D: the
    |   `-- NP: class
    `-- V: likes
```

---

### 2. The doctor teaches at a school

| Japanese | isha-wa | gakkou-de | oshieru |
|---|---|---|---|
| Gloss | doctor-TOP | school-LOC | teach.PRS |
| English | "The doctor teaches at a school." | | |

**Head Initial:**

Bracketed:

```
[S [DP [D the] [NP doctor]] [VP [V teaches] [PP [P at] [DP [D a] [NP school]]]]]
```

Tree:

```
S
|-- DP
|   |-- D: the
|   `-- NP: doctor
`-- VP
    |-- V: teaches
    `-- PP
        |-- P: at
        `-- DP
            |-- D: a
            `-- NP: school
```

**Head Final:**

Bracketed:

```
[S [DP [D the] [NP doctor]] [VP [PP [DP [D a] [NP school]] [P at]] [V teaches]]]
```

Tree:

```
S
|-- DP
|   |-- D: the
|   `-- NP: doctor
`-- VP
    |-- PP
    |   |-- DP
    |   |   |-- D: a
    |   |   `-- NP: school
    |   `-- P: at
    `-- V: teaches
```

---

### 3. Jia knows that Iskarous likes the class

| Japanese | Jia-wa | Iskarous-ga | jugyou-o | konomu | to | shitteiru |
|---|---|---|---|---|---|---|
| Gloss | Jia-TOP | Iskarous-NOM | class-ACC | like.PRS | COMP | know.PROG |
| English | "Jia knows that Iskarous likes the class." | | | | | |

**Head Initial:**

Bracketed:

```
[S [DP Jia] [VP [V knows] [CP [C that] [S [DP Iskarous] [VP [V likes] [DP [D the] [NP class]]]]]]]
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
                    |-- D: the
                    `-- NP: class
```

**Head Final:**

Bracketed:

```
[S [DP Jia] [VP [CP [S [DP Iskarous] [VP [DP [D the] [NP class]] [V likes]]] [C that]] [V knows]]]
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
    |   |       |   |-- D: the
    |   |       |   `-- NP: class
    |   |       `-- V: likes
    |   `-- C: that
    `-- V: knows
```

---

### 4. Iskarous believes that Jia teaches at a school

| Japanese | Iskarous-wa | Jia-ga | gakkou-de | oshieru | to | shinjiteiru |
|---|---|---|---|---|---|---|
| Gloss | Iskarous-TOP | Jia-NOM | school-LOC | teach.PRS | COMP | believe.PROG |
| English | "Iskarous believes that Jia teaches at a school." | | | | | |

**Head Initial:**

Bracketed:

```
[S [DP Iskarous] [VP [V believes] [CP [C that] [S [DP Jia] [VP [V teaches] [PP [P at] [DP [D a] [NP school]]]]]]]]
```

Tree:

```
S
|-- DP: Iskarous
`-- VP
    |-- V: believes
    `-- CP
        |-- C: that
        `-- S
            |-- DP: Jia
            `-- VP
                |-- V: teaches
                `-- PP
                    |-- P: at
                    `-- DP
                        |-- D: a
                        `-- NP: school
```

**Head Final:**

Bracketed:

```
[S [DP Iskarous] [VP [CP [S [DP Jia] [VP [PP [DP [D a] [NP school]] [P at]] [V teaches]]] [C that]] [V believes]]]
```

Tree:

```
S
|-- DP: Iskarous
`-- VP
    |-- CP
    |   |-- S
    |   |   |-- DP: Jia
    |   |   `-- VP
    |   |       |-- PP
    |   |       |   |-- DP
    |   |       |   |   |-- D: a
    |   |       |   |   `-- NP: school
    |   |       |   `-- P: at
    |   |       `-- V: teaches
    |   `-- C: that
    `-- V: believes
```
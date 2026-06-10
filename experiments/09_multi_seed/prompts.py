"""The fixed HI/HF prompt set for the experiment-09 cross-seed comparison.

WHY A FIXED, HAND-PICKED SET (not dataset rows)
-----------------------------------------------
The cross-seed grids only line up if EVERY seed is teacher-forced on the SAME
(hi, hf) pair, so token axes and tensor shapes [L, H, Tq, Tk] match. We also want
to probe how consistency changes with embedding DEPTH (clause nesting), so we fix
three examples at each depth 0-3. Naming each prompt `depth<D>-ex<N>` makes the
output path `figures/<arch>/heatmaps/depth<D>-ex<N>/...` self-documenting.

PROVENANCE (from the task spec)
-------------------------------
- depth 0: from the TRAIN split.
- depth 1-3: from the TEST split.
The `hf` here is the GOLD reordered target string (NOT model-generated), so the
decoder-self / cross attentions reflect the model's behavior on the correct parse.
"""

PROMPTS = [
    # ---- depth 0 (train) ----
    {"depth": 0, "ex": 0,
     "hi": "this friend knows the dog",
     "hf": "this friend the dog knows"},
    {"depth": 0, "ex": 1,
     "hi": "this cat pursues a musician",
     "hf": "this cat a musician pursues"},
    {"depth": 0, "ex": 2,
     "hi": "the boy chases this artist",
     "hf": "the boy this artist chases"},

    # ---- depth 1 (test) ----
    {"depth": 1, "ex": 0,
     "hi": "the dancer thinks that a cat knows Betty",
     "hf": "the dancer a cat Betty knows that thinks"},
    {"depth": 1, "ex": 1,
     "hi": "a cat assumes that this teacher pursues James",
     "hf": "a cat this teacher James pursues that assumes"},
    {"depth": 1, "ex": 2,
     "hi": "a musician thinks that this artist believes Jia",
     "hf": "a musician this artist Jia believes that thinks"},

    # ---- depth 2 (test) ----
    {"depth": 2, "ex": 0,
     "hi": "this dog knows that Jia claims that Iskarous dances",
     "hf": "this dog Jia Iskarous dances that claims that knows"},
    {"depth": 2, "ex": 1,
     "hi": "the boy hates that Shri claims that Mary sings",
     "hf": "the boy Shri Mary sings that claims that hates"},
    {"depth": 2, "ex": 2,
     "hi": "a artist hates that Jia likes that Iskarous laughs",
     "hf": "a artist Jia Iskarous laughs that likes that hates"},

    # ---- depth 3 (test) ----
    {"depth": 3, "ex": 0,
     "hi": "the boy likes that Iskarous assumes that a friend knows that this student chases Betty",
     "hf": "the boy Iskarous a friend this student Betty chases that knows that assumes that likes"},
    {"depth": 3, "ex": 1,
     "hi": "this engineer knows that James likes that a sister hates that a student loves Iskarous",
     "hf": "this engineer James a sister a student Iskarous loves that hates that likes that knows"},
    {"depth": 3, "ex": 2,
     "hi": "this engineer hates that Iskarous claims that this sister assumes that a dancer vexes Shri",
     "hf": "this engineer Iskarous this sister a dancer Shri vexes that assumes that claims that hates"},
]


def prompt_name(depth: int, ex: int) -> str:
    """Folder/file stem for one prompt: matches the requested `depth<D>-ex<N>` layout."""
    return f"depth{depth}-ex{ex}"

"""Dead-simple Dash viewer for the per-experiment attention figures.

Controls:
  - experiment:   discovered by scanning ``experiments/*/figures/``
  - folder levels: a CASCADE of dropdowns, one per directory level. Picking a
                level repopulates the next with that node's children, so a deep
                tree like ``vaswani_layer_2/heatmaps/depth3-ex1`` is navigated
                one segment at a time instead of as one giant flat path. The
                number of level-dropdowns shown adapts to the chosen branch's
                depth (extra ones hide); flat single-level layouts show one.

Renders every PNG in the deepest selected folder, stacked. No prettification —
this is a research scratch viewer.

Run on the cluster login node (no GPU needed):
    uv run python -m visualization.app.server

Then on your local machine:
    ssh -L 8050:localhost:8050 <user>@<carc-login-host>
    # open http://localhost:8050 in a local browser

If you're already inside a compute job, replace ``<carc-login-host>`` with the
compute node hostname (e.g. ``c06-02``) and SSH-jump through the login node:
    ssh -J <user>@<carc-login-host> -L 8050:localhost:8050 <user>@<compute-node>
"""

from __future__ import annotations

from pathlib import Path

from dash import ALL, Dash, Input, Output, ctx, dcc, html
from flask import abort, send_from_directory


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = REPO_ROOT / "experiments"

# Upper bound on how many cascading level-dropdowns we render. The pool is fixed
# (pattern-matching callbacks need the components to exist up front); unused ones
# hide. No experiment tree is anywhere near this deep.
MAX_LEVELS = 8


def discover_experiments() -> list[str]:
    """Experiment dirs that have at least one subfolder under figures/."""
    out: list[str] = []
    for p in sorted(EXP_ROOT.iterdir()):
        fig = p / "figures"
        if p.is_dir() and fig.is_dir() and any(c.is_dir() for c in fig.iterdir()):
            out.append(p.name)
    return out


def discover_folders(experiment: str) -> list[str]:
    """Every dir under ``figures/`` that directly holds PNGs, recursively.

    Returns POSIX-style paths relative to ``figures/`` (e.g.
    ``vaswani/heatmaps/depth3-ex1``), sorted for a stable dropdown order.
    Recursing means nested figure trees are found, not just the top level.
    """
    fig = EXP_ROOT / experiment / "figures"
    if not fig.is_dir():
        return []
    dirs = {
        p.parent for p in fig.rglob("*") if p.is_file() and p.suffix.lower() == ".png"
    }
    return sorted(d.relative_to(fig).as_posix() for d in dirs)


def list_pngs(experiment: str, folder: str) -> list[str]:
    """PNG filenames directly inside ``figures/<folder>`` (``folder`` may nest)."""
    d = EXP_ROOT / experiment / "figures" / folder
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.suffix.lower() == ".png")


# ---- cascading-dropdown navigation -----------------------------------------
# discover_folders() gives the flat list of PNG-holding paths; build_tree turns
# those into a nested dict so the UI can drill one segment at a time. A node's
# children are its sub-segments; a LEAF (empty dict) is a directory that holds
# PNGs directly.
def build_tree(experiment: str) -> dict:
    """Nested {segment: subtree} of every PNG-holding path under figures/."""
    tree: dict = {}
    for folder in discover_folders(experiment):
        node = tree
        for seg in folder.split("/"):
            node = node.setdefault(seg, {})
    return tree


def normalize_path(tree: dict, desired: list[str | None]) -> list[str]:
    """Resolve a (possibly partial/stale) selection into a full path to a leaf.

    Walks from the root following ``desired`` where each segment is a valid child,
    else defaulting to the first child, until a leaf (no children) is reached. So
    a fresh experiment or a just-changed upper level auto-fills the deeper levels.
    """
    node, path, i = tree, [], 0
    while node:                                   # node has children -> another level
        seg = desired[i] if i < len(desired) and desired[i] in node else sorted(node)[0]
        path.append(seg)
        node = node[seg]
        i += 1
    return path


def levels_along_path(tree: dict, path: list[str]) -> list[tuple[list[str], str]]:
    """For each level on ``path``, the (sibling options, chosen value) at that level."""
    out, node = [], tree
    for seg in path:
        out.append((sorted(node), seg))
        node = node[seg]
    return out


def render_images(experiment: str, folder: str):
    """The stacked-PNG gallery for one (already-resolved) folder path."""
    if not folder:
        return html.Em("nothing selected")
    pngs = list_pngs(experiment, folder)
    if not pngs:
        return html.Em(f"no PNGs in {folder}")
    return [
        html.Div(
            style={"marginBottom": "16px"},
            children=[
                html.Div(name, style={"fontSize": "12px", "marginBottom": "4px"}),
                html.Img(
                    src=f"/figure/{experiment}/{folder}/{name}",
                    style={"maxWidth": "100%", "border": "1px solid #ddd"},
                ),
            ],
        )
        for name in pngs
    ]


app = Dash(__name__)
app.title = "attention heatmaps"


# ``folder`` is a <path:> converter so it can span nested dirs (it matches
# slashes); ``filename`` greedily binds the final segment.
@app.server.route("/figure/<experiment>/<path:folder>/<filename>")
def serve_figure(experiment: str, folder: str, filename: str):
    # Path-traversal guard. ``folder`` is allowed to contain '/' (it's a nested
    # subpath), but never '..'; experiment/filename stay single segments.
    if "/" in experiment or "/" in filename:
        abort(400)
    for part in (experiment, folder, filename):
        if ".." in part:
            abort(400)
    d = EXP_ROOT / experiment / "figures" / folder
    if not d.is_dir():
        abort(404)
    return send_from_directory(d, filename)


_init_exps = discover_experiments()

# Hidden style for an unused level-dropdown; the navigate callback flips levels
# on/off by swapping this for the visible style below.
_HIDDEN = {"width": "240px", "display": "none"}
_VISIBLE = {"width": "240px"}

app.layout = html.Div(
    style={"fontFamily": "monospace", "padding": "12px"},
    children=[
        html.H3("attention heatmaps"),
        html.Div(
            style={"display": "flex", "gap": "12px", "alignItems": "center",
                   "flexWrap": "wrap"},
            children=[
                html.Label("experiment:"),
                dcc.Dropdown(
                    id="experiment",
                    options=[{"label": e, "value": e} for e in _init_exps],
                    value=_init_exps[0] if _init_exps else None,
                    clearable=False,
                    style={"width": "560px"},
                ),
                html.Label("folder:"),
                # Fixed pool of cascading level-dropdowns; the callback fills the
                # active ones and hides the rest. Populated on initial load.
                *[
                    dcc.Dropdown(
                        id={"type": "level", "index": k},
                        options=[],
                        clearable=False,
                        style=_HIDDEN,
                    )
                    for k in range(MAX_LEVELS)
                ],
            ],
        ),
        html.Hr(),
        html.Div(id="gallery"),
    ],
)


def resolve(experiment: str | None, level_values: list[str | None],
            changed_index: int | None):
    """Pure cascade logic (no Dash context), kept separate so it's unit-testable.

    ``changed_index`` is the level-dropdown the user just changed, or None when the
    experiment changed / on first load. Returns the four callback outputs:
    (options, values, styles, gallery), each list sized MAX_LEVELS.
    """
    empty = ([[]] * MAX_LEVELS, [None] * MAX_LEVELS, [_HIDDEN] * MAX_LEVELS)
    if not experiment:
        return (*empty, html.Em("nothing selected"))

    tree = build_tree(experiment)
    if not tree:
        return (*empty, html.Em("no figures"))

    # How much of the prior selection survives:
    #  - experiment changed (changed_index is None): start fresh from the root.
    #  - level k changed: keep picks 0..k, drop the now-stale deeper ones so they
    #    re-default within the newly chosen branch.
    desired = level_values[: changed_index + 1] if changed_index is not None else []

    path = normalize_path(tree, desired)
    levels = levels_along_path(tree, path)

    options_out, value_out, style_out = [], [], []
    for k in range(MAX_LEVELS):
        if k < len(levels):
            opts, val = levels[k]
            options_out.append([{"label": o, "value": o} for o in opts])
            value_out.append(val)
            style_out.append(_VISIBLE)
        else:
            options_out.append([])
            value_out.append(None)
            style_out.append(_HIDDEN)

    return options_out, value_out, style_out, render_images(experiment, "/".join(path))


@app.callback(
    Output({"type": "level", "index": ALL}, "options"),
    Output({"type": "level", "index": ALL}, "value"),
    Output({"type": "level", "index": ALL}, "style"),
    Output("gallery", "children"),
    Input("experiment", "value"),
    Input({"type": "level", "index": ALL}, "value"),
)
def navigate(experiment: str | None, level_values: list[str | None]):
    """Thin callback wrapper: read which input fired from ctx, delegate to resolve."""
    trig = ctx.triggered_id
    changed = trig["index"] if isinstance(trig, dict) and trig.get("type") == "level" else None
    return resolve(experiment, level_values, changed)


def main() -> None:
    # 127.0.0.1 only: rely on SSH port-forwarding for remote access. Bind to
    # 0.0.0.0 if you intentionally want LAN exposure.
    app.run(host="127.0.0.1", port=8050, debug=False)


if __name__ == "__main__":
    main()

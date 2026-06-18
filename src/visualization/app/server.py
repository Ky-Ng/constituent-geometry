"""Dead-simple Dash viewer for the per-experiment attention figures.

Controls:
  - experiment:   discovered by scanning ``experiments/*/figures/``
  - folder levels: cascading dropdowns, one per directory level.
  - layer:   multi-select — only shown when the folder/experiment path contains
             "gpt2" or "vaswani"; hidden for paths with no layer_{N}_ PNGs.
             Options are populated dynamically from the current folder's PNGs.
             Selections persist across folder changes; reset on experiment change.
  - type:    multi-select (cross / decoder_self / encoder_self) — only shown for
             vaswani paths. Selections persist across folder and experiment changes
             (it is Input-only, never an Output, so Dash never resets it).

Run on the cluster login node (no GPU needed):
    uv run python -m visualization.app.server
"""

from __future__ import annotations

import re
from pathlib import Path

from dash import ALL, Dash, Input, Output, State, ctx, dcc, html
from flask import abort, send_from_directory


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = REPO_ROOT / "experiments"

MAX_LEVELS = 8

VASWANI_ATTN_TYPES = ["cross", "decoder_self", "encoder_self"]


# ---- PNG attribute parsing --------------------------------------------------

def parse_png_attrs(name: str) -> tuple[str | None, str | None]:
    """Return (layer_key, attn_type) from a PNG filename.

    New-style (experiments 09+):
        layer_{N}_attention.png   -> ("layer_N", None)      [GPT2]
        layer_{N}_{type}.png      -> ("layer_N", type)      [Vaswani]
    Old-style (experiments 06-08):
        {slug}_{type}_{view}.png  -> (None, type)           [Vaswani]
        {slug}_self_{view}.png    -> (None, None)           [GPT2]
    Unrecognised filenames return (None, None) and always pass through.
    """
    # New style: filename starts with layer_{N}_
    m = re.match(r"^layer_(\d+)_(.*?)\.png$", name)
    if m:
        layer_key = f"layer_{m.group(1)}"
        rest = m.group(2)
        attn_type = rest if rest in VASWANI_ATTN_TYPES else None
        return layer_key, attn_type

    # Old style: match from the suffix (longer names first to avoid "self" eating
    # "decoder_self" / "encoder_self").
    for at in VASWANI_ATTN_TYPES:
        for vt in ("per_head", "head_avg"):
            if name.endswith(f"_{at}_{vt}.png"):
                return None, at

    return None, None  # GPT2 old-style or unrecognised


def list_available_layers(experiment: str, folder: str) -> list[str]:
    """Sorted list of layer keys ('layer_0', ...) present in the folder's PNGs."""
    nums: set[int] = set()
    for name in list_pngs(experiment, folder):
        m = re.match(r"^layer_(\d+)_", name)
        if m:
            nums.add(int(m.group(1)))
    return [f"layer_{n}" for n in sorted(nums)]


def is_gpt2_path(folder_path: str, experiment: str) -> bool:
    """True when the combined folder + experiment string signals a GPT2 model.

    Checks both so old-style experiments (where the folder itself is plain
    ``depth1-ex1``) are still classified from the experiment name.
    """
    combined = f"{folder_path}/{experiment}".lower()
    return "gpt2" in combined


def is_vaswani_path(folder_path: str, experiment: str) -> bool:
    combined = f"{folder_path}/{experiment}".lower()
    return "vaswani" in combined


# ---- experiment / folder discovery -----------------------------------------

def discover_experiments() -> list[str]:
    out: list[str] = []
    for p in sorted(EXP_ROOT.iterdir()):
        fig = p / "figures"
        if p.is_dir() and fig.is_dir() and any(c.is_dir() for c in fig.iterdir()):
            out.append(p.name)
    return out


def discover_folders(experiment: str) -> list[str]:
    fig = EXP_ROOT / experiment / "figures"
    if not fig.is_dir():
        return []
    dirs = {
        p.parent for p in fig.rglob("*") if p.is_file() and p.suffix.lower() == ".png"
    }
    return sorted(d.relative_to(fig).as_posix() for d in dirs)


def list_pngs(experiment: str, folder: str) -> list[str]:
    d = EXP_ROOT / experiment / "figures" / folder
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.suffix.lower() == ".png")


# ---- cascading-dropdown navigation -----------------------------------------

def build_tree(experiment: str) -> dict:
    tree: dict = {}
    for folder in discover_folders(experiment):
        node = tree
        for seg in folder.split("/"):
            node = node.setdefault(seg, {})
    return tree


def normalize_path(tree: dict, desired: list[str | None]) -> list[str]:
    node, path, i = tree, [], 0
    while node:
        seg = desired[i] if i < len(desired) and desired[i] in node else sorted(node)[0]
        path.append(seg)
        node = node[seg]
        i += 1
    return path


def levels_along_path(tree: dict, path: list[str]) -> list[tuple[list[str], str]]:
    out, node = [], tree
    for seg in path:
        out.append((sorted(node), seg))
        node = node[seg]
    return out


# ---- gallery rendering ------------------------------------------------------

def render_images(experiment: str, folder: str,
                  layer_sel: list[str] | None,
                  attn_sel: list[str] | None,
                  gpt2_mode: bool):
    """Stacked-PNG gallery filtered by selected layers and (for Vaswani) types.

    A PNG is shown when:
    - Its layer_key (if any) is in layer_sel (or layer_sel is None/empty).
    - gpt2_mode is False AND its attn_type (if any) is in attn_sel.
    PNGs with unrecognised names always pass through.
    """
    if not folder:
        return html.Em("nothing selected")

    layer_set = set(layer_sel) if layer_sel else None
    attn_set  = set(attn_sel)  if attn_sel  else None

    filtered = []
    for name in list_pngs(experiment, folder):
        lk, at = parse_png_attrs(name)
        if lk is not None and layer_set is not None and lk not in layer_set:
            continue
        if not gpt2_mode and at is not None and attn_set is not None and at not in attn_set:
            continue
        filtered.append(name)

    if not filtered:
        return html.Em(f"no PNGs match the current filter in {folder}")
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
        for name in filtered
    ]


# ---- Dash app ---------------------------------------------------------------

app = Dash(__name__)
app.title = "attention heatmaps"


@app.server.route("/figure/<experiment>/<path:folder>/<filename>")
def serve_figure(experiment: str, folder: str, filename: str):
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
_HIDDEN  = {"width": "240px", "display": "none"}
_VISIBLE = {"width": "240px"}

app.layout = html.Div(
    style={"fontFamily": "monospace", "padding": "12px"},
    children=[
        html.H3("attention heatmaps"),
        # Row 1: experiment + cascading folder dropdowns
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
        # Row 2: contextual filter checkboxes
        #
        # layer-container: shown when the current folder has layer_{N}_ PNGs.
        #   Options are dynamic (Output); value is also Output but computed to
        #   persist user selections across folder changes (reset on exp change).
        #
        # attn-type-container: shown for vaswani paths only.
        #   Options are fixed. Value is Input-only — Dash never resets it,
        #   so selections persist unconditionally (across folders AND experiments).
        html.Div(
            style={"display": "flex", "gap": "24px", "alignItems": "flex-start",
                   "marginTop": "8px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    id="layer-container",
                    style={"display": "none"},
                    children=[
                        html.Label("layer:", style={"marginRight": "8px"}),
                        dcc.Checklist(
                            id="layer-filter",
                            options=[],
                            value=[],
                            inline=True,
                        ),
                    ],
                ),
                html.Div(
                    id="attn-type-container",
                    style={"display": "none"},
                    children=[
                        html.Label("type:", style={"marginRight": "8px"}),
                        dcc.Checklist(
                            id="attn-type-filter",
                            options=[{"label": f" {t}", "value": t}
                                     for t in VASWANI_ATTN_TYPES],
                            value=VASWANI_ATTN_TYPES[:],
                            inline=True,
                        ),
                    ],
                ),
            ],
        ),
        html.Hr(),
        html.Div(id="gallery"),
    ],
)


def resolve(experiment: str | None, level_values: list[str | None],
            changed_index: int | None,
            layer_sel: list[str] | None,
            attn_sel: list[str] | None,
            trig_id):
    """Cascade + filter logic, kept pure so it's unit-testable."""
    empty_cascade = ([[]] * MAX_LEVELS, [None] * MAX_LEVELS, [_HIDDEN] * MAX_LEVELS)
    if not experiment:
        return (*empty_cascade, html.Em("nothing selected"),
                [], [], {"display": "none"}, {"display": "none"})

    tree = build_tree(experiment)
    if not tree:
        return (*empty_cascade, html.Em("no figures"),
                [], [], {"display": "none"}, {"display": "none"})

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

    folder = "/".join(path)
    gpt2    = is_gpt2_path(folder, experiment)
    vaswani = is_vaswani_path(folder, experiment)

    # --- layer filter ---
    available = list_available_layers(experiment, folder)
    is_exp_change = trig_id == "experiment" or trig_id is None
    if is_exp_change or layer_sel is None:
        # Reset to all available on experiment change or first load.
        new_layer_value = available
    elif changed_index is not None:
        # Folder changed: keep selections that still exist in the new folder.
        # If nothing survives (completely different set), fall back to all.
        kept = [l for l in layer_sel if l in available]
        new_layer_value = kept if kept else available
    else:
        # Filter change or other: pass through unchanged.
        new_layer_value = layer_sel if layer_sel is not None else available

    layer_opts = [{"label": f" {l}", "value": l} for l in available]
    layer_container_style = {} if available else {"display": "none"}

    # --- attn type filter ---
    # Visibility only; value is Input-only and persists naturally.
    attn_container_style = {} if vaswani else {"display": "none"}

    gallery = render_images(experiment, folder, new_layer_value, attn_sel, gpt2)

    return (options_out, value_out, style_out,
            gallery,
            layer_opts, new_layer_value, layer_container_style,
            attn_container_style)


@app.callback(
    Output({"type": "level", "index": ALL}, "options"),
    Output({"type": "level", "index": ALL}, "value"),
    Output({"type": "level", "index": ALL}, "style"),
    Output("gallery", "children"),
    Output("layer-filter", "options"),
    Output("layer-filter", "value"),
    Output("layer-container", "style"),
    Output("attn-type-container", "style"),
    Input("experiment", "value"),
    Input({"type": "level", "index": ALL}, "value"),
    Input("layer-filter", "value"),
    Input("attn-type-filter", "value"),
)
def navigate(experiment: str | None, level_values: list[str | None],
             layer_sel: list[str] | None, attn_sel: list[str] | None):
    trig = ctx.triggered_id
    changed = trig["index"] if isinstance(trig, dict) and trig.get("type") == "level" else None
    return resolve(experiment, level_values, changed, layer_sel, attn_sel, trig)


def main() -> None:
    app.run(host="127.0.0.1", port=8050, debug=False)


if __name__ == "__main__":
    main()

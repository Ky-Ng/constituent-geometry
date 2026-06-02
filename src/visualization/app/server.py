"""Dead-simple Dash viewer for the per-experiment attention figures.

Two dropdowns:
  - experiment: discovered by scanning ``experiments/*/figures/``
  - folder:     subdirs under the chosen experiment's ``figures/``

Renders every PNG in the chosen folder, stacked. No prettification — this
is a research scratch viewer.

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

from dash import Dash, Input, Output, dcc, html
from flask import abort, send_from_directory


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = REPO_ROOT / "experiments"


def discover_experiments() -> list[str]:
    """Experiment dirs that have at least one subfolder under figures/."""
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
    return sorted(p.name for p in fig.iterdir() if p.is_dir())


def list_pngs(experiment: str, folder: str) -> list[str]:
    d = EXP_ROOT / experiment / "figures" / folder
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.suffix.lower() == ".png")


app = Dash(__name__)
app.title = "attention heatmaps"


@app.server.route("/figure/<experiment>/<folder>/<filename>")
def serve_figure(experiment: str, folder: str, filename: str):
    # Path-traversal guard: URL params can't contain '/' or '..'. Flask's
    # send_from_directory also rejects these, but be belt-and-suspenders.
    for part in (experiment, folder, filename):
        if "/" in part or ".." in part:
            abort(400)
    d = EXP_ROOT / experiment / "figures" / folder
    if not d.is_dir():
        abort(404)
    return send_from_directory(d, filename)


_init_exps = discover_experiments()
_init_folders = discover_folders(_init_exps[0]) if _init_exps else []

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
                dcc.Dropdown(
                    id="folder",
                    options=[{"label": f, "value": f} for f in _init_folders],
                    value=_init_folders[0] if _init_folders else None,
                    clearable=False,
                    style={"width": "320px"},
                ),
            ],
        ),
        html.Hr(),
        html.Div(id="gallery"),
    ],
)


@app.callback(
    Output("folder", "options"),
    Output("folder", "value"),
    Input("experiment", "value"),
)
def update_folders(experiment: str | None):
    if not experiment:
        return [], None
    folders = discover_folders(experiment)
    return (
        [{"label": f, "value": f} for f in folders],
        folders[0] if folders else None,
    )


@app.callback(
    Output("gallery", "children"),
    Input("experiment", "value"),
    Input("folder", "value"),
)
def render_gallery(experiment: str | None, folder: str | None):
    if not experiment or not folder:
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


def main() -> None:
    # 127.0.0.1 only: rely on SSH port-forwarding for remote access. Bind to
    # 0.0.0.0 if you intentionally want LAN exposure.
    app.run(host="127.0.0.1", port=8050, debug=False)


if __name__ == "__main__":
    main()

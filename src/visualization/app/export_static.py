"""Export the attention-heatmap viewer as a shareable zip bundle.

Unlike server.py (which serves figures live over Flask), this walks the same
``experiments/*/figures/<folder>/*.png`` tree once and writes a single .zip:

    attention_heatmaps_<YYYYMMDD_HHMMSS>/
        index.html                       <- dropdown UX, vanilla JS, no server
        figures/<exp>/<folder>/*.png     <- the PNGs, copied verbatim

``index.html`` references the PNGs by relative path, so the recipient just
unzips and opens index.html in any browser — no server, no network. Keeping the
images as files (not base64-inlined) keeps the bundle lean and lets it scale as
the figures tree grows. The creation timestamp is stamped onto the zip filename
(and shown in the page header) so successive exports don't clobber each other.

    uv run python -m visualization.app.export_static
    uv run python -m visualization.app.export_static --out ~/share/heatmaps.zip
    uv run python -m visualization.app.export_static --experiment 08_GPT2_RoPE_hi_hf_frames_heads_4_layers_4

NOTE: importing ``.server`` here constructs the (unused) Dash ``app`` object as
an import side effect. It's cheap and binds no socket, so it's harmless. If you
dislike that, lift discover_experiments/discover_folders/list_pngs/EXP_ROOT into
the (currently empty) ``__init__.py`` and import them from there in both files.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from .server import EXP_ROOT, discover_experiments, discover_folders, list_pngs


def collect(
    experiments: list[str],
) -> tuple[dict[str, dict[str, list[dict]]], list[tuple[Path, str]]]:
    """Returns (manifest, files).

    manifest: {experiment: {folder: [{"name", "src"(relative url)}, ...]}}
    files:    [(abspath, "figures/<exp>/<folder>/<name>"), ...] to add to the zip
    Empty experiments/folders are dropped from the manifest.
    """
    data: dict[str, dict[str, list[dict]]] = {}
    files: list[tuple[Path, str]] = []
    for exp in experiments:
        folders: dict[str, list[dict]] = {}
        for folder in discover_folders(exp):
            items = []
            for name in list_pngs(exp, folder):
                abspath = EXP_ROOT / exp / "figures" / folder / name
                files.append((abspath, f"figures/{exp}/{folder}/{name}"))
                # URL-encode each path component so spaces/odd chars resolve.
                src = "/".join(quote(p) for p in ("figures", exp, folder, name))
                items.append({"name": name, "src": src})
            if items:
                folders[folder] = items
        if folders:
            data[exp] = folders
    return data, files


# Braces in the CSS/JS are doubled so str.format only fills the named fields.
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: monospace; padding: 12px; }}
  .controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  select {{ font-family: monospace; }}
  hr {{ margin: 12px 0; }}
  .gen {{ color: #888; font-size: 12px; margin: 0 0 8px; }}
  .fig {{ margin-bottom: 16px; }}
  .fig .name {{ font-size: 12px; margin-bottom: 4px; }}
  .fig img {{ max-width: 100%; border: 1px solid #ddd; }}
</style>
</head>
<body>
<h3>{title}</h3>
<p class="gen">generated {generated}</p>
<div class="controls">
  <label>experiment:</label>
  <select id="experiment" style="width:560px"></select>
  <label>folder:</label>
  <select id="folder" style="width:320px"></select>
</div>
<hr>
<div id="gallery"></div>
<script id="data" type="application/json">{data_json}</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const expSel = document.getElementById("experiment");
const folderSel = document.getElementById("folder");
const gallery = document.getElementById("gallery");

function fill(sel, values) {{
  sel.innerHTML = "";
  for (const v of values) {{
    const o = document.createElement("option");
    o.value = v; o.textContent = v;
    sel.appendChild(o);
  }}
}}

function render() {{
  gallery.innerHTML = "";
  const items = ((DATA[expSel.value] || {{}})[folderSel.value]) || [];
  if (!items.length) {{ gallery.innerHTML = "<em>nothing to show</em>"; return; }}
  for (const it of items) {{
    const wrap = document.createElement("div"); wrap.className = "fig";
    const name = document.createElement("div");
    name.className = "name"; name.textContent = it.name;
    const img = document.createElement("img");
    img.src = it.src; img.loading = "lazy"; img.alt = it.name;
    wrap.appendChild(name); wrap.appendChild(img);
    gallery.appendChild(wrap);
  }}
}}

function onExp() {{
  fill(folderSel, Object.keys(DATA[expSel.value] || {{}}));
  render();
}}

expSel.addEventListener("change", onExp);
folderSel.addEventListener("change", render);

fill(expSel, Object.keys(DATA));
onExp();
</script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export attention heatmaps as a shareable zip bundle."
    )
    p.add_argument("--out", type=Path, default=EXP_ROOT.parent / "attention_heatmaps.zip")
    p.add_argument("--experiment", action="append", default=None,
                   help="limit to this experiment (repeatable); default: all")
    p.add_argument("--title", default="attention heatmaps")
    args = p.parse_args()

    experiments = args.experiment or discover_experiments()
    data, files = collect(experiments)
    if not data:
        raise SystemExit("no figures found")

    # One datetime, two formats: filename-safe stamp + human-readable header.
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    out = args.out.with_name(f"{args.out.stem}_{stamp}{args.out.suffix}")

    # Guard against a literal </script> inside string values closing the tag.
    # Inside a JSON string, \/ is a legal escape for /, so this stays valid JSON.
    data_json = json.dumps(data).replace("</", "<\\/")
    page = PAGE.format(
        title=args.title,
        generated=now.strftime("%Y-%m-%d %H:%M:%S"),
        data_json=data_json,
    )

    # Everything under one top-level (timestamped) folder so unzipping stays tidy.
    root = out.stem
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{root}/index.html", page)
        for abspath, rel in files:
            zf.write(abspath, f"{root}/{rel}")

    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({len(data)} experiments, {len(files)} figures, {size_mb:.1f} MB)")
    print(f"recipient: unzip, then open {root}/index.html")


if __name__ == "__main__":
    main()

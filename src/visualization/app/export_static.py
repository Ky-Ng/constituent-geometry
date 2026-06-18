"""Export the attention-heatmap viewer as a shareable zip bundle.

Unlike server.py (which serves figures live over Flask), this walks the
``experiments/*/figures/`` tree once and writes a single .zip:

    attention_heatmaps_<YYYYMMDD_HHMMSS>/
        index.html                       <- dropdown UX, vanilla JS, no server
        figures/<exp>/<folder>/*.png     <- the PNGs, copied verbatim

The manifest now carries ``layer_key`` and ``attn_type`` per PNG so the viewer
can filter without a server. Filter behaviour mirrors server.py:
  - layer filter: shown when any PNG in the folder has a layer_{N}_ prefix;
    options built dynamically from the folder's PNGs; selections persist across
    folder changes (reset on experiment change).
  - type filter: shown for vaswani paths only; options fixed; selections persist
    across both folder and experiment changes.

    uv run python -m visualization.app.export_static
    uv run python -m visualization.app.export_static --out ~/share/heatmaps.zip
    uv run python -m visualization.app.export_static --experiment 09_multi_seed
"""
from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from .server import (
    EXP_ROOT,
    VASWANI_ATTN_TYPES,
    discover_experiments,
    discover_folders,
    is_gpt2_path,
    is_vaswani_path,
    list_pngs,
    parse_png_attrs,
)


def collect(
    experiments: list[str],
) -> tuple[dict[str, dict[str, list[dict]]], list[tuple[Path, str]]]:
    """Returns (manifest, files).

    manifest: {experiment: {folder: [{"name", "src", "layer_key", "attn_type"}, ...]}}
    files:    [(abspath, zip-relative-path), ...]
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
                parts = ("figures", exp, *folder.split("/"), name)
                src = "/".join(quote(p) for p in parts)
                lk, at = parse_png_attrs(name)
                items.append({"name": name, "src": src,
                               "layer_key": lk, "attn_type": at})
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
  .filter-row {{ display: flex; gap: 24px; align-items: flex-start; margin-top: 8px; flex-wrap: wrap; }}
  .filter-group {{ display: flex; align-items: center; gap: 8px; }}
  .filter-group .checks label {{ margin-right: 10px; cursor: pointer; }}
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
<div class="filter-row">
  <div class="filter-group" id="layer-container" style="display:none">
    <label>layer:</label>
    <span class="checks" id="layer-filters"></span>
  </div>
  <div class="filter-group" id="attn-type-container" style="display:none">
    <label>type:</label>
    <span class="checks" id="attn-type-filters"></span>
  </div>
</div>
<hr>
<div id="gallery"></div>
<script id="data" type="application/json">{data_json}</script>
<script>
const DATA       = JSON.parse(document.getElementById("data").textContent);
const expSel     = document.getElementById("experiment");
const folderSel  = document.getElementById("folder");
const gallery    = document.getElementById("gallery");
const VASWANI_ATTN_TYPES = {vaswani_attn_types_json};

// --- model-type detection (mirrors server.py) ---
function isGpt2(expName, folderName) {{
  return (expName + "/" + folderName).toLowerCase().includes("gpt2");
}}
function isVaswani(expName, folderName) {{
  return (expName + "/" + folderName).toLowerCase().includes("vaswani");
}}

// --- layer filter state ---
// layerState tracks what the user has checked per layer key.
// Keys present and true = checked; present and false = explicitly unchecked.
// Keys absent = newly seen, will default to checked.
const layerState = {{}};

function buildLayerCheckboxes(availableLayers) {{
  const c = document.getElementById("layer-filters");
  c.innerHTML = "";
  for (const l of availableLayers) {{
    if (!(l in layerState)) layerState[l] = true;  // new layer -> default checked
    const lbl = document.createElement("label");
    const cb  = document.createElement("input");
    cb.type = "checkbox"; cb.value = l; cb.checked = layerState[l];
    cb.addEventListener("change", (e) => {{ layerState[l] = e.target.checked; render(); }});
    lbl.appendChild(cb); lbl.appendChild(document.createTextNode(" " + l));
    c.appendChild(lbl);
  }}
}}

function getAvailableLayers(expName, folderName) {{
  const items = ((DATA[expName] || {{}})[folderName]) || [];
  const keys = [...new Set(items.map(it => it.layer_key).filter(l => l !== null))];
  return keys.sort();
}}

// --- type filter state (persists in DOM, never reset) ---
function buildTypeCheckboxes() {{
  const c = document.getElementById("attn-type-filters");
  if (c.childElementCount > 0) return;  // already built
  for (const t of VASWANI_ATTN_TYPES) {{
    const lbl = document.createElement("label");
    const cb  = document.createElement("input");
    cb.type = "checkbox"; cb.value = t; cb.checked = true;
    cb.addEventListener("change", render);
    lbl.appendChild(cb); lbl.appendChild(document.createTextNode(" " + t));
    c.appendChild(lbl);
  }}
}}

function getCheckedValues(containerId) {{
  const c = document.getElementById(containerId);
  return new Set([...c.querySelectorAll("input:checked")].map(cb => cb.value));
}}

// --- render ---
function fill(sel, values) {{
  sel.innerHTML = "";
  for (const v of values) {{
    const o = document.createElement("option"); o.value = v; o.textContent = v;
    sel.appendChild(o);
  }}
}}

function render() {{
  const expName    = expSel.value;
  const folderName = folderSel.value;
  const checkedLayers = getCheckedValues("layer-filters");
  const checkedTypes  = getCheckedValues("attn-type-filters");
  const gpt2 = isGpt2(expName, folderName);

  gallery.innerHTML = "";
  const items = ((DATA[expName] || {{}})[folderName]) || [];
  const filtered = items.filter(it => {{
    if (it.layer_key !== null && !checkedLayers.has(it.layer_key)) return false;
    if (!gpt2 && it.attn_type !== null && !checkedTypes.has(it.attn_type)) return false;
    return true;
  }});
  if (!filtered.length) {{
    gallery.innerHTML = "<em>nothing matches the current filter</em>"; return;
  }}
  for (const it of filtered) {{
    const wrap = document.createElement("div"); wrap.className = "fig";
    const name = document.createElement("div"); name.className = "name"; name.textContent = it.name;
    const img  = document.createElement("img");
    img.src = it.src; img.loading = "lazy"; img.alt = it.name;
    wrap.appendChild(name); wrap.appendChild(img);
    gallery.appendChild(wrap);
  }}
}}

function onFolder() {{
  const expName    = expSel.value;
  const folderName = folderSel.value;
  const gpt2    = isGpt2(expName, folderName);
  const vaswani = isVaswani(expName, folderName);

  // Layer filter: rebuild with current layerState (persists user unchecks).
  const layers = getAvailableLayers(expName, folderName);
  document.getElementById("layer-container").style.display = layers.length ? "" : "none";
  buildLayerCheckboxes(layers);

  // Type filter: show for vaswani; selections already persist in DOM.
  buildTypeCheckboxes();
  document.getElementById("attn-type-container").style.display = vaswani ? "" : "none";

  render();
}}

function onExp() {{
  // Reset layer state on experiment change so every folder starts all-checked.
  for (const k in layerState) delete layerState[k];
  fill(folderSel, Object.keys(DATA[expSel.value] || {{}}));
  onFolder();
}}

expSel.addEventListener("change", onExp);
folderSel.addEventListener("change", onFolder);

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

    now   = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    out   = args.out.with_name(f"{args.out.stem}_{stamp}{args.out.suffix}")

    data_json = json.dumps(data).replace("</", "<\\/")
    page = PAGE.format(
        title=args.title,
        generated=now.strftime("%Y-%m-%d %H:%M:%S"),
        data_json=data_json,
        vaswani_attn_types_json=json.dumps(VASWANI_ATTN_TYPES),
    )

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

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CJK_FONTS = ['Noto Sans CJK SC', 'Noto Sans CJK SC Regular', 'Noto Sans Mono CJK SC', 'Noto Sans Mono CJK SC Regular', 'Noto Serif CJK SC']
NOTO_CJK_SC_SANS_ID = 0

def plot_heatmap(
    *,
    Z: list[list[float]],
    Z_text: list[list[str]],
    x_labels: list[str],
    y_labels: list[str],
    x_axis_label: str,
    y_axis_label: str,
    title: str,
    out_path: str | Path,
    cmap: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
        "match", ["#1097EB", "#EB1010"])
):
    # Add fonts with the `mplfonts` dependency
    plt.rcParams["font.sans-serif"] = CJK_FONTS[NOTO_CJK_SC_SANS_ID] 
    plt.rcParams["axes.unicode_minus"] = False   # keep the minus sign rendering

    fig, ax = plt.subplots(figsize=(len(x_labels) * 1.4, len(y_labels) * 1.0))

    im = ax.imshow(Z, cmap=cmap, vmin=0, vmax=1,
                   origin="lower",      # layers, seq starts at left corner
                   aspect="auto")

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)

    for row in range(len(Z_text)):            # row = layer
        for col in range(len(Z_text[row])):   # col = token position
            ax.text(col, row, Z_text[row][col],   # note: (x=col, y=row)
                    ha="center", va="center", color="white")

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.set_title(title)

    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

"""Plot training/validation curves from a training history."""
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt


def plot_training_curves(
    history: Mapping[str, Sequence],
    save_path: Path | str,
    title: str = "Causal LM training",
    smooth_window: int = 50,
) -> Path:
    """Save a PNG with train-loss (raw + smoothed) and val-loss/val-accuracy."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_step = list(history["train_step"])
    train_loss = list(history["train_loss"])
    val_step = list(history.get("val_step", []))
    val_loss = list(history.get("val_loss", []))
    val_acc = list(history.get("val_accuracy", []))

    has_val = len(val_step) > 0
    has_acc = len(val_acc) > 0

    fig, axes = plt.subplots(
        1, 2 if has_acc else 1, figsize=(11 if has_acc else 6, 4), squeeze=False
    )
    ax_loss = axes[0, 0]
    ax_loss.plot(train_step, train_loss, alpha=0.3, linewidth=0.8, label="train (raw)")
    if smooth_window > 1 and len(train_loss) >= smooth_window:
        smoothed = _moving_avg(train_loss, smooth_window)
        ax_loss.plot(
            train_step[smooth_window - 1:],
            smoothed,
            label=f"train (MA-{smooth_window})",
            linewidth=1.5,
        )
    if has_val:
        ax_loss.plot(val_step, val_loss, marker="o", linewidth=1.5, label="val")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_yscale("log")
    ax_loss.set_title(title)
    ax_loss.legend()
    ax_loss.grid(True, which="both", alpha=0.3)

    if has_acc:
        ax_acc = axes[0, 1]
        ax_acc.plot(val_step, val_acc, marker="o", color="tab:green")
        ax_acc.set_xlabel("step")
        ax_acc.set_ylabel("val token accuracy")
        ax_acc.set_ylim(0.0, 1.0)
        ax_acc.set_title("Token-level accuracy (val)")
        ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path


def _moving_avg(xs: list[float], w: int) -> list[float]:
    if w <= 1:
        return list(xs)
    out, s = [], 0.0
    from collections import deque
    buf: deque = deque(maxlen=w)
    for x in xs:
        buf.append(x)
        s = sum(buf)
        if len(buf) == w:
            out.append(s / w)
    return out

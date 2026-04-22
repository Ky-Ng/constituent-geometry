"""Experiment 01 — 2-layer, 1-head, d_model=3 attention-only decoder on the toy CFG.

Trains a causal-LM that sees both translation directions:
    <hi> s_hi <translate> s_hf <eos>
    <hf> s_hf <translate> s_hi <eos>

Loss is applied on the target side only (everything after <translate> up to and
including <eos>). Saves training history and a PNG of the loss curves.
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

# Put `src/` on the path so we can import the project modules when running the
# script directly (no installed package).
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from architecture.decoder import Decoder, DecoderConfig  # noqa: E402
from training_loops.causal_lm import TrainConfig, evaluate, train  # noqa: E402
from training_loops.data import GrammarTranslationDataset, collate  # noqa: E402
from visualization.training_curves import plot_training_curves  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
RESULTS = EXP_DIR / "results"
FIGURES = EXP_DIR / "figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=REPO_ROOT / "data" / "grammar_samples.csv")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--d_model", type=int, default=3)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=16)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--loss_on", choices=["target", "all"], default="target")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    dataset = GrammarTranslationDataset(
        args.csv, max_seq_len=args.max_seq_len, loss_on=args.loss_on
    )
    print(f"loaded {len(dataset)} encoded sequences from {args.csv}")
    print(f"vocab size: {dataset.vocab_size}")

    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model_cfg = DecoderConfig(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
    )
    model = Decoder(model_cfg)
    print(f"model parameters: {model.num_parameters()}")

    train_cfg = TrainConfig(
        epochs=args.epochs, lr=args.lr, device=args.device, log_every=0
    )
    history = train(model, train_loader, val_loader, train_cfg)

    final_val_loss, final_val_acc = evaluate(model, val_loader, train_cfg.device)
    print(f"\nFinal val loss: {final_val_loss:.4f}  token accuracy: {final_val_acc:.3f}")

    # Persist
    hist_path = RESULTS / "history.json"
    with hist_path.open("w") as f:
        json.dump(
            {
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "model_cfg": asdict(model_cfg),
                "train_cfg": asdict(train_cfg),
                "history": history.to_dict(),
                "final_val_loss": final_val_loss,
                "final_val_accuracy": final_val_acc,
                "num_parameters": model.num_parameters(),
            },
            f,
            indent=2,
        )
    print(f"wrote {hist_path}")

    model_path = RESULTS / "model.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": asdict(model_cfg)}, model_path)
    print(f"wrote {model_path}")

    curve_path = plot_training_curves(
        history.to_dict(),
        FIGURES / "loss_curve.png",
        title=f"2L / 1H / d={args.d_model}  (loss_on={args.loss_on})",
    )
    print(f"wrote {curve_path}")


if __name__ == "__main__":
    main()

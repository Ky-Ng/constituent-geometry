"""Training loop for causal-LM on the HI<->HF translation corpus."""
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader

from architecture.decoder import Decoder, causal_lm_loss


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 3e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float | None = 1.0
    log_every: int = 20
    device: str = "cpu"


@dataclass
class History:
    train_step: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_step: list[int] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@torch.no_grad()
def evaluate(model: Decoder, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    correct, counted = 0, 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        logits, _ = model(tokens, key_padding_mask=attn_mask)
        loss = causal_lm_loss(logits, tokens, loss_mask=loss_mask)
        # weight by number of supervised tokens to get exact corpus avg
        n_sup = loss_mask[:, 1:].sum().item()
        total_loss += float(loss.item()) * n_sup
        total_tokens += int(n_sup)

        pred = logits[:, :-1, :].argmax(dim=-1)
        tgt = tokens[:, 1:]
        m = loss_mask[:, 1:].bool()
        correct += int(((pred == tgt) & m).sum().item())
        counted += int(m.sum().item())

    avg_loss = total_loss / max(total_tokens, 1)
    acc = correct / max(counted, 1)
    return avg_loss, acc


def train(
    model: Decoder,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: TrainConfig,
) -> History:
    device = torch.device(cfg.device)
    model.to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )

    history = History()
    step = 0
    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            logits, _ = model(tokens, key_padding_mask=attn_mask)
            loss = causal_lm_loss(logits, tokens, loss_mask=loss_mask)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            history.train_step.append(step)
            history.train_loss.append(float(loss.item()))
            step += 1

            if cfg.log_every and step % cfg.log_every == 0:
                print(f"  step {step:>5d}  train_loss={loss.item():.4f}")

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, cfg.device)
            history.val_step.append(step)
            history.val_loss.append(val_loss)
            history.val_accuracy.append(val_acc)

        dt = time.time() - t0
        history.epoch_times.append(dt)
        msg = f"epoch {epoch + 1:>3d}/{cfg.epochs}  train_loss={loss.item():.4f}"
        if val_loader is not None:
            msg += f"  val_loss={val_loss:.4f}  val_token_acc={val_acc:.3f}"
        msg += f"  ({dt:.1f}s)"
        print(msg)

    return history

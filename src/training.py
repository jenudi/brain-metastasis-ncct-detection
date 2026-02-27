from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class WarmupResult:
    best_state_dict: Dict[str, torch.Tensor]
    best_val_metric: float
    history: Dict[str, list]


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def _build_optimizer(
    params,
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
    nesterov: bool = False,
):
    name = optimizer_name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer_name: {optimizer_name}")


def freeze_all_train_head(model: nn.Module, head_attr: str) -> nn.Module:
    """
    Freeze entire model and unfreeze only model.<head_attr>.
    Returns the head module.
    """
    _set_requires_grad(model, False)
    if not hasattr(model, head_attr):
        raise AttributeError(f"Model has no attribute '{head_attr}' (head).")
    head = getattr(model, head_attr)
    _set_requires_grad(head, True)
    return head


def freeze_backbone_train_head(
    model: nn.Module,
    backbone_attr: str,
    head_attr: str,
) -> Tuple[nn.Module, nn.Module]:
    """
    Freeze model.<backbone_attr> and unfreeze model.<head_attr>.
    Returns (backbone_module, head_module).
    """
    if not hasattr(model, backbone_attr):
        raise AttributeError(f"Model has no attribute '{backbone_attr}' (backbone).")
    if not hasattr(model, head_attr):
        raise AttributeError(f"Model has no attribute '{head_attr}' (head).")

    backbone = getattr(model, backbone_attr)
    head = getattr(model, head_attr)

    _set_requires_grad(backbone, False)
    _set_requires_grad(head, True)

    return backbone, head


def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Supports:
      - dict with keys: image/label (or x/y)
      - tuple/list: (x, y) or (x, y, id)
    """
    if isinstance(batch, dict):
        x = batch.get("image", batch.get("x"))
        y = batch.get("label", batch.get("y"))
        if x is None or y is None:
            raise KeyError(f"Dict batch must contain image/label (or x/y). Keys: {list(batch.keys())}")
        return x, y

    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            x, y = batch
            return x, y
        if len(batch) >= 3:
            x, y, _ = batch[0], batch[1], batch[2]
            return x, y

    raise TypeError(f"Unsupported batch type/format: {type(batch)}")


@torch.no_grad()
def evaluate_binary(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    return_preds: bool = False
) -> Dict[str, float]:
    """
    Computes:
      - val_loss (if criterion provided)
      - val_acc (threshold 0.5)
      - val_auroc, val_auprc (if sklearn available and both classes exist)
    """
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0

    probs_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []

    for batch in loader:
        x, y = _unpack_batch(batch)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        logits = model(x).squeeze(-1)

        if criterion is not None:
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * x.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        correct += int((preds == y).sum().item())
        n += x.size(0)

        probs_all.append(probs.detach().cpu())
        y_all.append(y.detach().cpu())

    out: Dict[str, float] = {}
    if criterion is not None:
        out["val_loss"] = total_loss / max(1, n)
    out["val_acc"] = correct / max(1, n)

    try:
        import numpy as np
        from sklearn.metrics import roc_auc_score, average_precision_score

        y_true = torch.cat(y_all).numpy().astype(np.int32)
        y_prob = torch.cat(probs_all).numpy()

        if len(set(y_true.tolist())) > 1:
            out["val_auroc"] = float(roc_auc_score(y_true, y_prob))
            out["val_auprc"] = float(average_precision_score(y_true, y_prob))
        else:
            out["val_auroc"] = float("nan")
            out["val_auprc"] = float("nan")
    except Exception:
        # sklearn missing or other issues -> don't crash warmup
        out["val_auroc"] = float("nan")
        out["val_auprc"] = float("nan")


    if return_preds:
        return out, y_true, y_prob
    return out


def warmup_train_head_only(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    backbone_attr: Optional[str] = "backbone",
    head_attr: str = "classifier",
    epochs: int = 5,
    lr_head: float = 1e-3,
    weight_decay: float = 0.0,
    criterion: Optional[nn.Module] = None,
    amp: bool = True,
    grad_clip_norm: Optional[float] = 1.0,
    metric_to_optimize: str = "val_loss",   # e.g. "val_auprc"
    minimize_metric: bool = True,           # val_loss -> True ; AUPRC/AUROC -> False
    log_every: int = 50,
    early_stopping_patience: Optional[int] = None,
    optimizer_name: str = "adamw",
    optimizer_momentum: float = 0.9,
    optimizer_nesterov: bool = False,
) -> WarmupResult:
    """
    Warmup stage:
      - Option 1 (default): Freeze model.<backbone_attr> and train model.<head_attr>
      - Option 2 (ResNet-friendly): if backbone_attr is None -> freeze ALL, train only model.<head_attr>

    Returns:
      WarmupResult with best_state_dict and history.
    """
    model = model.to(device)

    # Freeze/unfreeze
    if backbone_attr is None:
        freeze_all_train_head(model, head_attr=head_attr)
    else:
        freeze_backbone_train_head(model, backbone_attr=backbone_attr, head_attr=head_attr)

    # Loss
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    # Optimizer only on trainable params (head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found (did you freeze everything?).")

    optimizer = _build_optimizer(
        trainable_params,
        optimizer_name=optimizer_name,
        lr=lr_head,
        weight_decay=weight_decay,
        momentum=optimizer_momentum,
        nesterov=optimizer_nesterov,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_auroc": [],
        "val_auprc": [],
    }

    best_metric = float("inf") if minimize_metric else float("-inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for i, batch in enumerate(train_loader, start=1):
            x, y = _unpack_batch(batch)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")):
                logits = model(x).squeeze(-1)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * x.size(0)
            n_seen += x.size(0)

            if log_every and (i % log_every == 0):
                pass  # keep quiet by default

        train_loss = running_loss / max(1, n_seen)
        history["train_loss"].append(train_loss)

        # Validate
        val_metrics = evaluate_binary(model, val_loader, device, criterion=criterion)
        history["val_loss"].append(val_metrics.get("val_loss", float("nan")))
        history["val_acc"].append(val_metrics.get("val_acc", float("nan")))
        history["val_auroc"].append(val_metrics.get("val_auroc", float("nan")))
        history["val_auprc"].append(val_metrics.get("val_auprc", float("nan")))

        current = val_metrics.get(metric_to_optimize)
        if current is None:
            raise ValueError(
                f"metric_to_optimize='{metric_to_optimize}' not found in val metrics: {list(val_metrics.keys())}"
            )

        improved = (current < best_metric) if minimize_metric else (current > best_metric)
        if improved:
            best_metric = float(current)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience is not None
            and early_stopping_patience >= 0
            and epochs_without_improvement > early_stopping_patience
        ):
            break

    return WarmupResult(
        best_state_dict=best_state,
        best_val_metric=best_metric,
        history=history,
    )


def finetune_all(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    criterion: Optional[nn.Module] = None,
    amp: bool = True,
    grad_clip_norm: Optional[float] = 1.0,
    metric_to_optimize: str = "val_auprc",   # e.g. "val_loss", "val_auprc", "val_auroc"
    minimize_metric: bool = False,           # AUPRC/AUROC -> False ; loss -> True
    log_every: int = 50,
    early_stopping_patience: Optional[int] = None,
    trainable_module_attrs: Optional[List[str]] = None,
    optimizer_name: str = "adamw",
    optimizer_momentum: float = 0.9,
    optimizer_nesterov: bool = False,
) -> WarmupResult:
    """
    Full fine-tuning stage:
      1) Unfreeze all params
      2) Train whole model for a few epochs
    Returns same structure as warmup for easy integration.
    """
    model = model.to(device)

    # By default unfreeze all; optionally restrict fine-tuning to selected modules.
    if trainable_module_attrs is None:
        _set_requires_grad(model, True)
    else:
        _set_requires_grad(model, False)
        for attr in trainable_module_attrs:
            if not hasattr(model, attr):
                raise AttributeError(f"Model has no attribute '{attr}' in trainable_module_attrs.")
            _set_requires_grad(getattr(model, attr), True)

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found.")

    optimizer = _build_optimizer(
        trainable_params,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        momentum=optimizer_momentum,
        nesterov=optimizer_nesterov,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_auroc": [],
        "val_auprc": [],
    }

    best_metric = float("inf") if minimize_metric else float("-inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for i, batch in enumerate(train_loader, start=1):
            x, y = _unpack_batch(batch)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")):
                logits = model(x).squeeze(-1)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * x.size(0)
            n_seen += x.size(0)

        train_loss = running_loss / max(1, n_seen)
        history["train_loss"].append(train_loss)

        val_metrics = evaluate_binary(model, val_loader, device, criterion=criterion)
        history["val_loss"].append(val_metrics.get("val_loss", float("nan")))
        history["val_acc"].append(val_metrics.get("val_acc", float("nan")))
        history["val_auroc"].append(val_metrics.get("val_auroc", float("nan")))
        history["val_auprc"].append(val_metrics.get("val_auprc", float("nan")))

        current = val_metrics.get(metric_to_optimize)
        if current is None:
            raise ValueError(
                f"metric_to_optimize='{metric_to_optimize}' not found in val metrics: {list(val_metrics.keys())}"
            )

        improved = (current < best_metric) if minimize_metric else (current > best_metric)
        if improved:
            best_metric = float(current)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience is not None
            and early_stopping_patience >= 0
            and epochs_without_improvement > early_stopping_patience
        ):
            break

    return WarmupResult(
        best_state_dict=best_state,
        best_val_metric=best_metric,
        history=history,
    )

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from src.dataset import BrainCTDataset
from src.eval import eval_many_thresholds
from src.models import build_resnet18_binary
from src.training import evaluate_binary, finetune_all, warmup_train_head_only
from src.transforms import build_transforms


def _resolve_cv_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(config or {})
    default_wd = cfg.get("weight_decay", 1e-4)
    return {
        "epochs_head": cfg.get("epochs_head", 5),
        "epochs_ft": cfg.get("epochs_ft", 12),
        "optimizer": cfg.get("optimizer", "adamw"),
        "optimizer_momentum": cfg.get("optimizer_momentum", 0.9),
        "optimizer_nesterov": cfg.get("optimizer_nesterov", False),
        "lr_head": cfg.get("lr_head", 1e-3),
        "lr_ft": cfg.get("lr_ft", 1e-4),
        "weight_decay_head": cfg.get("weight_decay_head", default_wd),
        "weight_decay_ft": cfg.get("weight_decay_ft", default_wd),
        "patience_head": cfg.get("patience_head", 2),
        "patience_ft": cfg.get("patience_ft", 3),
        "dropout_p": cfg.get("dropout_p", 0.3),
        "ft_trainable_attrs": cfg.get("ft_trainable_attrs", None),
        "label_smoothing": cfg.get("label_smoothing", 0.0),
        "use_cosine_schedule": cfg.get("use_cosine_schedule", False),
    }


def _run_single_fold(
    *,
    fold: int,
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    X: pd.Series,
    y: pd.Series,
    ids: pd.Index,
    train_tfm,
    val_tfm,
    cfg: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    pretrained: bool,
) -> Dict[str, Any]:
    X_train = X.iloc[train_pos]
    y_train = y.iloc[train_pos]
    ids_train = ids[train_pos]

    X_val = X.iloc[val_pos]
    y_val = y.iloc[val_pos]
    ids_val = ids[val_pos]

    train_ds = BrainCTDataset(X_train, y_train, ids_train, transform=train_tfm)
    val_ds = BrainCTDataset(X_val, y_val, ids_val, transform=val_tfm)

    counts = y_train.value_counts()
    neg = int(counts.get(0, 0))
    pos = int(counts.get(1, 0))
    if neg == 0 or pos == 0:
        raise ValueError("Training fold must contain both classes for WeightedRandomSampler.")

    class_weights = {0: 1.0 / neg, 1: 1.0 / pos}
    sample_weights = y_train.map(class_weights).to_numpy(dtype=np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    train_eval_ds = BrainCTDataset(X_train, y_train, ids_train, transform=val_tfm)
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    model = build_resnet18_binary(pretrained=pretrained, dropout_p=cfg["dropout_p"]).to(device)
    criterion = nn.BCEWithLogitsLoss()

    warmup_res = warmup_train_head_only(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        backbone_attr=None,
        head_attr="fc",
        epochs=cfg["epochs_head"],
        lr_head=cfg["lr_head"],
        weight_decay=cfg["weight_decay_head"],
        criterion=criterion,
        metric_to_optimize="val_auprc",
        minimize_metric=False,
        early_stopping_patience=cfg["patience_head"],
        optimizer_name=cfg["optimizer"],
        optimizer_momentum=cfg["optimizer_momentum"],
        optimizer_nesterov=cfg["optimizer_nesterov"],
        label_smoothing=cfg["label_smoothing"],
        use_cosine_schedule=cfg["use_cosine_schedule"],
    )
    model.load_state_dict(warmup_res.best_state_dict)

    ft_res = finetune_all(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=cfg["epochs_ft"],
        lr=cfg["lr_ft"],
        weight_decay=cfg["weight_decay_ft"],
        criterion=criterion,
        metric_to_optimize="val_auprc",
        minimize_metric=False,
        early_stopping_patience=cfg["patience_ft"],
        trainable_module_attrs=cfg["ft_trainable_attrs"],
        optimizer_name=cfg["optimizer"],
        optimizer_momentum=cfg["optimizer_momentum"],
        optimizer_nesterov=cfg["optimizer_nesterov"],
        label_smoothing=cfg["label_smoothing"],
        use_cosine_schedule=cfg["use_cosine_schedule"],
    )
    model.load_state_dict(ft_res.best_state_dict)

    _, y_true, y_prob = evaluate_binary(
        model=model,
        loader=val_loader,
        device=device,
        criterion=None,
        return_preds=True,
    )
    _, y_true_trn, y_prob_trn = evaluate_binary(
        model=model,
        loader=train_eval_loader,
        device=device,
        criterion=None,
        return_preds=True,
    )
    auc_val = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    auc_trn = (
        float(roc_auc_score(y_true_trn, y_prob_trn))
        if len(np.unique(y_true_trn)) > 1
        else float("nan")
    )

    return {
        "fold": fold,
        "metric": float(ft_res.best_val_metric),
        "history": {"warmup": warmup_res.history, "finetune": ft_res.history},
        "best_state_dict": ft_res.best_state_dict,
        "oof_true": y_true,
        "oof_prob": y_prob,
        "oof_ids": ids_val.to_numpy(),
        "auc_val": auc_val,
        "auc_trn": auc_trn,
    }


def run_cv(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    *,
    k_folds: int = 5,
    seed: int = 42,
    out_size: int = 256,
    batch_size: int = 16,# 16
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    pretrained: bool = True,
    show_progress: bool = True,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    cfg = _resolve_cv_config(config)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_pool = df[~df["test"]]
    X = df_pool["img"]
    y = df_pool["label"]
    ids = df_pool.index

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    train_tfm = build_transforms(out_size=out_size, train=True)
    val_tfm = build_transforms(out_size=out_size, train=False)

    cv_metrics: Dict[int, float] = {}
    cv_histories: Dict[int, Dict[str, Any]] = {}
    best_models: Dict[int, Dict[str, torch.Tensor]] = {}
    oof_true, oof_prob, oof_ids = [], [], []
    y_true_fold: Dict[int, np.ndarray] = {}
    y_score_fold: Dict[int, np.ndarray] = {}
    df_all_parts = []

    iterator = skf.split(X, y)
    if show_progress:
        iterator = tqdm(iterator, total=k_folds, desc="CV folds")

    for fold, (train_pos, val_pos) in enumerate(iterator, start=1):
        fold_out = _run_single_fold(
            fold=fold,
            train_pos=train_pos,
            val_pos=val_pos,
            X=X,
            y=y,
            ids=ids,
            train_tfm=train_tfm,
            val_tfm=val_tfm,
            cfg=cfg,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            pretrained=pretrained,
        )

        cv_metrics[fold] = fold_out["metric"]
        cv_histories[fold] = fold_out["history"]
        best_models[fold] = fold_out["best_state_dict"]
        oof_true.append(fold_out["oof_true"])
        oof_prob.append(fold_out["oof_prob"])
        oof_ids.append(fold_out["oof_ids"])
        y_true_fold[fold] = fold_out["oof_true"]
        y_score_fold[fold] = fold_out["oof_prob"]
        df_all_parts.append(
            eval_many_thresholds(
                y_true=fold_out["oof_true"],
                y_score=fold_out["oof_prob"],
                thresholds=thresholds,
                fold=fold,
            )
        )
        df_all_parts[-1]["roc_auc"] = fold_out["auc_val"]
        df_all_parts[-1]["roc_auc_trn"] = fold_out["auc_trn"]

        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"fold": fold, "best_metric": f"{fold_out['metric']:.4f}"})

    metric_values = list(cv_metrics.values())
    mean_metric = float(np.mean(metric_values)) if metric_values else float("nan")
    std_metric = float(np.std(metric_values)) if metric_values else float("nan")
    best_fold = max(cv_metrics, key=cv_metrics.get) if cv_metrics else None
    df_all = pd.concat(df_all_parts, ignore_index=True) if df_all_parts else pd.DataFrame()
    df_summary = (
        df_all.groupby("threshold")
        .agg(
            sens_mean=("sens", "mean"),
            sens_std=("sens", "std"),
            spec_mean=("spec", "mean"),
            spec_std=("spec", "std"),
            ppv_mean=("ppv", "mean"),
            ppv_std=("ppv", "std"),
            npv_mean=("npv", "mean"),
            npv_std=("npv", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            acc_mean=("accuracy_avg", "mean"),
            acc_std=("accuracy_avg", "std"),
            ap_mean=("precision_avg", "mean"),
            ap_std=("precision_avg", "std"),
            auc_mean_trn=("roc_auc_trn", "mean"),
            auc_mean=("roc_auc", "mean"),
            auc_std=("roc_auc", "std"),
            tp_sum=("tp", "sum"),
            fp_sum=("fp", "sum"),
            tn_sum=("tn", "sum"),
            fn_sum=("fn", "sum"),
        )
        .reset_index()
        if not df_all.empty
        else pd.DataFrame()
    )

    return {
        "mean_metric": mean_metric,
        "std_metric": std_metric,
        "fold_metrics": cv_metrics,
        "histories": cv_histories,
        "best_fold": best_fold,
        #"best_models": best_models,
        "oof_true": np.concatenate(oof_true) if oof_true else np.array([]),
        "oof_prob": np.concatenate(oof_prob) if oof_prob else np.array([]),
        "oof_ids": np.concatenate(oof_ids) if oof_ids else np.array([]),
        "y_true_fold": y_true_fold,
        "y_score_fold": y_score_fold,
        "df_all": df_all,
        "df_summary": df_summary,
    }

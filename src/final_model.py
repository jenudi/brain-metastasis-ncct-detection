# Train final model on ALL non-test samples using cfg, then score TEST only

from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset import BrainCTDataset
from src.models import build_resnet18_binary
from src.training import warmup_train_head_only, finetune_all
from src.transforms import build_transforms


def train_final_and_score_test(
    features_df,
    cfg,
    batch_size=64,
    num_workers=8,
    out_size=256,
    pretrained=True,
    holdout_frac=0.10,
    holdout_seed=42,
    device: Optional[torch.device] = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1) split non-test into train + small holdout (holdout only for stopping/model selection)
    df_all_train = features_df[~features_df["test"]].copy()
    df_train, df_holdout = train_test_split(
        df_all_train,
        test_size=holdout_frac,
        random_state=holdout_seed,
        stratify=df_all_train["label"],
    )

    X_train = df_train["img"]
    y_train = df_train["label"]
    ids_train = df_train.index

    train_tfm = build_transforms(out_size=out_size, train=True)
    eval_tfm = build_transforms(out_size=out_size, train=False)

    train_ds = BrainCTDataset(X_train, y_train, ids_train, transform=train_tfm)
    X_holdout = df_holdout["img"]
    y_holdout = df_holdout["label"]
    ids_holdout = df_holdout.index
    train_eval_ds = BrainCTDataset(X_holdout, y_holdout, ids_holdout, transform=eval_tfm)

    counts = y_train.value_counts()
    class_weights = {0: 1.0 / int(counts.get(0, 1)), 1: 1.0 / int(counts.get(1, 1))}
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
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    model = build_resnet18_binary(pretrained=pretrained, dropout_p=cfg.get("dropout_p", 0.3)).to(device)
    criterion = nn.BCEWithLogitsLoss()

    warm = warmup_train_head_only(
        model=model,
        train_loader=train_loader,
        val_loader=train_eval_loader,
        device=device,
        backbone_attr=None,
        head_attr="fc",
        epochs=cfg.get("epochs_head", 5),
        lr_head=cfg.get("lr_head", 1e-3),
        weight_decay=cfg.get("weight_decay_head", cfg.get("weight_decay", 1e-4)),
        criterion=criterion,
        metric_to_optimize="val_auprc",
        minimize_metric=False,
        early_stopping_patience=cfg.get("patience_head", 2),
        optimizer_name=cfg.get("optimizer", "adamw"),
        optimizer_momentum=cfg.get("optimizer_momentum", 0.9),
        optimizer_nesterov=cfg.get("optimizer_nesterov", False),
        label_smoothing=cfg.get("label_smoothing", 0.0),
        use_cosine_schedule=cfg.get("use_cosine_schedule", False),
    )
    model.load_state_dict(warm.best_state_dict)

    ft = finetune_all(
        model=model,
        train_loader=train_loader,
        val_loader=train_eval_loader,
        device=device,
        epochs=cfg.get("epochs_ft", 12),
        lr=cfg.get("lr_ft", 1e-4),
        weight_decay=cfg.get("weight_decay_ft", cfg.get("weight_decay", 1e-4)),
        criterion=criterion,
        metric_to_optimize="val_auprc",
        minimize_metric=False,
        early_stopping_patience=cfg.get("patience_ft", 3),
        trainable_module_attrs=cfg.get("ft_trainable_attrs", ["layer3", "layer4", "fc"]),
        optimizer_name=cfg.get("optimizer", "adamw"),
        optimizer_momentum=cfg.get("optimizer_momentum", 0.9),
        optimizer_nesterov=cfg.get("optimizer_nesterov", False),
        label_smoothing=cfg.get("label_smoothing", 0.0),
        use_cosine_schedule=cfg.get("use_cosine_schedule", False),
    )
    model.load_state_dict(ft.best_state_dict)

    # 2) score test only
    df_test = features_df[features_df["test"]].copy()
    test_ds = BrainCTDataset(df_test["img"], labels=df_test["label"], ids=df_test.index, transform=eval_tfm)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    model.eval()
    test_ids, test_prob = [], []
    with torch.no_grad():
        for x, _, sid in test_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x).squeeze(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            test_prob.extend(probs.tolist())
            test_ids.extend(list(sid))

    test_pred_df = pd.DataFrame({"id": test_ids, "prob": test_prob}).set_index("id").loc[df_test.index]
    return model, test_pred_df




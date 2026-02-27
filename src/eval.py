import numpy as np
import pandas as pd
from sklearn import metrics as sm
import matplotlib.pyplot as plt
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from typing import Dict
from pathlib import Path

from src.preprocessing import brain_window, crop_to_skull_bbox
from src.utils import load_raw_hu


def calculate_metrics(y_true, y_pred, y_score=None):
    """
    Metrics without CI. If y_score is provided, uses it for average precision (AP).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = sm.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    spec = sm.recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    sens = sm.recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    ppv  = sm.precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    npv  = np.nan if (tn + fn) == 0 else tn / (tn + fn)

    f1 = sm.f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    accuracy_avg = sm.balanced_accuracy_score(y_true, y_pred)

    # Average precision: prefer scores (standard); fallback to y_pred if not provided
    if y_score is not None:
        y_score = np.asarray(y_score)
        precision_avg = sm.average_precision_score(y_true, y_score)
    else:
        precision_avg = sm.average_precision_score(y_true, y_pred)

    return {
        "sens": float(sens),
        "spec": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),

        "f1": float(f1),
        "precision_avg": float(precision_avg),
        "accuracy_avg": float(accuracy_avg),

        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "p": int(tp + fn),
        "n": int(tn + fp),
    }


def eval_many_thresholds(
    y_true,
    y_score,                 # predict_proba[:,1]
    thresholds=None,
    fold=None,
):
    """
    Evaluates metrics across thresholds, WITHOUT CI columns.
    Also computes AP from y_score (same value across thresholds within a fold).
    """
    if thresholds is None:
        thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)

    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score)

    rows = []
    for t in thresholds:
        y_pred = (y_score_arr >= t).astype(int)

        rs = calculate_metrics(
            y_true=y_true_arr,
            y_pred=y_pred,
            y_score=y_score_arr,   # <-- AP computed from scores (standard)
        )

        rows.append({
            "fold": fold,
            "threshold": float(t),

            "sens": rs["sens"],
            "spec": rs["spec"],
            "ppv":  rs["ppv"],
            "npv":  rs["npv"],

            "f1": rs["f1"],
            "precision_avg": rs["precision_avg"],
            "accuracy_avg": rs["accuracy_avg"],

            "tp": rs["tp"], "tn": rs["tn"], "fp": rs["fp"], "fn": rs["fn"],
            "p": rs["p"], "n": rs["n"],
        })

    return pd.DataFrame(rows)


def sample_and_plot_raw_mask_prep_grid(
    df: pd.DataFrame,
    n_samples: int,
    *,
    label: int,
    path_col: str = "path",
    label_col: str = "label",
    test_col: str = "test",
    percentile: float = 98.0,
    window_center: float = 40.0,
    window_width: float = 80.0,
    random_state: int = 42,
    figsize_per_cell: float = 2.3,
    save_path: str | Path | None = None,
):
    """
    1) Sample n train rows from a requested label (0 or 1).
    2) Read DICOM and compute raw HU with -1024 floor by p1 threshold.
    3) Build skull mask by hu > percentile(hu, percentile).
    4) Crop with crop_to_skull_bbox.
    5) Apply brain windowing on cropped image.
    6) Return four lists and plot a grid with rows=n, cols=4:
       raw, skull_mask, cropped, brain-windowed.
    """
    if path_col not in df.columns:
        raise ValueError(f"Missing required column '{path_col}'.")
    if label_col not in df.columns:
        raise ValueError(f"Missing required column '{label_col}'.")
    if test_col not in df.columns:
        raise ValueError(f"Missing required column '{test_col}'.")
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")
    if label not in (0, 1):
        raise ValueError("label must be 0 or 1.")

    train_df = df[df[test_col] == False]
    lbl_df = train_df[train_df[label_col] == label]

    if len(lbl_df) < n_samples:
        raise ValueError(
            f"Not enough train samples for label={label}. "
            f"Have {len(lbl_df)}, requested={n_samples}."
        )

    sample_df = lbl_df.sample(n=n_samples, random_state=random_state)

    raw_list = []
    skull_masks = []
    prep_list = []
    window_list = []
    sample_ids = []

    def _process_subset(
        sample_df: pd.DataFrame,
        raw_out: list,
        mask_out: list,
        prep_out: list,
        window_out: list,
        ids_out: list,
    ) -> None:
        for sample_id, row in sample_df.iterrows():
            hu = load_raw_hu(row[path_col])
            thr = np.percentile(hu, percentile)
            skull_mask = hu > thr
            cropped, _ = crop_to_skull_bbox(hu, percentile=percentile)
            windowed = brain_window(cropped, center=window_center, width=window_width).astype(np.float32)

            raw_out.append(hu)
            mask_out.append(skull_mask)
            prep_out.append(cropped)
            window_out.append(windowed)
            ids_out.append(sample_id)

    _process_subset(sample_df, raw_list, skull_masks, prep_list, window_list, sample_ids)

    # Plot rows for selected label, columns: raw | mask | crop | window
    total_rows = n_samples
    fig, axes = plt.subplots(total_rows, 4, figsize=(4 * figsize_per_cell, total_rows * figsize_per_cell))
    if total_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_samples):
        axes[i, 0].imshow(raw_list[i], cmap="gray")
        axes[i, 1].imshow(skull_masks[i], cmap="gray")
        axes[i, 2].imshow(prep_list[i], cmap="gray")
        axes[i, 3].imshow(window_list[i], cmap="gray", vmin=0.0, vmax=1.0)

    axes[0, 0].set_title("Raw HU")
    axes[0, 1].set_title("Skull Mask")
    axes[0, 2].set_title("Cropped")
    axes[0, 3].set_title("Brain Window")

    for r in range(total_rows):
        for c in range(4):
            axes[r, c].axis("off")
        # Keep sample ID visible even with axes hidden.
        axes[r, 0].text(
            -0.03,
            0.5,
            str(sample_ids[r]),
            transform=axes[r, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8,
        )

    fig.suptitle(f"Preprocessing Pipeline Samples (train, label={label}, n={n_samples})", y=1.02)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return {
        "raw_list": raw_list,
        "skull_masks": skull_masks,
        "prep_list": prep_list,
        "window_list": window_list,
        "sample_ids": sample_ids,
    }


def plot_cv_mean_roc(
    y_true_fold: Dict[int, np.ndarray],
    y_score_fold: Dict[int, np.ndarray],
    save_path: str | Path | None = None,
) -> None:
    """
    Plot per-fold ROC curves and the mean ROC across cross-validation folds.
    """
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []
    fig = plt.figure(figsize=(7, 7))
    for fold in sorted(y_true_fold.keys()):
        y_true = y_true_fold[fold]
        y_score = y_score_fold[fold]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Interpolate TPR on common FPR grid
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        plt.plot(fpr, tpr, alpha=0.25, label=f"Fold {fold} (AUC={roc_auc:.2f})")
    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    # Mean ROC curve
    plt.plot(mean_fpr, mean_tpr, linewidth=2, label=f"Mean ROC (AUC={mean_auc:.2f} ± {std_auc:.2f})")

    # ±1 std band (clipped to [0,1])
    tpr_upper = np.clip(mean_tpr + std_tpr, 0, 1)
    tpr_lower = np.clip(mean_tpr - std_tpr, 0, 1)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=0.2, label="±1 std (TPR)")

    # Chance line
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Cross-Validated ROC: mean ± std")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.2)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_cv_mean_pr(
    y_true_fold: Dict[int, np.ndarray],
    y_score_fold: Dict[int, np.ndarray],
    save_path: str | Path | None = None,
) -> None:
    """
    Plot per-fold Precision-Recall curves and the mean PR curve across CV folds.
    """
    mean_recall = np.linspace(0, 1, 200)
    precisions_interp = []
    aps = []
    fig = plt.figure(figsize=(7, 7))

    for fold in sorted(y_true_fold.keys()):
        y_true = y_true_fold[fold]
        y_score = y_score_fold[fold]

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = auc(recall, precision)
        aps.append(ap)

        # Interpolate precision on common recall grid
        # precision_recall_curve returns recall decreasing; reverse for interpolation.
        recall_rev = recall[::-1]
        precision_rev = precision[::-1]
        prec_interp = np.interp(mean_recall, recall_rev, precision_rev)
        precisions_interp.append(prec_interp)

        plt.plot(recall, precision, alpha=0.25, label=f"Fold {fold} (AP={ap:.2f})")

    precisions_interp = np.array(precisions_interp)
    mean_precision = precisions_interp.mean(axis=0)
    std_precision = precisions_interp.std(axis=0)
    mean_ap = float(np.mean(aps))
    std_ap = float(np.std(aps))

    # Mean PR curve
    plt.plot(
        mean_recall,
        mean_precision,
        linewidth=2,
        label=f"Mean PR (AP={mean_ap:.2f} ± {std_ap:.2f})",
    )

    # ±1 std band (clipped to [0,1])
    upper = np.clip(mean_precision + std_precision, 0, 1)
    lower = np.clip(mean_precision - std_precision, 0, 1)
    plt.fill_between(mean_recall, lower, upper, alpha=0.2, label="±1 std (Precision)")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Cross-Validated PR: mean ± std")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.2)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_test_roc_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    save_path_roc: str | Path | None = None,
    save_path_pr: str | Path | None = None,
) -> Dict[str, float]:
    """
    Plot ROC and PR curves for a single test set (no folds).
    Returns {'roc_auc': ..., 'auprc': ...}.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)

    if len(np.unique(y_true_arr)) < 2:
        raise ValueError("Both classes must be present in y_true to plot ROC/PR metrics.")

    roc_auc_val = float(roc_auc_score(y_true_arr, y_score_arr))
    auprc_val = float(average_precision_score(y_true_arr, y_score_arr))

    # ROC
    fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)
    fig_roc = plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC={roc_auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Test ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    if save_path_roc is not None:
        save_path_roc = Path(save_path_roc)
        save_path_roc.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_roc, dpi=200, bbox_inches="tight")
        plt.close(fig_roc)
    else:
        plt.show()

    # PR
    precision, recall, _ = precision_recall_curve(y_true_arr, y_score_arr)
    fig_pr = plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, linewidth=2, label=f"PR (AP={auprc_val:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Test Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.2)
    if save_path_pr is not None:
        save_path_pr = Path(save_path_pr)
        save_path_pr.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_pr, dpi=200, bbox_inches="tight")
        plt.close(fig_pr)
    else:
        plt.show()

    return {"roc_auc": roc_auc_val, "auprc": auprc_val}

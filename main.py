from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.cv import run_cv
from src.eval import plot_cv_mean_pr, plot_cv_mean_roc, plot_test_roc_pr, sample_and_plot_raw_mask_prep_grid
from src.final_model import train_final_and_score_test
from src.preprocessing import compute_dicom_features_df
from src.utils import run_features_df_checks, validate_dataset_structure


DEFAULT_CFG: Dict[str, Any] = {
    "optimizer": "adamw",
    "lr_ft": 0.00015,
    "lr_head": 0.003,
    "weight_decay_head": 0.00005,
    "weight_decay_ft": 0.0007,
    "epochs_head": 12,
    "epochs_ft": 12,
    "patience_head": 2,
    "patience_ft": 2,
    "dropout_p": 0.3,
    "ft_trainable_attrs": None,
}


def _to_builtin(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _to_builtin(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_builtin(v) for v in x]
    if isinstance(x, tuple):
        return [_to_builtin(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _select_threshold(df_summary: pd.DataFrame, strategy: str, sens_target: float) -> Dict[str, Any]:
    if df_summary.empty:
        raise ValueError("df_summary is empty; cannot select threshold.")

    if strategy == "max_f1":
        row = df_summary.sort_values("f1_mean", ascending=False).iloc[0]
        return {"strategy": strategy, "threshold": float(row["threshold"]), "row": _to_builtin(row.to_dict())}

    if strategy == "sens_target":
        cand = df_summary[df_summary["sens_mean"] >= sens_target]
        if len(cand) == 0:
            row = df_summary.sort_values("f1_mean", ascending=False).iloc[0]
            return {
                "strategy": strategy,
                "fallback": "max_f1",
                "sens_target": sens_target,
                "threshold": float(row["threshold"]),
                "row": _to_builtin(row.to_dict()),
            }
        row = cand.sort_values("ppv_mean", ascending=False).iloc[0]
        return {
            "strategy": strategy,
            "sens_target": sens_target,
            "threshold": float(row["threshold"]),
            "row": _to_builtin(row.to_dict()),
        }

    raise ValueError(f"Unknown threshold strategy: {strategy}")


def main() -> None:
    # CLI configuration for data paths, training setup, and selection policies.
    parser = argparse.ArgumentParser(description="Run full clinical-neuro-oncology pipeline.")
    parser.add_argument("--data-root", default="data", help="Data root containing CTs/ and labels csv.")
    parser.add_argument("--labels", default="labels1.csv", help="Labels csv filename inside data-root.")
    parser.add_argument("--cts", default="CTs", help="CT directory name inside data-root.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for data split and CV.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Fraction of samples reserved for the final test set.")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds used in cross-validation.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for CV and final model training.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for feature extraction and DataLoader workers.",
    )
    parser.add_argument("--out-size", type=int, default=256, help="Target spatial image size (out_size x out_size).")
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.10,
        help="Internal holdout fraction from train data for early stopping in final training.",
    )
    parser.add_argument("--prep-plot-n", type=int, default=5, help="Num samples per label for preprocessing plots.")
    parser.add_argument(
        "--threshold-strategy",
        choices=["max_f1", "sens_target"],
        default="sens_target",
        help="Threshold selection policy from CV summary: maximize F1 or enforce a sensitivity target.",
    )
    parser.add_argument(
        "--sens-target",
        type=float,
        default=0.70,
        help="Minimum sensitivity target used only when --threshold-strategy=sens_target.",
    )
    parser.add_argument("--run-grid", action="store_true", help="Run src.grids candidates and pick best by CV mean.")
    parser.add_argument("--config-json", default=None, help="Optional JSON file with a config dict override.")
    args = parser.parse_args()

    # Create a unique output directory for this run and keep all artifacts there.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Running main pipeline for clinical neuro-oncology modeling.")
    print(f"Run ID: {run_id}")
    print(f"Output directory: {out_dir}")
    # Persist raw CLI arguments used for this run.
    with open(out_dir / "run_args.json", "w") as f:
        json.dump(_to_builtin(vars(args)), f, indent=2)

    # 1) Basic dataset integrity check (folder and labels structure).
    print(f"[1/7] Validate data structure in: {args.data_root}")
    validate_dataset_structure(args.data_root, ct_dir_name=args.cts, labels_filename=args.labels)

    # 2) Feature table build from DICOM + labels.
    print("[2/7] Load labels and extract features")
    labels_df = pd.read_csv(Path(args.data_root) / args.labels, index_col="ID")
    ct_dir = Path(args.data_root) / args.cts
    features_df = compute_dicom_features_df(str(ct_dir), labels_df, processes=args.num_workers)
    features_df.to_pickle(out_dir / "features_df.pkl")

    # 3) Save sanity checks on the generated features table.
    print("[3/7] Run features_df data checks")
    checks = run_features_df_checks(features_df)
    with open(out_dir / "features_checks_summary.json", "w") as f:
        json.dump(
            {
                "duplicate_id_count": checks["duplicate_id_count"],
                "error_rows_count": checks["error_rows_count"],
                "missing_key": _to_builtin(checks["missing_key"].to_dict()),
            },
            f,
            indent=2,
        )

    # 4) Create a fixed train/test split and persist it for reproducibility.
    print("[4/7] Create train/test split flags")
    train_idx, test_idx = train_test_split(
        features_df.index,
        test_size=args.test_size,
        stratify=features_df["label"],
        random_state=args.seed,
    )
    features_df.loc[train_idx, "test"] = False
    features_df.loc[test_idx, "test"] = True
    features_df.to_pickle(out_dir / "features_df_split.pkl")

    # 4.5) Save qualitative preprocessing examples for each class.
    print("[4.5/7] Save preprocessing example plots")
    sample_and_plot_raw_mask_prep_grid(
        features_df,
        args.prep_plot_n,
        label=0,
        random_state=args.seed,
        save_path=out_dir / "preprocessing_examples_label0.png",
    )
    sample_and_plot_raw_mask_prep_grid(
        features_df,
        args.prep_plot_n,
        label=1,
        random_state=args.seed + 1,
        save_path=out_dir / "preprocessing_examples_label1.png",
    )

    cfg = dict(DEFAULT_CFG)
    if args.config_json:
        with open(args.config_json) as f:
            cfg = json.load(f)

    # 5) Run CV (optionally over a grid), then save threshold tables and CV curves.
    print("[5/7] Run CV")
    if args.run_grid:
        from src.grids import grid

        grid_results: List[Dict[str, Any]] = []
        for i, item in enumerate(grid, start=1):
            name = item["name"]
            cand_cfg = dict(item["config"])
            t0 = time.time()
            cv_res = run_cv(
                features_df,
                config=cand_cfg,
                k_folds=args.k_folds,
                seed=args.seed,
                out_size=args.out_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                show_progress=True,
            )
            dt = time.time() - t0
            grid_results.append(
                {
                    "name": name,
                    "mean_metric": cv_res["mean_metric"],
                    "std_metric": cv_res["std_metric"],
                    "seconds": dt,
                    "config": cand_cfg,
                }
            )
            print(
                f"  [{i}/{len(grid)}] {name}: {cv_res['mean_metric']:.6f} +/- {cv_res['std_metric']:.6f} ({dt:.1f}s)"
            )

        grid_df = pd.DataFrame(grid_results).sort_values("mean_metric", ascending=False)
        grid_df.to_csv(out_dir / "grid_results.csv", index=False)
        best_name = str(grid_df.iloc[0]["name"])
        cfg = next(x["config"] for x in grid if x["name"] == best_name)
        print(f"Selected best grid config: {best_name}")

    # Persist the final effective config (after optional grid selection / JSON override).
    with open(out_dir / "effective_config.json", "w") as f:
        json.dump(_to_builtin(cfg), f, indent=2)

    # Main CV run for selected configuration.
    cv_res = run_cv(
        features_df,
        config=cfg,
        k_folds=args.k_folds,
        seed=args.seed,
        out_size=args.out_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        show_progress=True,
    )
    cv_res["df_all"].to_csv(out_dir / "cv_thresholds_by_fold.csv", index=False)
    cv_res["df_summary"].to_csv(out_dir / "cv_thresholds_summary.csv", index=False)
    plot_cv_mean_roc(cv_res["y_true_fold"], cv_res["y_score_fold"], save_path=out_dir / "cv_mean_roc.png")
    plot_cv_mean_pr(cv_res["y_true_fold"], cv_res["y_score_fold"], save_path=out_dir / "cv_mean_pr.png")

    # Select operating threshold from CV summary according to requested strategy.
    threshold_info = _select_threshold(cv_res["df_summary"], args.threshold_strategy, args.sens_target)
    threshold = threshold_info["threshold"]
    with open(out_dir / "threshold_selection.json", "w") as f:
        json.dump(_to_builtin(threshold_info), f, indent=2)

    with open(out_dir / "cv_metrics.json", "w") as f:
        json.dump(
            _to_builtin(
                {
                    "mean_metric": cv_res["mean_metric"],
                    "std_metric": cv_res["std_metric"],
                    "fold_metrics": cv_res["fold_metrics"],
                    "config": cfg,
                    "batch_size": args.batch_size,
                    "k_folds": args.k_folds,
                    "seed": args.seed,
                }
            ),
            f,
            indent=2,
        )

    # 6) Train final model on train split and score only the held-out test split.
    print("[6/7] Train final model and score test")
    model, test_pred_df = train_final_and_score_test(
        features_df,
        cfg=cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        out_size=args.out_size,
        holdout_frac=args.holdout_frac,
        holdout_seed=args.seed,
    )
    test_pred_df["pred"] = (test_pred_df["prob"] >= threshold).astype(int)
    test_pred_df["label"] = features_df.loc[test_pred_df.index, "label"].astype(int)
    test_pred_df.to_csv(out_dir / "test_predictions.csv")
    torch.save(model.state_dict(), out_dir / "final_model_state_dict.pt")

    # Compute test metrics and save ROC/PR figures using probability outputs.
    y_true = test_pred_df["label"].to_numpy().astype(int)
    y_prob = test_pred_df["prob"].to_numpy().astype(float)
    y_pred = test_pred_df["pred"].to_numpy().astype(int)

    test_auc = float(roc_auc_score(y_true, y_prob))
    test_auprc = float(average_precision_score(y_true, y_prob))
    curve_metrics = plot_test_roc_pr(
        y_true,
        y_prob,
        save_path_roc=out_dir / "test_roc_curve.png",
        save_path_pr=out_dir / "test_pr_curve.png",
    )

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    test_metrics = {
        "threshold": threshold,
        "test_roc_auc": test_auc,
        "test_auprc": test_auprc,
        "test_roc_auc_from_plot_fn": curve_metrics["roc_auc"],
        "test_auprc_from_plot_fn": curve_metrics["auprc"],
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(_to_builtin(test_metrics), f, indent=2)

    print("[7/7] Done")
    print(f"Outputs saved to: {out_dir}")
    print(f"Test ROC-AUC: {test_auc:.6f}")
    print(f"Test AUPRC:   {test_auprc:.6f}")


if __name__ == "__main__":
    main()

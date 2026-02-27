# Clinical Neuro-Oncology Modeling

End-to-end pipeline for binary classification on brain CT DICOM studies, including:
- DICOM validation and metadata/feature extraction
- HU-based preprocessing and skull-based cropping
- Two-stage transfer learning (head warmup + fine-tuning)
- Cross-validation for model selection and threshold analysis
- Final training and held-out test evaluation
- Saved artifacts and plots for reproducibility

The implementation supports both notebook-driven experimentation (`notebooks/eda.ipynb`) and a reproducible CLI pipeline (`main.py`).

## 1) Project Goal

Predict a binary label (`0/1`) from brain CT slices using a ResNet18 backbone with transfer learning.

The workflow emphasizes:
- strong cross-validation reporting (AUPRC-first),
- threshold selection based on CV behavior (including sensitivity-driven selection),
- and strict separation between CV/selection and final test scoring.

## 2) Repository Structure

```text
clinical-neuro-oncology/
├── main.py
├── requirements.txt
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── transforms.py
│   ├── models.py
│   ├── training.py
│   ├── cv.py
│   ├── final_model.py
│   ├── eval.py
│   ├── grids.py
│   └── utils.py
└── outputs/
```

## Requirements

Tested with:
- Python 3.10+
- Linux (Ubuntu VM)
- PyTorch + CUDA-capable environment (GPU recommended for training speed)
- Dependencies pinned in `requirements.txt`

Install:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Key files

- `main.py`: full production pipeline runner (CLI + artifact saving).
- `notebooks/eda.ipynb`: exploratory workflow and iterative experiments.
- `src/cv.py`: cross-validation pipeline (`run_cv`), fold-level training/evaluation, threshold tables.
- `src/final_model.py`: final train-on-train / score-on-test routine.
- `src/training.py`: warmup/fine-tune training loops, optimizers, AMP, early stopping.
- `src/preprocessing.py`: DICOM -> HU conversion usage, skull mask crop, feature extraction.
- `src/eval.py`: threshold metrics and visualization functions (CV ROC/PR, test ROC/PR, preprocessing grids).

## 3) Data Expectations

`main.py` expects:

```text
data/
├── CTs/
│   ├── ID_xxx.dcm
│   └── ...
└── labels1.csv
```

### Label format

- CSV must include a label column named `Label` (uppercase in raw CSV).
- IDs are expected to match DICOM filename stems (e.g. `ID_123` -> `ID_123.dcm`).

### Validation behavior

`validate_dataset_structure(...)` checks:
- CT folder exists and has DICOM files.
- labels CSV exists and has `Label`.
- duplicate IDs/files.
- class distribution and imbalance.
- ID-to-DICOM filename consistency.

Important: current `validate_dataset_structure` has `expected_num_dicoms=5123` by default in `src/utils.py`.
If your dataset size differs, update that default (or adapt the function to accept a CLI value).

## 4) End-to-End Pipeline (What `main.py` Runs)

`main.py` executes these stages:

1. Validate data structure.
2. Load labels and extract features from all DICOMs via `compute_dicom_features_df(...)`.
3. Run `features_df` quality checks.
4. Create stratified train/test split (`features_df["test"]` flag).
5. Save preprocessing visualization examples for label `0` and label `1`.
6. Run cross-validation:
   - optional grid sweep (`--run-grid`, candidates from `src/grids.py`)
   - selected configuration CV run
   - save fold-level and aggregated threshold tables
   - save mean CV ROC and PR plots
7. Select threshold from CV summary:
   - `max_f1`, or
   - `sens_target` (filter by `sens_mean >= target`, then maximize `ppv_mean`).
8. Train final model on train split with internal holdout for early stopping.
9. Score held-out test only, apply selected threshold, save metrics/predictions/plots/model weights.

## 5) Modeling Approach

## Backbone + head

- Architecture: `torchvision` `resnet18`
- Output head: `Dropout(p=dropout_p) -> Linear(in_features, 1 logit)`
- Binary objective: `BCEWithLogitsLoss`

## Two-stage training

### Stage A: warmup head
- Freeze backbone, train only `fc`.
- Metric to optimize: `val_auprc`.
- Early stopping enabled (`patience_head`).

### Stage B: fine-tuning
- Unfreeze full model or selected modules (`ft_trainable_attrs`).
- Metric to optimize: `val_auprc`.
- Early stopping enabled (`patience_ft`).

## Class imbalance handling

- `WeightedRandomSampler` in training DataLoader.
- Class weights inverse to class frequency per fold/train split.

## Optimizer support

- `adamw` (default)
- `sgd` (with momentum/nesterov options)

## 6) Preprocessing and Input Pipeline

### DICOM -> HU

Using shared logic in `src/utils.py`:
- read pixel array,
- apply `RescaleSlope` and `RescaleIntercept`,
- floor very low intensities (`<= p1`) to `-1024`.

### Skull-based crop

In `crop_to_skull_bbox(...)`:
- threshold at high percentile (default `98.0`) to estimate skull mask,
- compute tight bbox,
- crop HU image to bbox.

### Brain windowing

`brain_window(hu, center=40, width=80)`:
- clip HU to window,
- min-max normalize to `[0, 1]`.

### Transforms

`build_transforms(...)`:
- convert to float tensor (CHW),
- aspect-ratio-preserving letterbox resize,
- optional train augmentations (`RandomAffine`, `RandomHorizontalFlip`),
- repeat to 3 channels,
- ImageNet normalization.

## 7) Cross-Validation Outputs

`run_cv(...)` returns:

- `mean_metric`, `std_metric` (best fold metric aggregate)
- `fold_metrics`
- `histories` (warmup + finetune curves per fold)
- `oof_true`, `oof_prob`, `oof_ids`
- `y_true_fold`, `y_score_fold` (for fold-wise ROC/PR plotting)
- `df_all`: per-fold metrics per threshold
- `df_summary`: threshold-wise mean/std across folds

`df_summary` includes:
- `sens_mean`, `spec_mean`, `ppv_mean`, `npv_mean`
- `f1_mean`, `acc_mean`
- `ap_mean`, `auc_mean`, `auc_mean_trn`, and std columns
- confusion components aggregated (`tp_sum`, `fp_sum`, `tn_sum`, `fn_sum`)

This table is the basis for choosing an operating threshold before touching test results.

## 8) Final Model + Test Evaluation

`train_final_and_score_test(...)`:
- takes all non-test samples,
- makes a small internal holdout (`holdout_frac`) for early stopping only,
- trains with same two-stage routine,
- scores only rows with `test == True`.

Then `main.py`:
- applies selected threshold to `prob`,
- computes confusion (`tp/tn/fp/fn`),
- computes threshold-independent test metrics:
  - ROC-AUC
  - AUPRC
- saves test ROC and PR plots.

## 9) How to Run

## Default run

```bash
python3 main.py
```

This runs the full pipeline:
- data validation,
- feature extraction,
- split creation,
- preprocessing visual diagnostics,
- CV + threshold selection,
- final model training,
- test scoring and plots.

## Show CLI options

```bash
python3 main.py --help
```

## Run with grid search

```bash
python3 main.py --run-grid
```

## Run with custom config JSON

```bash
python3 main.py --config-json path/to/cfg.json
```

## Sensitivity-oriented threshold policy

```bash
python3 main.py --threshold-strategy sens_target --sens-target 0.75
```

## 10) Main CLI Arguments

- `--data-root`: data root directory (default: `data`)
- `--labels`: labels filename (default: `labels1.csv`)
- `--cts`: CT folder name (default: `CTs`)
- `--seed`: global seed for reproducibility
- `--test-size`: held-out test fraction
- `--k-folds`: number of CV folds
- `--batch-size`: training/eval batch size
- `--num-workers`: multiprocessing/DataLoader workers
- `--out-size`: model input image size
- `--holdout-frac`: internal train holdout for early stopping in final stage
- `--prep-plot-n`: examples per class for preprocessing visualization
- `--threshold-strategy`: `max_f1` or `sens_target`
- `--sens-target`: sensitivity target for `sens_target` mode
- `--run-grid`: evaluate `src/grids.py` candidates and select best mean CV score
- `--config-json`: override configuration from JSON

## 11) Output Artifacts

Each run creates:

```text
outputs/run_YYYYMMDD_HHMMSS/
├── run_args.json
├── effective_config.json
├── features_df.pkl
├── features_checks_summary.json
├── features_df_split.pkl
├── preprocessing_examples_label0.png
├── preprocessing_examples_label1.png
├── grid_results.csv                  # only if --run-grid
├── cv_thresholds_by_fold.csv
├── cv_thresholds_summary.csv
├── cv_mean_roc.png
├── cv_mean_pr.png
├── threshold_selection.json
├── cv_metrics.json
├── final_model_state_dict.pt
├── test_predictions.csv
├── test_roc_curve.png
├── test_pr_curve.png
└── test_metrics.json
```

## 12) Reproducibility Notes

`main.py` seeds:
- Python `random`
- NumPy
- PyTorch CPU/CUDA
- `PYTHONHASHSEED`
- cuDNN deterministic mode

Even with deterministic settings, tiny variations can still happen across environments due to CUDA/kernel/library behavior.

## 13) Typical Training Configuration

Current default in `main.py` (`DEFAULT_CFG`):

```python
{
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
    "ft_trainable_attrs": None
}
```

In this codebase, `ft_trainable_attrs=None` means full fine-tuning; lists like `["layer3", "layer4", "fc"]` restrict trainable modules in stage B.

## 14) Notebook vs Pipeline

- `notebooks/eda.ipynb`: experimentation, diagnostics, iterative grid testing, visual checks.
- `main.py`: reproducible production run that persists all key artifacts.

Recommended practice:
- explore in notebook,
- freeze final policy/config,
- run once through `main.py` for reportable outputs.

## 15) Known Caveats / Future Improvements

- `validate_dataset_structure` currently enforces a fixed expected DICOM count by default.
- Feature extraction uses multiprocessing and can be RAM/CPU intensive.
- `run_cv` currently excludes returning per-fold best model state dicts (intentionally commented out to reduce output size).
- Consider adding unit tests and CI for data validation + metric sanity checks.

## 16) License / Usage

No license file is currently included. Add a `LICENSE` if you plan to publish or share externally.

## Evaluation Notes

- CV is performed only on the train pool (`test == False`), using stratified folds.
- Threshold selection is done from CV-derived summaries (`df_summary`) to avoid tuning on test.
- Final test metrics are computed once after fixing config and threshold policy.
- Both threshold-dependent metrics (sensitivity/specificity/PPV/NPV/F1) and threshold-independent metrics (ROC-AUC, AUPRC) are saved.
- Mean CV ROC and PR plots are saved with fold variability context.

## Exploratory Analysis

Exploratory analysis, diagnostics, and iterative experiments are in:

```text
notebooks/eda.ipynb
```

Use the notebook for investigation and `main.py` for final reproducible runs.

## Environment

Experiments in this project were developed and run on a Linux VM setup with Python 3.10 and GPU acceleration (CUDA-enabled PyTorch), suitable for repeated cross-validation and transfer-learning workflows.

## Author

Yonatan Jenudi  
MD | Data Scientist

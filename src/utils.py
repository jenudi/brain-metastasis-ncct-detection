import os
import random
from typing import Any, Dict

import numpy as np
import pandas as pd
import pydicom
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate_dataset_structure(
    data_root: str,
    ct_dir_name: str = "CTs",
    labels_filename: str = "labels1.csv",
    expected_num_dicoms: int = 5123,
) -> None:
    """
    Validate expected dataset structure and integrity:
    data_root/
        ├── CTs/
        │    └── *.dcm
        └── labels1.csv
    Checks:
    - CTs folder exists and contains expected_num_dicoms DICOM files
    - labels1.csv exists and contains a 'Label' column (ID is index or column)
    - Duplicates in labels index
    - Duplicates in CT filenames
    - Class imbalance ratio
    - (Optional but very useful) Cross-check label IDs <-> dicom filenames
    """
    print("\nValidating dataset structure...\n")
    # Root
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Data root not found: {data_root}")

    # CT directory + DICOM count
    ct_dir = os.path.join(data_root, ct_dir_name)
    if not os.path.isdir(ct_dir):
        raise FileNotFoundError(f"CT directory not found: {ct_dir}")

    dcm_files = sorted([f for f in os.listdir(ct_dir) if f.lower().endswith(".dcm")])
    n_dicoms = len(dcm_files)

    if n_dicoms == 0:
        raise ValueError(f"No .dcm files found inside: {ct_dir}")

    print(f"CT folder found: {ct_dir}")
    print(f"   → {n_dicoms} DICOM files detected")

    if expected_num_dicoms is not None and n_dicoms != expected_num_dicoms:
        raise ValueError(
            f"Expected {expected_num_dicoms} DICOM files, but found {n_dicoms} in {ct_dir}"
        )

    # Detect duplicate filenames (rare, but check anyway)
    dup_ct = pd.Series(dcm_files).duplicated().sum()
    if dup_ct > 0:
        print(f"Warning: {dup_ct} duplicate CT filenames detected (unexpected).")
    else:
        print("No duplicate CT filenames detected")

    # Labels file
    labels_path = os.path.join(data_root, labels_filename)
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    df = pd.read_csv(labels_path)

    print(f"\nLabels file found: {labels_path}")
    print(f"   → {len(df)} rows")
    print(f"   → Columns: {list(df.columns)}")

    # Handle your case: ID is index in notebook (index_col='ID') but raw csv might include it as a column.
    if "ID" in df.columns:
        ids = df["ID"].astype(str)
    else:
        # If you already loaded it with index_col='ID', then raw csv might have no ID col.
        # We'll assume the first column is ID in that case.
        ids = df.iloc[:, 0].astype(str)

    # Label column check
    if "Label" not in df.columns:
        raise ValueError("Missing required column 'Label' in labels CSV")

    labels = df["Label"]

    # Duplicate detection in labels
    dup_labels = ids.duplicated().sum()
    print(f"\nDuplicates check:")
    print(f"   → Duplicate IDs in labels: {dup_labels}")

    if dup_labels > 0:
        # show a few examples
        dup_examples = ids[ids.duplicated()].head(5).tolist()
        print(f"   → Example duplicate IDs: {dup_examples}")

    # Class imbalance ratio
    vc = labels.value_counts(dropna=False)
    print(f"\nClass distribution:")
    print(vc)

    if 0 in vc.index and 1 in vc.index:
        n0, n1 = int(vc.loc[0]), int(vc.loc[1])
        total = n0 + n1
        pos_rate = n1 / total if total > 0 else float("nan")
        neg_rate = n0 / total if total > 0 else float("nan")
        imbalance_ratio = (n0 / n1) if n1 > 0 else float("inf")

        print(f"\nClass imbalance:")
        print(f"   → Positive rate (Label=1): {pos_rate:.3f}")
        print(f"   → Negative rate (Label=0): {neg_rate:.3f}")
        print(f"   → Imbalance ratio (neg/pos): {imbalance_ratio:.2f}")
    else:
        print("Warning: Expected binary labels {0,1} but did not find both classes.")


    # Cross-check labels <-> CT files
    # Remove obvious duplicate copies like "ID_xxx (1).dcm"
    clean_ct_files = []
    duplicate_like_files = []

    for f in dcm_files:
        if " (" in f:  # typical OS duplicate pattern
            duplicate_like_files.append(f)
        else:
            clean_ct_files.append(f)

    if duplicate_like_files:
        print("\nDetected duplicate-like CT files (ignored):")
        for f in duplicate_like_files[:5]:
            print(f"   → {f}")
        print(f"   → Total ignored duplicates: {len(duplicate_like_files)}")

    # Use only clean filenames for matching
    expected_filenames = set(ids.astype(str) + ".dcm")
    ct_filenames = set(clean_ct_files)

    missing_ct_for_labels = sorted(list(expected_filenames - ct_filenames))
    extra_ct_without_labels = sorted(list(ct_filenames - expected_filenames))

    print("\nID ↔ filename cross-check:")
    print(f"   → Missing CT files for labels: {len(missing_ct_for_labels)}")
    print(f"   → Extra CT files without labels (after cleaning duplicates): {len(extra_ct_without_labels)}")

    if missing_ct_for_labels:
        raise ValueError(
            f"Found {len(missing_ct_for_labels)} label IDs with no matching DICOM file."
        )

    if extra_ct_without_labels:
        print("Warning: Found extra CT files without matching labels:")
        print(extra_ct_without_labels[:5])

    print("\nDataset structure + integrity validation PASSED.\n")


def run_features_df_checks(features_df: pd.DataFrame, top_n: int = 20) -> Dict[str, Any]:
    """
    Run compact data-quality checks on features_df and return structured outputs.
    Also prints human-readable summaries.
    """
    out: Dict[str, Any] = {}

    # 1) Missing values in key columns
    key_cols = [c for c in ["img", "label", "test"] if c in features_df.columns]
    if key_cols:
        out["missing_key"] = features_df[key_cols].isna().sum().to_frame("missing_count")
    else:
        out["missing_key"] = pd.DataFrame(columns=["missing_count"])

    out["missing_all_top"] = (
        features_df.isna().sum().sort_values(ascending=False).head(top_n).to_frame("missing_count")
    )

    # 4) Intensity sanity checks
    intensity_cols = [
        c
        for c in [
            "hu_min",
            "hu_max",
            "hu_mean",
            "pixel_array_min",
            "pixel_array_max",
            "pixel_array_mean",
        ]
        if c in features_df.columns
    ]
    out["intensity_cols"] = intensity_cols

    if intensity_cols:
        out["intensity_summary"] = features_df[intensity_cols].describe().T
        out["img_stats"] = None
    else:
        def basic_stats(x):
            arr = np.asarray(x, dtype=np.float32)
            return pd.Series(
                {
                    "img_min": float(np.min(arr)),
                    "img_max": float(np.max(arr)),
                    "img_mean": float(np.mean(arr)),
                }
            )

        if "img" in features_df.columns:
            img_stats = features_df["img"].apply(basic_stats)
        else:
            img_stats = pd.DataFrame(columns=["img_min", "img_max", "img_mean"])
        out["img_stats"] = img_stats
        out["intensity_summary"] = img_stats.describe().T if len(img_stats) else pd.DataFrame()

    # 5) Duplicate IDs / broken rows
    out["duplicate_id_count"] = int(features_df.index.duplicated().sum())
    out["duplicate_ids"] = (
        features_df.index[features_df.index.duplicated()].tolist()
        if out["duplicate_id_count"] > 0
        else []
    )

    has_ok = "ok" in features_df.columns
    has_error = "error" in features_df.columns
    out["has_ok_col"] = has_ok
    out["has_error_col"] = has_error
    out["ok_counts"] = features_df["ok"].value_counts(dropna=False) if has_ok else None

    if has_error:
        err_mask = features_df["error"].notna() & (features_df["error"].astype(str).str.strip() != "")
        out["error_rows_count"] = int(err_mask.sum())
        out["error_examples"] = features_df.loc[err_mask, ["error"]].head(10)
    else:
        out["error_rows_count"] = 0
        out["error_examples"] = pd.DataFrame(columns=["error"])

    # Print compact report
    print("=== Missing Values (key columns) ===")
    if len(out["missing_key"]):
        print(out["missing_key"])
    else:
        print("No key columns found among ['img', 'label', 'test'].")

    print("\n=== Top Missing Counts (all columns) ===")
    print(out["missing_all_top"])

    print("\n=== Intensity Summary ===")
    print(out["intensity_summary"])

    print("\n=== Duplicate/Broken Rows ===")
    print(f"Duplicate IDs: {out['duplicate_id_count']}")
    if out["duplicate_ids"]:
        print("Example duplicate IDs:", out["duplicate_ids"][:10])
    if out["has_ok_col"]:
        print("\n'ok' counts:")
        print(out["ok_counts"])
    if out["has_error_col"]:
        print(f"\nRows with non-empty 'error': {out['error_rows_count']}")
        if out["error_rows_count"] > 0:
            print(out["error_examples"])

    return out


def dicom_to_hu(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Convert a loaded DICOM dataset to HU and apply low-end floor:
    values below/at the 1st percentile are set to -1024.
    """
    pixel_array = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = pixel_array * slope + intercept
    hu = np.where(hu <= np.percentile(hu, 1), -1024.0, hu).astype(np.float32)
    return hu


def load_raw_hu(dcm_path: str) -> np.ndarray:
    """Read DICOM from disk and return HU array using shared conversion logic."""
    ds = pydicom.dcmread(dcm_path)
    return dicom_to_hu(ds)

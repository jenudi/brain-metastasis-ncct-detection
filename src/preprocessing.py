import os
import numpy as np
import pydicom
from multiprocessing import Pool
from typing import Optional
import pandas as pd
from tqdm import tqdm
from src.utils import dicom_to_hu


def crop_to_skull_bbox(
    hu: np.ndarray,
    percentile: float = 98.0,
    pad: int = 0,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Build skull mask using hu > percentile, compute bbox, and return cropped HU image.
    Returns:
      cropped_hu, (upper, lower, left, right)  with lower/right EXCLUSIVE.
    """

    if hu.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {hu.shape}")

    thr = np.percentile(hu, percentile)
    skull_mask = hu > thr

    # rows/cols that contain any skull pixel
    rows_nonzero = np.where(skull_mask.any(axis=1))[0]
    cols_nonzero = np.where(skull_mask.any(axis=0))[0]

    if len(rows_nonzero) == 0 or len(cols_nonzero) == 0:
        # fallback: return full image
        return hu.copy(), (0, hu.shape[0], 0, hu.shape[1])

    upper = rows_nonzero.min()
    lower = rows_nonzero.max() + 1   # make exclusive
    left  = cols_nonzero.min()
    right = cols_nonzero.max() + 1   # make exclusive

    # optional padding
    upper = max(0, upper - pad)
    left = max(0, left - pad)
    lower = min(hu.shape[0], lower + pad)
    right = min(hu.shape[1], right + pad)

    cropped = hu[upper:lower, left:right].astype(np.float32)

    return cropped, (upper, lower, left, right)


def extract_dicom_metadata_features(path: str) -> dict:
    """
    Load DICOM, convert to HU, and extract metadata + raw intensity stats.
    No clipping or windowing applied.
    """

    out = {
        #"filename": os.path.basename(path),
        "ID": path[path.index('ID_'):-4],#.rstrip('.dcm'),
        "path": path,
        "ok": 0,
        "error": None,
    }

    try:
        ds = pydicom.dcmread(path)

        # Convert to HU using shared logic
        hu = dicom_to_hu(ds)
        pixel_array = ds.pixel_array.astype(np.float32)

        # Metadata extraction
        pixel_spacing = getattr(ds, "PixelSpacing", [None, None])
        slice_thickness = getattr(ds, "SliceThickness", None)
        orientation = getattr(ds, "ImageOrientationPatient", None)
        window_center = getattr(ds, "WindowCenter", None)
        window_width = getattr(ds, "WindowWidth", None)

        # Handle multi-value window fields
        if isinstance(window_center, pydicom.multival.MultiValue):
            window_center = float(window_center[0])
        if isinstance(window_width, pydicom.multival.MultiValue):
            window_width = float(window_width[0])

        # Raw HU statistics
        out.update({
            "pixel_spacing_x": float(pixel_spacing[0]) if pixel_spacing[0] is not None else None,
            "pixel_spacing_y": float(pixel_spacing[1]) if pixel_spacing[1] is not None else None,
            "slice_thickness": float(slice_thickness) if slice_thickness is not None else None,
            "orientation": orientation,
            "window_center": float(window_center) if window_center is not None else None,
            "window_width": float(window_width) if window_width is not None else None,

            "shape_0": hu.shape[0],
            "shape_1": hu.shape[1],

            "hu_min": float(np.min(hu)),
            "hu_max": float(np.max(hu)),
            "hu_mean": float(np.mean(hu)),

            "pixel_array_min": float(np.min(pixel_array)),
            "pixel_array_max": float(np.max(pixel_array)),
            "pixel_array_mean": float(np.mean(pixel_array)),

            "img":crop_to_skull_bbox(hu)[0],
            #"uid": getattr(ds, "SOPInstanceUID", None)
        })

        out["ok"] = 1

    except Exception as e:
        out["error"] = str(e)

    return out


def compute_dicom_features_df(
    dir_path: str,
    labels_df: pd.DataFrame,
    processes: Optional[int] = None,
) -> pd.DataFrame:
    """
    Extract DICOM metadata + raw HU stats using filenames
    derived from labels_df index.

    The returned DataFrame will preserve the same index as labels_df.
    """

    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    # Build file paths from CSV index
    ids = labels_df.index.astype(str)

    files = [os.path.join(dir_path, f"{idx}.dcm") for idx in ids]

    # Check missing files
    missing_files = [f for f in files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(
            f"Missing {len(missing_files)} DICOM files. "
            f"Example: {missing_files[:3]}"
        )

    if processes is None:
        processes = max(1, (os.cpu_count() or 2) - 1)

    # Multiprocessing extraction
    out = []

    with Pool(processes=processes) as pool:
        for row in tqdm(
            pool.imap(extract_dicom_metadata_features, files, chunksize=32),
            total=len(files),
            desc=f"Extracting features: {os.path.basename(dir_path)}",
        ):
            out.append(row)

    df = pd.DataFrame(out).set_index('ID').join(labels_df)

    # Align index to labels_df
    #df.index = labels_df.index

    # Add label column directly from CSV
    #if "Label" in labels_df.columns:
     #   df["label"] = labels_df["Label"].values

    # Column ordering
    meta_cols_order = [
        "ID", "path", "Label", "ok", "error",
        "pixel_spacing_x", "pixel_spacing_y", "slice_thickness",
        "orientation", "window_center", "window_width",
        "shape_0", "shape_1",
        "hu_min", "hu_max", "hu_mean","pixel_array_min", "pixel_array_max", "pixel_array_mean","img"
    ]

    meta_cols = [c for c in meta_cols_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in meta_cols]

    df = df[meta_cols + other_cols]
    df.columns = df.columns.str.lower()
    df.index.name = df.index.name.lower() if df.index.name else None
    
    
    return df.drop(columns=['unnamed: 0'], errors="ignore").sort_index()


#print('d')
#features_df = compute_dicom_features_df('data/CTs', 
                                        #pd.read_csv(os.path.join('data/', "labels1.csv"), index_col="ID"),
                                        #processes=1)



def brain_window(hu, center=40, width=80):
    lower = center - width / 2
    upper = center + width / 2

    hu_clipped = np.clip(hu, lower, upper)
    hu_norm = (hu_clipped - lower) / (upper - lower)

    return hu_norm

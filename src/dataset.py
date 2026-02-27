from src.preprocessing import brain_window
from typing import Optional, Sequence
from torch.utils.data import Dataset
import numpy as np
import torch

class BrainCTDataset(Dataset):
    def __init__(
        self,
        values: Sequence[np.ndarray],
        labels: Optional[Sequence[int]] = None,
        ids: Optional[Sequence[str]] = None,
        transform=None,
        apply_brain_window: bool = True,
        window_center: int = 40,
        window_width: int = 80,
    ):
        self.values = list(values)
        self.labels = None if labels is None else list(labels)
        self.ids = None if ids is None else list(ids)
        self.transform = transform

        self.apply_brain_window = apply_brain_window
        self.window_center = window_center
        self.window_width = window_width

        if self.labels is not None and len(self.labels) != len(self.values):
            raise ValueError("labels length must match values length")
        if self.ids is not None and len(self.ids) != len(self.values):
            raise ValueError("ids length must match values length")

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, idx: int):
        x = self.values[idx]

        # ensure numpy float32
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        x = x.astype(np.float32, copy=False)

        # HU -> [0,1] brain window
        if self.apply_brain_window:
            x = brain_window(x, center=self.window_center, width=self.window_width)  # -> [0,1] float

        # transforms: letterbox + aug + 3ch + normalize
        if self.transform is not None:
            x = self.transform(x)
        else:
            x = torch.tensor(x).float().unsqueeze(0)  # (1,H,W)

        # label
        if self.labels is None:
            y = None
        else:
            y = torch.tensor(self.labels[idx]).float()

        sample_id = None if self.ids is None else self.ids[idx]
        return x, y, sample_id
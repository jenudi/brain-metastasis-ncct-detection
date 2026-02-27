import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision


class ToFloatTensorCHW:
    """
    Input:  np.ndarray or torch.Tensor, shape (H, W) or (H, W, 1) or (1, H, W)
    Output: torch.FloatTensor, shape (1, H, W)
    Assumes values already in [0, 1] (after HU windowing).
    """
    def __call__(self, img):
        if not torch.is_tensor(img):
            img = torch.tensor(img)

        img = img.float()

        # (H,W) -> (1,H,W)
        if img.ndim == 2:
            img = img.unsqueeze(0)

        # (H,W,1) -> (1,H,W)
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = img.permute(2, 0, 1)

        # already (1,H,W)
        elif img.ndim == 3 and img.shape[0] == 1:
            pass
        else:
            raise ValueError(f"Unexpected image shape: {tuple(img.shape)}")

        return img


class ResizeLetterbox:
    """
    Resize while keeping aspect ratio, then pad to (out_h, out_w).
    Uses bilinear interpolation.
    """
    def __init__(self, out_h: int, out_w: int, pad_value: float = 0.0):
        self.out_h = int(out_h)
        self.out_w = int(out_w)
        self.pad_value = float(pad_value)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1,H,W)
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(f"Expected (1,H,W), got {tuple(x.shape)}")

        _, h, w = x.shape
        scale = min(self.out_h / h, self.out_w / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        # resize
        x = x.unsqueeze(0)  # (N=1,C=1,H,W)
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        x = x.squeeze(0)    # (1,new_h,new_w)

        # pad to target
        pad_h = self.out_h - new_h
        pad_w = self.out_w - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        x = F.pad(x, (left, right, top, bottom), value=self.pad_value)
        return x


class RepeatTo3Channels:
    """(1,H,W) -> (3,H,W) by channel replication."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(f"Expected (1,H,W), got {tuple(x.shape)}")
        return x.repeat(3, 1, 1)


class Normalize01ToMinus1Plus1:
    """
    Map [0,1] -> [-1,1] using (x - 0.5) / 0.5.
    This is a safe, generic normalization for medical grayscale.
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - 0.5) / 0.5


def build_transforms(
    out_size: int = 256,
    pad_value_01: float = 0.0,
    train: bool = True,
    base_model: bool = True,
):
    """
    Returns a torchvision.transforms.Compose that expects an image already windowed to [0,1].

    Suggested usage:
      - In your Dataset __getitem__:
          hu01 = brain_window(hu)  # -> [0,1]
          x = tfm(hu01)

    Notes:
      - We letterbox to preserve aspect ratio and avoid distortion.
      - We replicate channel to 3 for RadImageNet-pretrained backbones that expect 3ch.
      - We normalize to [-1,1].
    """

    base = [
        ToFloatTensorCHW(),
        ResizeLetterbox(out_size, out_size, pad_value=pad_value_01),
    ]

    if train:
        # light, anatomy-safe augmentations (on tensor)
        aug = [
            # Random small rotations and translations; keep it mild for CT.
            transforms.RandomAffine(
                degrees=10,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
                shear=None,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=pad_value_01,
            ),
            # Optional: horizontal flip is generally acceptable in axial brain slices
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    else:
        aug = []

    tail = [
        RepeatTo3Channels(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                            std=[0.229,0.224,0.225])
    ]

    return transforms.Compose(base + aug + tail)

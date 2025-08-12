# g3_attacks/_common.py
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Union
from art.estimators.classification import PyTorchClassifier
import cv2

def _to_uint8_rgb(arr01: np.ndarray) -> np.ndarray:
    arr = np.clip(arr01 * 255.0, 0, 255).round().astype(np.uint8)
    return arr

def _from_uint8_rgb(img: np.ndarray) -> np.ndarray:
    # (H,W,3) uint8 -> float32 in [0,1]
    return img.astype(np.float32) / 255.0

def _load_rgb_uint8(image: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(image, str):
        bgr = cv2.imread(image, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            raise ValueError("If passing ndarray, please pass uint8 RGB in [0,255].")
        return image
    else:
        raise TypeError("image must be a filepath or an np.ndarray (uint8 RGB).")

def get_art_classifier(
    model: nn.Module,
    n_classes: int,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    loss_fn: nn.Module = None,
    device: Union[str, torch.device] = None,
) -> PyTorchClassifier:
    """
    Wrap your PyTorch classifier for ART attacks.
    Assumes model expects CHW normalized to [0,1].
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    dev = device if device is not None else next(model.parameters()).device
    model.eval().to(dev)

    # A simple preprocessing: just convert [0,1] float to tensor CHW
    # If your model needs normalization (mean/std), add it in a custom forward or here.
    def _forward(x: torch.Tensor) -> torch.Tensor:
        # x: N,H,W,C in [0,1] from ART -> move to CHW
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return model(x)

    class _Wrap(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            return _forward(x)

    wrapped = _Wrap(model)

    classifier = PyTorchClassifier(
        model=wrapped,
        loss=loss_fn,
        input_shape=(input_shape[1], input_shape[2], input_shape[0]),  # ART expects HWC
        nb_classes=n_classes,
        clip_values=(0.0, 1.0),  # we keep images in [0,1]
    )
    return classifier

def _prepare_batch_uint8(rgb_uint8: np.ndarray, size: Tuple[int,int]=(224,224)) -> np.ndarray:
    h,w = size
    im = cv2.resize(rgb_uint8, (w,h), interpolation=cv2.INTER_LINEAR)
    x = _from_uint8_rgb(im)[None, ...]  # (1,H,W,3)
    return x

def _finalize(orig_uint8: np.ndarray, x_adv01: np.ndarray) -> tuple:
    adv_uint8 = _to_uint8_rgb(x_adv01[0])  # (H_adv, W_adv, 3)

    # Resize original to match adv size if needed
    H_adv, W_adv = adv_uint8.shape[:2]
    if orig_uint8.shape[:2] != (H_adv, W_adv):
        import cv2
        orig_match = cv2.resize(orig_uint8, (W_adv, H_adv), interpolation=cv2.INTER_LINEAR)
    else:
        orig_match = orig_uint8

    noise = adv_uint8.astype(np.int16) - orig_match.astype(np.int16)
    return orig_match, noise, adv_uint8



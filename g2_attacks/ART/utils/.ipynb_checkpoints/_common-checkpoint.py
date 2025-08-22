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
    n_classes: int = 2,
    input_shape: Tuple[int, int, int] = (3, 256, 256),  # CHW
    loss_fn: nn.Module = None,
    device: str = "cuda:0",
) -> PyTorchClassifier:
    """
    Wrap your PyTorch classifier for ART attacks.
    Assumes model expects CHW normalized to [0,1].
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    #print(device)
    model.eval().to(device)

    # A simple preprocessing: just convert [0,1] float to tensor CHW
    # If your model needs normalization (mean/std), add it in a custom forward or here.
    def _forward(x: torch.Tensor) -> torch.Tensor:
        return model(x.float())

    class _Wrap(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            return _forward(x)

    wrapped = _Wrap(model)

    device_type = "cpu" if device=="cpu" else "cuda"
    #print(device_type)
    classifier = PyTorchClassifier(
        model=wrapped,
        loss=loss_fn,
        input_shape=input_shape,
        nb_classes=n_classes,
        clip_values=(0.0, 1.0),  # we keep images in [0,1]
        device_type=device_type
    )
    return classifier

def _prepare_batch_uint8(rgb_uint8: np.ndarray, size: Tuple[int,int]=(256,256)) -> np.ndarray:
    h,w = size
    im = cv2.resize(rgb_uint8, (w,h), interpolation=cv2.INTER_LINEAR)
    x = _from_uint8_rgb(im)[None, ...]  # (1,H,W,3)
    return np.transpose(x, (0, 3, 1, 2))




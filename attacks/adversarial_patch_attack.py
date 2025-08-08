# adversarial_patch_attack.py
#
# Unified “Adversarial-Patch” helper — works with any recent
# version of Adversarial-Robustness-Toolbox (ART).
#
# ---------------------------------------------------------------------
# Function
#   adversarial_patch_attack(model, image, …)
#
# Returns
#   original_img_uint8   – H×W×C  uint8
#   noise_int16          – H×W×C  int16   (adv − orig)
#   adversarial_img_uint8– H×W×C  uint8
#
# All ART hyper-parameters are exposed with their default values so
# you can tune them when calling the function from test.py.
# ---------------------------------------------------------------------

from typing import Optional, Tuple, Union

import inspect
import numpy as np
import torch
from PIL import Image
from art.attacks.evasion import AdversarialPatchPyTorch
from art.estimators.classification import PyTorchClassifier


# ---------------------------------------------------------------------
# Helper: convert anything (path / np / tensor / PIL) to a PIL RGB img
# ---------------------------------------------------------------------
def _to_pil(img: Union[str, np.ndarray, torch.Tensor, Image.Image]) -> Image.Image:
    if isinstance(img, str):                       # file path
        return Image.open(img).convert("RGB")
    if isinstance(img, Image.Image):              # PIL image
        return img.convert("RGB")
    if isinstance(img, torch.Tensor):             # C×H×W  torch
        if img.ndim != 3:
            raise ValueError("Torch tensor must have shape C×H×W")
        img = img.detach().cpu().permute(1, 2, 0).numpy()
    if isinstance(img, np.ndarray):               # H×W×C  numpy
        if img.ndim != 3:
            raise ValueError("NumPy array must have shape H×W×C")
        img = img.astype(np.float32)
        if img.max() <= 1.0:
            img *= 255.0
        return Image.fromarray(img.astype(np.uint8))
    raise TypeError("Unsupported image type supplied.")


# ---------------------------------------------------------------------
# Main attack function
# ---------------------------------------------------------------------
def adversarial_patch_attack(
    model: torch.nn.Module,
    image: Union[str, np.ndarray, torch.Tensor, Image.Image],
    target_label: Optional[int] = None,
    # ---- ART AdversarialPatchPyTorch hyper-parameters (defaults) ----
    rotation_max: float = 22.5,
    scale_min: float = 0.10,
    scale_max: float = 1.00,
    learning_rate: float = 5.0,
    max_iter: int = 500,
    batch_size: int = 16,
    patch_shape: Optional[Tuple[int, int, int]] = None,
    random_init: bool = True,
    verbose: bool = False,
    # ----------------------------------------------------------------
    clip_values: Tuple[float, float] = (0.0, 1.0),
    nb_classes: int = 2,
    device: Union[str, torch.device] = "auto",
):
    """
    Build & apply an adversarial patch on a single image.

    Parameters
    ----------
    model         : torch.nn.Module in eval mode
    image         : path | PIL | np.ndarray | torch.Tensor (single image)
    target_label  : int for targeted attack, None for untargeted
    All remaining args map 1-to-1 to ART’s AdversarialPatchPyTorch.
    """

    # -------------------------- device ------------------------------
    dev = (
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    model = model.to(dev).eval()

    # ----------------------- image → numpy --------------------------
    pil_img = _to_pil(image)
    img_np = np.asarray(pil_img).astype(np.float32) / 255.0      # H×W×C
    img_np = np.transpose(img_np, (2, 0, 1))                     # C×H×W
    x = np.expand_dims(img_np, axis=0).astype(np.float32)        # 1×C×H×W

    # ---------------- ART classifier wrapper -----------------------
    loss_fn = torch.nn.CrossEntropyLoss()
    dummy_opt = torch.optim.SGD(model.parameters(), lr=1e-4)     # never used
    classifier = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=dummy_opt,
        input_shape=x.shape[1:],          # (C,H,W)
        nb_classes=nb_classes,
        clip_values=clip_values,
        device_type="cuda" if dev.type == "cuda" else "cpu",
    )


    # ---------- provide default patch_shape if None ----------
    if patch_shape is None:
        # x has shape 1×C×H×W
        _, C, H, W = x.shape
        side = max(1, int(min(H, W) * 0.30))   # 30 % of the shorter side
        patch_shape = (C, side, side)

    # -------------- prepare kwargs accepted by this ART ------------
    init_sig = inspect.signature(AdversarialPatchPyTorch.__init__)
    valid_keys = set(init_sig.parameters.keys())

    atk_kwargs = {
        "rotation_max": rotation_max,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "batch_size": batch_size,
        "patch_shape": patch_shape,
        "patch_type": "circle",
        "verbose": verbose,
        "random_init": random_init,
        # Provide whichever targeting keyword is supported
        "targeted": target_label is not None,      # some versions
        "target": target_label,                    # others
    }
    # keep only keys that this ART version recognises
    atk_kwargs = {k: v for k, v in atk_kwargs.items() if k in valid_keys}

    # ------------------------- build attack -------------------------
    attack = AdversarialPatchPyTorch(estimator=classifier, **atk_kwargs)

    # ---------------------- generate & apply ------------------------
    y_target = None
    if target_label is not None:
        y_target = np.array([target_label])
        
    # patch = attack.generate(x=x, y=y_target)
    # x_adv = attack.apply_patch(x, scale=scale_max, patch_external=patch)
    patch, _ = attack.generate(x=x, y=y_target)
    x_adv = attack.apply_patch(x, scale=scale_max, patch_external=patch)



    # ----------------------- post-processing ------------------------
    orig_uint8 = (np.clip(x[0],     0.0, 1.0) * 255).transpose(1, 2, 0).astype(np.uint8)
    adv_uint8  = (np.clip(x_adv[0], 0.0, 1.0) * 255).transpose(1, 2, 0).astype(np.uint8)
    noise_int16 = adv_uint8.astype(np.int16) - orig_uint8.astype(np.int16)

    return orig_uint8, noise_int16, adv_uint8
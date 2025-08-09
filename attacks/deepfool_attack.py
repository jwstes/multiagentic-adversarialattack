# deepfool_attack.py
#
# Wrapper for ART's DeepFool attack.
# This attack finds the minimal perturbation to cause a misclassification.
#
# See: https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#deepfool
# ---------------------------------------------------------------------

from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from art.attacks.evasion import DeepFool
from art.estimators.classification import PyTorchClassifier

# ---------------------------------------------------------------------
# Helper: convert anything (path / np / tensor / PIL) to a PIL RGB img
# ---------------------------------------------------------------------
def _to_pil(img: Union[str, np.ndarray, torch.Tensor, Image.Image]) -> Image.Image:
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, torch.Tensor):
        if img.ndim != 3:
            raise ValueError("Torch tensor must have shape C×H×W")
        img = img.detach().cpu().permute(1, 2, 0).numpy()
    if isinstance(img, np.ndarray):
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
def deepfool_attack(
    model: torch.nn.Module,
    image: Union[str, np.ndarray, torch.Tensor, Image.Image],
    # ---- ART DeepFool hyper-parameters ----
    max_iter: int = 100,
    epsilon: float = 1e-6,
    nb_grads: int = 10,
    batch_size: int = 1,
    verbose: bool = True,
    # ----------------------------------------------------------------
    n_classes: Optional[int] = None,
    clip_values: Tuple[float, float] = (0.0, 1.0),
    device: Union[str, torch.device] = "auto",
):
    """
    Run ART's DeepFool attack on a single image. This is an untargeted attack.

    Parameters
    ----------
    model       : torch.nn.Module in eval mode.
    image       : path | PIL | np.ndarray | torch.Tensor for a single image.
    max_iter    : The maximum number of iterations.
    epsilon     : Overshoot parameter to push the example over the boundary.
    nb_grads    : Number of top gradients to compute for approximation.
    batch_size  : Batch size for the attack.
    verbose     : Show progress bars.
    n_classes   : The number of classes of the model. Must be provided.
    clip_values : Tuple of min/max values for model inputs.
    device      : 'auto', 'cpu', 'cuda:0', or a torch.device.
    """
    if n_classes is None:
        raise ValueError("`n_classes` must be provided for the DeepFool wrapper.")

    # -------------------------- device ------------------------------
    dev = (
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    model = model.to(dev).eval()

    # ----------------------- image → numpy --------------------------
    pil_img = _to_pil(image)
    img_np = np.asarray(pil_img).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    x = np.expand_dims(img_np, axis=0).astype(np.float32)

    # ---------------- ART classifier wrapper -----------------------
    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=x.shape[1:],
        nb_classes=n_classes,
        clip_values=clip_values,
        device_type="cuda" if dev.type == "cuda" else "cpu",
    )

    # ------------------------- build attack -------------------------
    attack = DeepFool(
        classifier=classifier,
        max_iter=max_iter,
        epsilon=epsilon,
        nb_grads=nb_grads,
        batch_size=batch_size,
        verbose=verbose,
    )

    # ---------------------- generate adversarial example ------------------------
    print("Generating adversarial example with DeepFool Attack...")
    # y=None makes it an untargeted attack, which is the purpose of DeepFool.
    x_adv = attack.generate(x=x, y=None)

    # ----------------------- post-processing ------------------------
    orig_uint8 = (np.clip(x[0], 0.0, 1.0) * 255).transpose(1, 2, 0).astype(np.uint8)
    adv_uint8  = (np.clip(x_adv[0], 0.0, 1.0) * 255).transpose(1, 2, 0).astype(np.uint8)
    noise_int16 = adv_uint8.astype(np.int16) - orig_uint8.astype(np.int16)

    return orig_uint8, noise_int16, adv_uint8
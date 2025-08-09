# composite_attack.py
#
# Wrapper for ART's CompositeAdversarialAttackPyTorch.
# This attack combines semantic and pixel-level perturbations.
# Note: Requires the `kornia` library (`pip install kornia`).
#
# See: https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#composite-adversarial-attack-pytorch
# ---------------------------------------------------------------------

from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from art.attacks.evasion import CompositeAdversarialAttackPyTorch
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
def composite_attack(
    model: torch.nn.Module,
    image: Union[str, np.ndarray, torch.Tensor, Image.Image],
    # ---- ART CompositeAdversarialAttackPyTorch hyper-parameters ----
    enabled_attack: Tuple = (0, 1, 2, 3, 4, 5),
    attack_order: str = 'scheduled',
    max_iter: int = 10,
    max_inner_iter: int = 20,
    verbose: bool = True,
    # ----------------------------------------------------------------
    n_classes: Optional[int] = None,
    clip_values: Tuple[float, float] = (0.0, 1.0),
    device: Union[str, torch.device] = "auto",
):
    """
    Run ART's CompositeAdversarialAttackPyTorch on a single image.
    This attack is UNTARGETED.

    Parameters
    ----------
    model          : torch.nn.Module in eval mode.
    image          : path | PIL | np.ndarray | torch.Tensor for a single image.
    enabled_attack : Tuple of ints selecting attacks (0:Hue, 1:Sat, 2:Rot, 3:Bri, 4:Con, 5:PGD).
    attack_order   : 'scheduled', 'random', or 'fixed'.
    max_iter       : Max iterations for attack order optimization.
    max_inner_iter : Max iterations for each individual attack type.
    verbose        : Show progress bars.
    n_classes      : The number of classes of the model. Must be provided.
    clip_values    : Tuple of min/max values for model inputs.
    device         : 'auto', 'cpu', 'cuda:0', or a torch.device.
    """
    if n_classes is None:
        raise ValueError("`n_classes` must be provided for the Composite Attack wrapper.")

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
    # This attack requires the PyTorchClassifier object directly.
    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=x.shape[1:],
        nb_classes=n_classes,
        clip_values=clip_values,
        device_type="cuda" if dev.type == "cuda" else "cpu",
    )

    # --- Get the original label for the untargeted attack ---
    pred_orig = classifier.predict(x, batch_size=1)
    y = np.argmax(pred_orig, axis=1)
    print(f"Original image predicted as class: {y[0]}")


    # ------------------------- build attack -------------------------
    # Note: This attack is untargeted by design.
    attack = CompositeAdversarialAttackPyTorch(
        classifier=classifier,
        enabled_attack=enabled_attack,
        attack_order=attack_order,
        max_iter=max_iter,
        max_inner_iter=max_inner_iter,
        batch_size=1,
        verbose=verbose,
    )

    # ---------------------- generate adversarial example ------------------------
    print("Generating adversarial example with Composite Attack...")
    x_adv = attack.generate(x=x, y=y) # `y` is the original label

    # ----------------------- post-processing ------------------------
    orig_uint8 = (np.clip(x[0], 0.0, 1.0) * 255).transpose(1, 2, 0).astype(np.uint8)
    adv_uint8  = (np.clip(x_adv[0], 0.0, 1.0) * 255).transpose(1, 2, 0).astype(np.uint8)
    noise_int16 = adv_uint8.astype(np.int16) - orig_uint8.astype(np.int16)

    # Check the final prediction
    pred_adv = np.argmax(classifier.predict(x_adv, batch_size=1), axis=1)
    print(f"Adversarial image predicted as class: {pred_adv[0]}")


    return orig_uint8, noise_int16, adv_uint8
# boundary_attack.py
#
# Wrapper for ART's BoundaryAttack.
# This is a powerful decision-based (black-box) attack.
#
# See: https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#boundary-attack
# ---------------------------------------------------------------------

from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from art.attacks.evasion import BoundaryAttack
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
def boundary_attack(
    model: torch.nn.Module,
    image: Union[str, np.ndarray, torch.Tensor, Image.Image],
    target_label: int,
    # ---- ART BoundaryAttack hyper-parameters ----
    delta: float = 0.01,
    epsilon: float = 0.01,
    step_adapt: float = 0.667,
    max_iter: int = 5000, # Default is 5000, reduced for quicker testing
    num_trial: int = 25,
    sample_size: int = 20,
    init_size: int = 1000,
    verbose: bool = True,
    # ----------------------------------------------------------------
    n_classes: Optional[int] = None,
    clip_values: Tuple[float, float] = (0.0, 1.0),
    device: Union[str, torch.device] = "auto",
):
    """
    Run ART's BoundaryAttack on a single image. This attack is targeted by design.

    Parameters
    ----------
    model        : torch.nn.Module in eval mode.
    image        : path | PIL | np.ndarray | torch.Tensor for a single image.
    target_label : The integer label to target. This is mandatory for BoundaryAttack.
    All other params map directly to ART's BoundaryAttack constructor.
    """
    if n_classes is None:
        raise ValueError("`n_classes` must be provided for the Boundary Attack wrapper.")
    if target_label is None:
         raise ValueError("`target_label` must be provided; BoundaryAttack is targeted.")

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
        loss=torch.nn.CrossEntropyLoss(), # Not used by attack, but required by wrapper
        input_shape=x.shape[1:],
        nb_classes=n_classes,
        clip_values=clip_values,
        device_type="cuda" if dev.type == "cuda" else "cpu",
    )

    # --- Find a starting adversarial example (x_adv_init) ---
    print(f"Searching for a starting image classified as the target ({target_label})...")
    x_adv_init = None
    for i in range(init_size):
        # Generate a random image
        random_img = np.random.rand(*x.shape).astype(np.float32)
        # Check its prediction
        pred = np.argmax(classifier.predict(random_img, batch_size=1), axis=1)
        if pred[0] == target_label:
            print(f"Found a suitable starting point after {i + 1} attempts.")
            x_adv_init = random_img
            break

    if x_adv_init is None:
        raise RuntimeError(
            f"Could not find an initial adversarial example classified as {target_label} "
            f"after {init_size} attempts. Try increasing `init_size` or check the model."
        )

    # ------------------------- build attack -------------------------
    attack = BoundaryAttack(
        estimator=classifier,
        targeted=True, # Boundary attack is designed to be targeted
        delta=delta,
        epsilon=epsilon,
        step_adapt=step_adapt,
        max_iter=max_iter,
        num_trial=num_trial,
        sample_size=sample_size,
        init_size=init_size,
        verbose=verbose,
    )

    # ---------------------- generate adversarial example ------------------------
    y_target = np.array([target_label])
    print("Generating adversarial example with Boundary Attack...")
    x_adv = attack.generate(x=x, y=y_target, x_adv_init=x_adv_init)

    # ----------------------- post-processing ------------------------
    orig_uint8 = (np.clip(x[0], 0.0, 1.0) * 255).transpose(1, 2, 0).astype(np.uint8)
    adv_uint8  = (np.clip(x_adv[0], 0.0, 1.0) * 255).transpose(1, 2, 0).astype(np.uint8)
    noise_int16 = adv_uint8.astype(np.int16) - orig_uint8.astype(np.int16)

    return orig_uint8, noise_int16, adv_uint8
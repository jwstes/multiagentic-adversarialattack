import argparse
import cv2
import numpy as np
from pathlib import Path
import modelLoader

# ---- g3_attacks (your 10 methods) ----
from g3_attacks import (
    pixel_attack,
    threshold_attack,
    jsma_attack,
    sign_opt_attack,
    simple_black_box_attack,
    spatial_transformations_attack,
    square_attack,
    universal_perturbation_attack,
    targeted_universal_perturbation_attack,
    virtual_adversarial_method,
    zoo_attack,
    shadow_attack,
    wasserstein_attack
)

ATTACKS = {
    "pixel": (pixel_attack, dict(max_iter=500)),
    "threshold": (threshold_attack, dict()),
    "jsma": (jsma_attack, dict(theta=1.0, gamma=0.1, targeted=True, target_label=1)),
    "signopt": (sign_opt_attack, dict()),
    "simba": (simple_black_box_attack, dict(epsilon=0.2, max_iter=2000)),
    "spatial": (spatial_transformations_attack, dict(
        max_translation=3.0,  # 3% (will be converted to 0.03)
        num_translations=5,
        max_rotation=0.0,  # start with no rotation; try 5–10 later
        num_rotations=1,
        verbose=False,
    )),
    "square": (square_attack, dict(
        eps=8.0 / 255.0,
        max_iter=100,
        p_init=0.8,
        nb_restarts=1
    )),
    "uap": (universal_perturbation_attack, dict(
        eps=8 / 255, max_iter=1  # base attacker auto=FGM (untargeted)
    )),
    "uap_targeted": (targeted_universal_perturbation_attack, dict(
        target_label=1, eps=8 / 255, max_iter=1  # base attacker auto=FGM targeted
    )),
    "vat": (virtual_adversarial_method, dict(
        eps=10 / 255,  # 8–12/255 is fine
        max_iter=1
    )),
    "zoo": (zoo_attack, dict(
        confidence=0.0,
        max_iter=10,
        binary_search_steps=1,
        attack_size=(96, 96),
        learning_rate=1e-2,
        nb_parallel=32,
        abort_early=True,
    )),
    "shadow": (shadow_attack, dict(
        darkness=0.28,
        blur_sigma=36,
        search_trials=40
    )),
    "wasserstein": (wasserstein_attack, dict(
        targeted=False,
        eps=6 / 255,
        eps_step=0.05 / 255,  # gentler per-step
        max_iter=100,
        regularization=1e7,  # stronger entropy → safer numerics
        p=1,
        kernel_size=5,
        eps_iter=2,  # small number of eps growth steps
        eps_factor=1.02,  # MUST be > 1
        conjugate_sinkhorn_max_iter=1,  # >=1
        projected_sinkhorn_max_iter=200,
        verbose=False,
    ))
}

MODEL2WEIGHTS = {
    "resnet50": ".models/resnet50.pth",
    "densenet121": ".models/densenet121.pth",
    "densenet121_dct": ".models/densenet121_dct.pth",
    "vit_b_16": ".models/vit_b_16.pth",
}

def load_model_any(model_name: str, weights: str | None = None):
    if weights is None:
        if model_name not in MODEL2WEIGHTS:
            raise ValueError(f"No default weights for '{model_name}'. "
                             f"Pass --weights or add to MODEL2WEIGHTS.")
        weights = MODEL2WEIGHTS[model_name]
    if not Path(weights).exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    model = modelLoader.loadModel(model_name, weights)
    return model

def match_shapes(orig, adv):
    """Resize orig to match adv if needed."""
    if orig.shape != adv.shape:
        adv_h, adv_w = adv.shape[:2]
        orig = cv2.resize(orig, (adv_w, adv_h), interpolation=cv2.INTER_LINEAR)
    return orig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to RGB image")
    ap.add_argument("--attack", choices=ATTACKS.keys(), default="square")
    ap.add_argument("--model", choices=list(MODEL2WEIGHTS.keys()), default="resnet50")
    ap.add_argument("--weights", default=None, help="Optional custom path to .pth")
    ap.add_argument("--n_classes", type=int, default=2)
    ap.add_argument("--H", type=int, default=224)
    ap.add_argument("--W", type=int, default=224)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--show", action="store_true", help="Show images in GUI windows")
    args = ap.parse_args()

    # 1) Load model
    model = load_model_any(args.model, args.weights)

    # 2) Pick attack + kwargs
    fn, defaults = ATTACKS[args.attack]
    kwargs = dict(defaults)
    kwargs.setdefault("n_classes", args.n_classes)
    kwargs.setdefault("image_size", (args.H, args.W))

    # 3) Run attack
    orig, noise, adv = fn(model, args.image, **kwargs)

    # Ensure same shapes
    orig = match_shapes(orig, adv)

    # 4) Save
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem + f"_{args.attack}_{args.model}"
    cv2.imwrite(str(outdir / f"{stem}_orig.png"), cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(outdir / f"{stem}_noise.png"), (np.clip(noise + 127, 0, 255)).astype(np.uint8))
    cv2.imwrite(str(outdir / f"{stem}_adv.png"), cv2.cvtColor(adv, cv2.COLOR_RGB2BGR))

    # 5) Optionally show
    if args.show:
        cv2.imshow("original", cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
        cv2.imshow("noise (amplified)", (np.clip(noise + 127, 0, 255)).astype(np.uint8))
        cv2.imshow("adversarial", cv2.cvtColor(adv, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
    # Get model predictions before and after


if __name__ == "__main__":
    main()

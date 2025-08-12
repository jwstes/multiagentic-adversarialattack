# g3_attacks/shadow_attack.py
import cv2
import numpy as np
import torch
from typing import List, Tuple, Literal, Optional
from ._common import _load_rgb_uint8, _finalize, _prepare_batch_uint8, get_art_classifier

ShapeType = Literal["ellipse", "polygon"]

def _soft_mask(h: int, w: int, shapes: List[Tuple[ShapeType, dict]], blur_sigma: float) -> np.ndarray:
    """
    Build a [H,W] float mask in [0,1] from one or more shapes, then Gaussian-blur.
    Each shape is ("ellipse", params) or ("polygon", params).
    Ellipse params: cx, cy, ax, ay, angle_deg
    Polygon params: points (list of (x,y) in pixel coords)
    """
    mask = np.zeros((h, w), np.uint8)

    for kind, p in shapes:
        if kind == "ellipse":
            cx, cy = int(p["cx"]), int(p["cy"])
            ax, ay = int(p["ax"]), int(p["ay"])
            ang  = float(p.get("angle_deg", 0.0))
            cv2.ellipse(mask, (cx, cy), (max(1,ax), max(1,ay)), ang, 0, 360, 255, -1)
        elif kind == "polygon":
            pts = np.array(p["points"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 255)
        else:
            raise ValueError(f"Unknown shape kind: {kind}")

    # Soften edges
    if blur_sigma > 0:
        mask = cv2.GaussianBlur(mask, (0,0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    return (mask.astype(np.float32) / 255.0)  # [0..1]

def _apply_shadow(orig_uint8: np.ndarray, mask01: np.ndarray, darkness: float) -> np.ndarray:
    """
    Apply multiplicative darkening: out = orig * (1 - darkness * mask)
    """
    m = mask01[..., None]  # [H,W,1]
    adv = (orig_uint8.astype(np.float32) * (1.0 - float(darkness) * m)).clip(0,255).astype(np.uint8)
    return adv

def shadow_attack(
    model,
    image,
    n_classes: int = 2,
    image_size: Tuple[int,int] = (224,224),
    # shadow params
    darkness: float = 0.5,              # 0..1 (0=no change, 1=black)
    blur_sigma: float = 21.0,           # Gaussian sigma in px (soft edge)
    shapes: Optional[List[Tuple[ShapeType, dict]]] = None,
    # optional search (black-box) to pick a stronger placement
    search_trials: int = 0,             # 0 = no search; >0 = try N random shadows
    target_label: Optional[int] = None  # set an int for targeted (increase P(target))
):
    """
    Physical-style 'shadow' attack (not in ART). Applies soft dark regions.
    If search_trials > 0, runs a quick black-box search over random ellipses.

    Returns: (orig_uint8, noise_int16, adv_uint8)
    """
    # Load original (keep original size for output/noise)
    orig = _load_rgb_uint8(image)
    H0, W0 = orig.shape[:2]

    # If we just want a single, user-defined shadow:
    if (search_trials is None) or (search_trials <= 0):
        # Default ellipse if none provided
        if not shapes:
            h, w = H0, W0
            shapes = [("ellipse", dict(cx=int(0.5*w), cy=int(0.6*h), ax=int(0.45*w), ay=int(0.25*h), angle_deg=0.0))]
        mask = _soft_mask(H0, W0, shapes, blur_sigma)
        adv = _apply_shadow(orig, mask, darkness)
        return _finalize(orig, adv.astype(np.float32)[None, ...] / 255.0)

    # Else: quick black-box search to pick a strong placement
    # Wrap model as ART classifier for convenience; use model’s working size
    clf = get_art_classifier(model, n_classes, input_shape=(3, image_size[0], image_size[1]))

    def _score(img_uint8: np.ndarray) -> float:
        """
        Untargeted: return max prob (we try to MINIMIZE it).
        Targeted: return target prob (we try to MAXIMIZE it).
        """
        x = _prepare_batch_uint8(img_uint8, size=image_size)  # (1,H,W,3) in [0,1]
        p = clf.predict(x)[0]  # probabilities
        if target_label is None:
            return float(p.max())
        return float(p[int(target_label)])

    best_score = _score(orig)
    best_adv = orig.copy()

    h, w = H0, W0
    rng = np.random.default_rng()

    for _ in range(int(search_trials)):
        # Random ellipse proposal
        ax = max(8, int(rng.uniform(0.15, 0.5) * w))
        ay = max(8, int(rng.uniform(0.10, 0.4) * h))
        cx = int(rng.uniform(ax, w - ax))
        cy = int(rng.uniform(ay, h - ay))
        ang = float(rng.uniform(0, 180))
        cand_shapes = [("ellipse", dict(cx=cx, cy=cy, ax=ax, ay=ay, angle_deg=ang))]

        mask = _soft_mask(h, w, cand_shapes, blur_sigma)
        cand = _apply_shadow(orig, mask, darkness)

        sc = _score(cand)
        improved = (sc > best_score) if (target_label is not None) else (sc < best_score)
        if improved:
            best_score, best_adv = sc, cand

    return _finalize(orig, best_adv.astype(np.float32)[None, ...] / 255.0)


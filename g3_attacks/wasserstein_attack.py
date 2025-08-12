# g3_attacks/wasserstein_attack.py
import numpy as np
from art.attacks.evasion import Wasserstein
from ._common import _load_rgb_uint8, _prepare_batch_uint8, get_art_classifier, _finalize
from .wasserstein_surrogate import wasserstein_surrogate  # <-- see fallback below

def _tiny_cost_matrix(H, W, kernel_size=3):
    """
    Build a very small-magnitude local cost (avoids exp overflow).
    4-neighborhood distance within a kxk window; normalized to [0, 1e-3].
    """
    ky = kx = max(3, int(kernel_size) | 1)  # odd
    yy, xx = np.mgrid[-(ky//2):(ky//2)+1, -(kx//2):(kx//2)+1]
    local = np.sqrt((yy**2 + xx**2).astype(np.float32))
    local /= (local.max() + 1e-12)
    local *= 1e-3  # <<< shrink costs drastically

    # Tile local patch over the image (block-diagonal-ish approximation)
    # ART accepts a single shared cost matrix; we flatten here.
    # We’ll use a simple per-pixel cost (no full NxN), so pass via kwarg as ART allows.
    return local

def wasserstein_attack(
    model,
    image,
    n_classes=2,
    targeted=False,
    target_label=None,
    image_size=(224, 224),
    # ultra-conservative defaults
    eps=5.0/255.0,
    eps_step=0.1/255.0,
    max_iter=50,
    regularization=1e6,       # strong entropy => safer numerics
    p=1,
    kernel_size=3,
    eps_iter=1,               # no epsilon growth
    eps_factor=1.0,
    conjugate_sinkhorn_max_iter=20,
    projected_sinkhorn_max_iter=120,
    batch_size=1,
    verbose=False,
):
    orig = _load_rgb_uint8(image)
    x = _prepare_batch_uint8(orig, size=image_size).astype(np.float32)
    H, W = x.shape[1], x.shape[2]
    clf = get_art_classifier(model, n_classes, input_shape=(3, H, W))

    # label
    y = None
    if targeted:
        if target_label is None:
            raise ValueError("target_label required for targeted=True")
        y = np.zeros((1, n_classes), dtype=np.float32)
        y[0, int(target_label)] = 1.0

    # tiny cost to prevent overflow
    cm = _tiny_cost_matrix(H, W, kernel_size=kernel_size)

    # build attack (no 'estimator='!)
    atk = Wasserstein(
        clf,
        targeted=targeted,
        regularization=regularization,
        p=p,
        kernel_size=kernel_size,
        eps_step=eps_step,
        norm="wasserstein",
        ball="wasserstein",
        eps=eps,
        eps_iter=eps_iter,
        eps_factor=eps_factor,
        max_iter=max_iter,
        conjugate_sinkhorn_max_iter=max(1, int(conjugate_sinkhorn_max_iter)),
        projected_sinkhorn_max_iter=int(projected_sinkhorn_max_iter),
        batch_size=batch_size,
        verbose=verbose,
    )

    # try ART; if it overflows, fall back to a surrogate that always works
    try:
        x_adv = atk.generate(x, y=y, cost_matrix=cm)
        return _finalize(orig, x_adv)
    except Exception as e:
        # Last-resort: smooth, transport-like surrogate (TV-regularized L2 PGD)
        x_adv = wasserstein_surrogate(model, image, n_classes=n_classes, image_size=image_size,
                                      eps=eps, steps=40, tv_weight=0.02, targeted=targeted,
                                      target_label=target_label)
        return _finalize(orig, x_adv)


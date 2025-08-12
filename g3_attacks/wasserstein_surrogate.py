# g3_attacks/wasserstein_surrogate.py
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from ._common import _load_rgb_uint8, _prepare_batch_uint8

def _expand_binary_logits(z: torch.Tensor) -> torch.Tensor:
    if z.ndim == 1: z = z.unsqueeze(1)
    if z.size(1) == 1: z = torch.cat([torch.zeros_like(z), z], dim=1)
    return z

def _tv(delta: torch.Tensor) -> torch.Tensor:
    dx = delta[:, :, 1:, :] - delta[:, :, :-1, :]
    dy = delta[:, :, :, 1:] - delta[:, :, :, :-1]
    return dx.abs().mean() + dy.abs().mean()

def wasserstein_surrogate(
    model, image, n_classes=2, image_size=(224,224),
    eps=6/255, steps=40, tv_weight=0.02, targeted=False, target_label=None,
):
    dev = next(model.parameters()).device
    model = model.eval()
    orig = _load_rgb_uint8(image)
    x01 = _prepare_batch_uint8(orig, size=image_size)
    x = torch.from_numpy(x01).permute(0,3,1,2).to(dev)

    with torch.no_grad():
        z0 = _expand_binary_logits(model(x))
        y0 = z0.argmax(dim=1)
    if targeted:
        if target_label is None: target_label = 1
        y_t = torch.tensor([int(target_label)], device=dev)

    adv = x.clone().detach().requires_grad_(True)
    ce = nn.CrossEntropyLoss()
    alpha = (2.0 * eps) / max(1, steps)

    for _ in range(max(1, steps)):
        z = _expand_binary_logits(model(adv))
        loss_cls = ce(z, y_t) if targeted else -ce(z, y0)
        loss = loss_cls + tv_weight * _tv(adv - x)
        loss.backward()

        g = adv.grad
        g_flat = g.reshape(g.size(0), -1)
        g_norm = torch.linalg.vector_norm(g_flat, ord=2, dim=1).clamp_min(1e-12)
        g_unit = (g_flat / g_norm[:, None]).reshape_as(g)
        adv = (adv + alpha * g_unit).detach().clamp_(0,1).requires_grad_(True)

        delta = adv - x
        d_flat = delta.reshape(delta.size(0), -1)
        d_norm = torch.linalg.vector_norm(d_flat, ord=2, dim=1)
        over = d_norm > eps
        if over.any():
            scale = (eps / d_norm.clamp_min(1e-12))
            scale = torch.where(over, scale, torch.ones_like(scale))
            delta = (d_flat * scale[:, None]).reshape_as(delta)
            adv = (x + delta).detach().clamp_(0,1).requires_grad_(True)

    return adv.detach().cpu().permute(0,2,3,1).numpy()


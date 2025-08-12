# g3_attacks/simple_black_box_attack.py
import numpy as np
import torch
import torch.nn as nn
from ._common import _load_rgb_uint8, _prepare_batch_uint8, _finalize

class _ProbWrap(nn.Module):
    """
    Wrap base model so outputs are probabilities (softmax),
    and ensure binary single-logit models become 2-class.
    """
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x_nhwc):
        # ART-style NHWC in [0,1] -> NCHW
        x = x_nhwc.permute(0, 3, 1, 2)
        z = self.base(x)  # logits or single logit

        # Make shape (N,C)
        if z.ndim == 1:
            z = z.unsqueeze(1)                 # (N,) -> (N,1)
        if z.size(1) == 1:
            # expand to two logits [0, z] so we can softmax to 2-class probs
            z = torch.cat([torch.zeros_like(z), z], dim=1)

        # return probabilities
        return torch.softmax(z, dim=1)

@torch.no_grad()
def _score_prob(model_prob: nn.Module, x: torch.Tensor, target: int | None):
    """
    Return probability score:
      - untargeted: max class probability (we try to DECREASE it)
      - targeted: target class probability (we try to INCREASE it)
    """
    p = model_prob(x)  # (1,C)
    if target is None:
        return p.max(dim=1).values.item()
    return p[0, int(target)].item()

def simple_black_box_attack(
    model,
    image,
    n_classes=2,
    epsilon=0.2,
    max_iter=10000,
    image_size=(224, 224),
    target_label=None,
):
    """
    Simple Black-box Adversarial Attack (SimBA-style), safe indexing and probability-based.
    Keeps the same name/signature; no ART SimBA dependency.
    """
    # 1) Load & prepare image
    orig = _load_rgb_uint8(image)
    x = _prepare_batch_uint8(orig, size=image_size)  # (1,H,W,3) in [0,1]

    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    n_dims = H * W * C
    max_iter = int(min(max_iter, n_dims))  # cap to number of coordinates

    dev = next(model.parameters()).device
    wrap = _ProbWrap(model).to(dev).eval()

    xt = torch.from_numpy(x).to(dev)       # (1,H,W,3) float32 in [0,1]
    flat = xt.view(1, -1)                  # (1, H*W*C)

    # 2) Baseline score
    base_score = _score_prob(wrap, xt, target_label)

    # 3) Random coordinate order
    idx = np.random.permutation(n_dims)[:max_iter]

    # 4) Coordinate-wise perturbation
    for k in idx:
        old = flat[0, k].item()

        # try +epsilon
        flat[0, k] = torch.clamp(torch.tensor(old, device=dev) + epsilon, 0.0, 1.0)
        s_plus = _score_prob(wrap, xt, target_label)

        improved = (s_plus > base_score) if target_label is not None else (s_plus < base_score)
        if not improved:
            # try -epsilon
            flat[0, k] = torch.clamp(torch.tensor(old, device=dev) - epsilon, 0.0, 1.0)
            s_minus = _score_prob(wrap, xt, target_label)
            improved = (s_minus > base_score) if target_label is not None else (s_minus < base_score)
            if improved:
                base_score = s_minus
            else:
                # revert
                flat[0, k] = old
        else:
            base_score = s_plus

    x_adv = xt.detach().cpu().numpy()
    return _finalize(orig, x_adv)


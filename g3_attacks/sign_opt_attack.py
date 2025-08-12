# g3_attacks/sign_opt_attack.py
import numpy as np
import torch
import torch.nn as nn
from art.attacks.evasion import SignOPTAttack
from art.estimators.classification import PyTorchClassifier
from ._common import _load_rgb_uint8, _prepare_batch_uint8, _finalize

class _Wrap3Logits(nn.Module):
    """
    Ensure the estimator exposes 3 logits to bypass Sign-OPT's single-logit ban.
    - If base returns (N,) or (N,1): make (N,2) = [0, z].
    - If base returns (N,2): keep as-is.
    - Then append a very negative dummy logit -> (N,3).
    """
    def __init__(self, base):
        super().__init__()
        self.base = base
        self._printed = False  # print once for sanity

    def forward(self, x_nhwc):
        # ART passes NHWC in [0,1]; convert to NCHW for PyTorch
        x = x_nhwc.permute(0, 3, 1, 2)
        z = self.base(x)  # logits or single logit

        if z.ndim == 1:
            z = z.unsqueeze(1)                    # (N,) -> (N,1)
        if z.size(1) == 1:
            z = torch.cat([torch.zeros_like(z), z], dim=1)  # (N,2) = [0, z]

        if z.size(1) >= 3:
            z3 = z[:, :3]
        else:
            dummy = torch.full_like(z[:, :1], -1e6)
            z3 = torch.cat([z, dummy], dim=1)     # (N,3)

        if not self._printed:
            # One-time sanity print so you can see what ART will get
            print(f"[SignOPT] wrapped logits shape {tuple(z3.shape)} (expect (N,3))")
            self._printed = True
        return z3

def sign_opt_attack(
    model,
    image,
    n_classes=2,                 # ignored; we force 3 in the wrapper classifier
    target_label=None,           # set int for targeted, None for untargeted
    image_size=(224, 224),
    **kwargs                     # extra Sign-OPT kwargs if you want
):
    """
    Decision-based Sign-OPT that works even if the original model is single-logit binary.
    """
    # 1) Load & prep image
    orig = _load_rgb_uint8(image)
    x = _prepare_batch_uint8(orig, size=image_size)  # (1,H,W,3) in [0,1]

    # 2) Our own ART classifier with 3 classes
    wrapped = _Wrap3Logits(model.eval())
    clf = PyTorchClassifier(
        model=wrapped,
        loss=nn.CrossEntropyLoss(),
        input_shape=(image_size[0], image_size[1], 3),  # HWC (ART)
        nb_classes=3,                                   # force 3
        clip_values=(0.0, 1.0),
    )

    # 3) Target handling
    targeted = target_label is not None
    y = None
    if targeted:
        t = 0 if int(target_label) <= 0 else 1   # map your intended {0,1} into our 3-class space
        y = np.zeros((1, 3), dtype=np.float32)
        y[0, t] = 1.0

    # 4) Attack
    attack = SignOPTAttack(clf, targeted=targeted, **kwargs)
    x_adv = attack.generate(x, y=y)

    # 5) Return triplet
    return _finalize(orig, x_adv)


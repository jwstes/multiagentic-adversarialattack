# g3_attacks/virtual_adversarial_method.py
import torch
import torch.nn as nn
from art.attacks.evasion import VirtualAdversarialMethod
from art.estimators.classification import PyTorchClassifier
from ._common import _load_rgb_uint8, _prepare_batch_uint8, _finalize

class _ProbWrap(nn.Module):
    """Make model output probabilities in [0,1] for VAT."""
    def __init__(self, base): super().__init__(); self.base = base
    def forward(self, x_nhwc):
        x = x_nhwc.permute(0, 3, 1, 2)
        z = self.base(x)
        if z.ndim == 1: z = z.unsqueeze(1)
        if z.size(1) == 1:  # single-logit binary -> two logits
            z = torch.cat([torch.zeros_like(z), z], dim=1)
        return torch.softmax(z, dim=1)

def virtual_adversarial_method(
    model,
    image,
    n_classes=2,
    eps=10.0/255.0,      # slightly larger -> converges faster
    max_iter=1,          # VAT typically 1 step
    image_size=(224,224),
):
    orig = _load_rgb_uint8(image)
    x = _prepare_batch_uint8(orig, size=image_size)
    H, W = x.shape[1], x.shape[2]

    dev = next(model.parameters()).device
    prob_model = _ProbWrap(model.eval()).to(dev)

    # ART classifier on correct device
    clf = PyTorchClassifier(
        model=prob_model,
        loss=nn.CrossEntropyLoss(),
        input_shape=(H, W, 3),                 # HWC
        nb_classes=max(2, int(n_classes)),
        clip_values=(0.0, 1.0),
        device_type=("cuda" if dev.type == "cuda" else "cpu"),
    )

    # Newer ART accepts positional or classifier=
    try:
        attack = VirtualAdversarialMethod(clf, eps=eps, max_iter=max_iter)
    except TypeError:
        attack = VirtualAdversarialMethod(classifier=clf, eps=eps, max_iter=max_iter)

    x_adv = attack.generate(x)
    return _finalize(orig, x_adv)




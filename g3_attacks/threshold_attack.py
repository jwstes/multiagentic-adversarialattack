# g3_attacks/threshold_attack.py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from art.attacks.evasion import ThresholdAttack
from ._common import _load_rgb_uint8, _prepare_batch_uint8, get_art_classifier, _finalize

class _UpsampleToModel(nn.Module):
    """
    Expects NCHW from get_art_classifier (it already converts NHWC->NCHW).
    Just upsample spatially to the model's working size, then call base.
    """
    def __init__(self, base, model_hw=(224, 224)):
        super().__init__()
        self.base = base
        self.model_hw = model_hw

    def forward(self, x_nchw):
        # x_nchw: (N, C, H_atk, W_atk)
        x = F.interpolate(x_nchw, size=self.model_hw, mode="bilinear", align_corners=False)
        return self.base(x)  # logits (N, C_classes)

def threshold_attack(
    model,
    image,
    n_classes=2,
    targeted=False,
    target_label=None,
    image_size=(224, 224),      # model working size
    attack_size=(96, 96),       # smaller CMA-ES dimension to avoid OOM
):
    """
    Run ThresholdAttack at smaller attack_size to avoid O(N^2) CMA-ES memory,
    while evaluating the model at image_size.
    """
    orig = _load_rgb_uint8(image)

    # Wrap the model to upsample NCHW to the model resolution.
    wrapped = _UpsampleToModel(model, model_hw=image_size)
    # IMPORTANT: input_shape here is the *attack* HWC; get_art_classifier will
    # convert NHWC->NCHW, then our wrapper upsamples NCHW to image_size.
    clf = get_art_classifier(
        wrapped,
        n_classes,
        input_shape=(3, attack_size[0], attack_size[1]),
    )

    # Prepare the small attack input (NHWC in [0,1] for ART)
    x_small = _prepare_batch_uint8(orig, size=attack_size)

    attack = ThresholdAttack(clf, targeted=targeted)

    y = None
    if targeted and target_label is not None:
        y = np.zeros((1, n_classes), dtype=np.float32)
        y[0, int(target_label)] = 1.0

    x_adv_small = attack.generate(x_small, y=y)
    return _finalize(orig, x_adv_small)


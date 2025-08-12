# g3_attacks/zoo_attack.py
import torch.nn as nn
import torch.nn.functional as F
from art.attacks.evasion import ZooAttack
from ._common import _load_rgb_uint8, _prepare_batch_uint8, get_art_classifier, _finalize

class _UpsampleToModel(nn.Module):
    """Expect NCHW (ART->PyTorch), upsample to model size, run base."""
    def __init__(self, base, model_hw=(224,224)):
        super().__init__(); self.base = base; self.model_hw = model_hw
    def forward(self, x_nchw):
        x = F.interpolate(x_nchw, size=self.model_hw, mode="bilinear", align_corners=False)
        return self.base(x)

def zoo_attack(
    model,
    image,
    n_classes=2,
    confidence=0.0,
    max_iter=40,                 # smaller for speed
    binary_search_steps=1,       # huge speed win
    image_size=(224,224),        # model resolution
    attack_size=(224,224),         # optimization resolution (small = fast)
    learning_rate=1e-2,
    nb_parallel=32,              # try 32/64 if supported by your ART
    abort_early=True,
):
    # 1) small attack input (NHWC)
    orig = _load_rgb_uint8(image)
    x_small = _prepare_batch_uint8(orig, size=attack_size)
    Hs, Ws = attack_size

    # 2) wrap model to upsample NCHW -> image_size before forward
    wrapped = _UpsampleToModel(model, model_hw=image_size)
    clf = get_art_classifier(wrapped, n_classes, input_shape=(3, Hs, Ws))

    # 3) build attack (defensive kwargs; some may be ignored by your ART)
    zoo_kwargs = dict(
        confidence=float(confidence),
        max_iter=int(max_iter),
        binary_search_steps=int(binary_search_steps),
        learning_rate=float(learning_rate),
        abort_early=bool(abort_early),
    )
    # optional extras (if your ART has them)
    try:
        attack = ZooAttack(
            clf,
            **zoo_kwargs,
            nb_parallel=int(nb_parallel),
            use_resize=False,
            use_importance=False,
            initial_const=1e-3,
        )
    except TypeError:
        attack = ZooAttack(clf, **zoo_kwargs)

    x_adv_small = attack.generate(x_small)
    return _finalize(orig, x_adv_small)


from art.attacks.evasion import GRAPHITEWhiteboxPyTorch
from utils._common import _load_rgb_uint8, _prepare_batch_uint8, get_art_classifier
import numpy as np

def GRAPHITEWhitebox_attack(
    # Attack arguments
    model,
    image,
    net_size=(256, 256),
    # Classifier arguments
    batch_size=1,
    # Additional keyword arguments
    **kwargs,
):
    orig = _load_rgb_uint8(image)
    x = _prepare_batch_uint8(orig)
    clf = get_art_classifier(model)

    attack = GRAPHITEWhiteboxPyTorch(
        clf,
        net_size=net_size,
        batch_size=batch_size,
        **kwargs,
    )
    #print(x.shape)
    y = np.array([1], dtype=np.int64)
    x_adv = attack.generate(x, y=y)[0]
    #print(x_adv.shape)
    return orig, x_adv





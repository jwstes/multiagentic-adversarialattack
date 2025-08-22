from art.attacks.evasion import CarliniLInfMethod
from utils._common import _load_rgb_uint8, _prepare_batch_uint8, get_art_classifier
import numpy as np

def carlini_linf_attack(
    # Attack arguments
    model,
    image,
    # Classifier arguments
    batch_size=1,
    # Additional keyword arguments
    **kwargs,
):
    orig = _load_rgb_uint8(image)
    x = _prepare_batch_uint8(orig)
    clf = get_art_classifier(model)

    attack = CarliniLInfMethod(
        clf,
        batch_size=batch_size,
        **kwargs,
    )

    x_adv = attack.generate(x)[0]
    return orig, x_adv


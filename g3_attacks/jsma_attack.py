# g3_attacks/jsma_attack.py
import numpy as np
from art.attacks.evasion import SaliencyMapMethod
from ._common import _load_rgb_uint8, _prepare_batch_uint8, get_art_classifier, _finalize

def jsma_attack(
    model, image, n_classes=2, theta=1.0, gamma=0.1,
    targeted=False, target_label=None, image_size=(224,224)
):
    orig = _load_rgb_uint8(image)
    x = _prepare_batch_uint8(orig, size=image_size)
    clf = get_art_classifier(model, n_classes, input_shape=(3, image_size[0], image_size[1]))

    # New ART API: no 'targeted' kwarg here
    attack = SaliencyMapMethod(classifier=clf, theta=theta, gamma=gamma)

    y = None
    if targeted:
        if target_label is None:
            raise ValueError("JSMA is targeted: provide target_label when targeted=True.")
        y = np.zeros((1, n_classes), dtype=np.float32)
        y[0, target_label] = 1.0

    x_adv = attack.generate(x, y=y)  # pass target one-hot here for targeted JSMA
    return _finalize(orig, x_adv)


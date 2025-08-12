# g3_attacks/pixel_attack.py
from art.attacks.evasion import PixelAttack
from ._common import _load_rgb_uint8, _prepare_batch_uint8, get_art_classifier, _finalize

def pixel_attack(model, image, n_classes=2, max_iter: int = 500, th: int = 10, image_size=(224,224)):
    orig = _load_rgb_uint8(image)
    x = _prepare_batch_uint8(orig, size=image_size)
    clf = get_art_classifier(model, n_classes, input_shape=(3, image_size[0], image_size[1]))
    th = int(th)
    attack = PixelAttack(clf, max_iter=max_iter, th=th)
    x_adv = attack.generate(x)
    return _finalize(orig, x_adv)




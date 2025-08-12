# g3_attacks/universal_perturbation_attack.py
import numpy as np
from art.attacks.evasion import (
    UniversalPerturbation,
    TargetedUniversalPerturbation,
)
from ._common import _load_rgb_uint8, _prepare_batch_uint8, get_art_classifier, _finalize


def universal_perturbation_attack(
    model,
    image,
    n_classes=2,
    eps=8.0 / 255.0,
    max_iter=10,
    image_size=(224, 224),
    attacker: str | None = None,      # e.g., "fgsm", "deepfool", "pgd"
    attacker_params: dict | None = None,
):
    """
    Untargeted Universal Perturbation (ART).
    In this ART version, the base attacker must be a STRING key, not an instance.
    """
    orig = _load_rgb_uint8(image)
    x = _prepare_batch_uint8(orig, size=image_size)
    H, W = x.shape[1], x.shape[2]

    clf = get_art_classifier(model, n_classes, input_shape=(3, H, W))

    # Defaults: use FGSM as the base attack
    if attacker is None:
        attacker = "fgsm"
    if attacker_params is None:
        attacker_params = {"eps": eps, "targeted": False}

    attack = UniversalPerturbation(
        clf,
        eps=eps,
        max_iter=max_iter,
        attacker=attacker,                 # string key (e.g., "fgsm")
        attacker_params=attacker_params,   # dict of params for that base attack
    )

    x_adv = attack.generate(x)
    return _finalize(orig, x_adv)


def targeted_universal_perturbation_attack(
    model,
    image,
    n_classes=2,
    target_label=1,
    eps=8.0 / 255.0,
    max_iter=10,
    image_size=(224, 224),
    attacker: str | None = None,          # e.g., "fgsm", "pgd"
    attacker_params: dict | None = None,
):
    """
    Targeted Universal Perturbation (ART).
    Provide `target_label`. Base attacker must be a STRING key with params dict.
    """
    orig = _load_rgb_uint8(image)
    x = _prepare_batch_uint8(orig, size=image_size)
    H, W = x.shape[1], x.shape[2]

    clf = get_art_classifier(model, n_classes, input_shape=(3, H, W))

    y = np.zeros((1, n_classes), dtype=np.float32)
    y[0, int(target_label)] = 1.0

    if attacker is None:
        attacker = "fgsm"
    if attacker_params is None:
        attacker_params = {"eps": eps, "targeted": True}

    attack = TargetedUniversalPerturbation(
        clf,
        eps=eps,
        max_iter=max_iter,
        attacker=attacker,                 # string key
        attacker_params=attacker_params,   # params for the base attack
    )

    x_adv = attack.generate(x, y=y)
    return _finalize(orig, x_adv)

from art.attacks.evasion import SquareAttack
from ._common import _load_rgb_uint8, _prepare_batch_uint8, get_art_classifier, _finalize
import numpy as np

def square_attack(
    model, image, n_classes=2,
    norm=np.inf, eps=8.0/255.0, max_iter=100,
    p_init=0.8, nb_restarts=1, batch_size=1,
    image_size=(224, 224),
    verbose=True,
):
    orig = _load_rgb_uint8(image)
    x = _prepare_batch_uint8(orig, size=image_size)   # (1,H,W,3)
    H, W = x.shape[1], x.shape[2]

    clf = get_art_classifier(model, n_classes, input_shape=(3, H, W))

    # Backoff: keep cutting p_init until the first tile must fit
    # (ART’s internals can round up; we just try smaller values on failure)
    p_try = float(p_init)
    min_area = 1.0 / (H * W)  # at least one pixel
    for _ in range(6):  # 6 backoff steps are plenty
        try:
            attack = SquareAttack(
                clf,
                norm=norm,
                eps=eps,
                max_iter=max_iter,
                p_init=p_try,
                nb_restarts=nb_restarts,
                batch_size=batch_size,
                verbose=verbose,
            )
            x_adv = attack.generate(x)
            return _finalize(orig, x_adv)
        except ValueError as e:
            msg = str(e)
            # Catch the specific tile-placement crash
            if "high <= 0" in msg or "randint" in msg:
                # halve p_init and retry, but never below one-pixel area
                p_try = max(p_try * 0.5, min_area * 1.1)
                continue
            raise  # different error: bubble up

    # If all retries fail, fall back to a very small p_init
    attack = SquareAttack(
        clf,
        norm=norm,
        eps=eps,
        max_iter=max_iter,
        p_init=min(0.02, max(p_try, min_area * 1.1)),
        nb_restarts=nb_restarts,
        batch_size=batch_size,
        verbose=verbose,
    )
    x_adv = attack.generate(x)
    return _finalize(orig, x_adv)




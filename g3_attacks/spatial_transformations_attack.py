# g3_attacks/spatial_transformations_attack.py
from art.attacks.evasion import SpatialTransformation
from ._common import _load_rgb_uint8, _prepare_batch_uint8, get_art_classifier, _finalize

def _as_fraction(x, cap=0.25):
    """
    ART expects a fraction of image size (0..1). If user gives 3.0 (percent),
    interpret as 3% => 0.03. Clamp to <= cap (default 25%).
    """
    if x is None:
        return 0.0
    xf = float(x)
    if xf > 1.0:       # treat as percent
        xf = xf / 100.0
    return max(0.0, min(xf, cap))

def spatial_transformations_attack(
    model,
    image,
    n_classes=2,
    # If you previously used "3.0" thinking "3%", leave it—this converts to 0.03.
    max_translation=3.0,       # percent or fraction; we sanitize below
    num_translations=5,
    max_rotation=0.0,          # degrees; small values recommended (e.g., 5–10)
    num_rotations=1,
    image_size=(224, 224),
    verbose=False,
):
    orig = _load_rgb_uint8(image)
    x = _prepare_batch_uint8(orig, size=image_size)
    H, W = x.shape[1], x.shape[2]

    # Wrap model for ART
    clf = get_art_classifier(model, n_classes, input_shape=(3, H, W))

    # Sanitize parameters
    max_translation_frac = _as_fraction(max_translation, cap=0.25)  # <= 25% of image
    num_translations = max(1, int(num_translations))
    num_rotations = max(1, int(num_rotations))
    max_rotation = float(max_rotation)  # ART expects degrees here

    # Build attack (pass classifier positionally; no 'estimator=' kwarg)
    attack = SpatialTransformation(
        clf,
        max_translation=max_translation_frac,  # fraction of H/W
        num_translations=num_translations,
        max_rotation=max_rotation,            # degrees
        num_rotations=num_rotations,
        verbose=verbose,
    )

    x_adv = attack.generate(x)  # returns float in [0,1]
    return _finalize(orig, x_adv)

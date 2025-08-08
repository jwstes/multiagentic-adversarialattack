# modelLoader.py

import warnings
from pathlib import Path
from typing import Optional, Union, Dict, Any

import torch
import torch.nn as nn
from torchvision import models as tv_models

__all__ = ["loadModel"]

# Default number of classes (kept consistent with the reference script)
DEFAULT_NUM_CLASSES = 2


def _create_resnet18_dct(num_classes: int) -> nn.Module:
    model = tv_models.resnet18(weights=None)
    # Single-channel input for DCT images
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Replace final classifier head
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def _create_densenet121_dct(num_classes: int) -> nn.Module:
    model = tv_models.densenet121(weights=None)
    # Single-channel input for DCT images
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Replace final classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier.in_features, num_classes),
    )
    return model


def _build_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet50":
        model = tv_models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "densenet121":
        model = tv_models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif arch == "vit_b_16":
        # torchvision's ViT
        model = tv_models.vit_b_16(weights=None)
        # Replace classifier head
        # vit_b_16 has 'heads' Sequential with 'head' Linear
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    elif arch == "resnet18_dct":
        model = _create_resnet18_dct(num_classes)
    elif arch == "densenet121_dct":
        model = _create_densenet121_dct(num_classes)
    else:
        raise ValueError(
            f"Unsupported model '{arch}'. "
            "Supported: resnet50, densenet121, vit_b_16, resnet18_dct, densenet121_dct"
        )
    return model


def _get_device(device: Union[str, torch.device, None]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Allow strings like "cpu", "cuda:0"
    return torch.device(device)


def _extract_state_dict(loaded: Any) -> Dict[str, torch.Tensor]:
    # Handle common checkpoint structures
    if isinstance(loaded, dict):
        if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            sd = loaded["state_dict"]
        elif "model" in loaded and isinstance(loaded["model"], dict):
            sd = loaded["model"]
        else:
            # Assume it's already a state_dict
            sd = loaded
    else:
        raise TypeError("Loaded checkpoint is not a dict or a state_dict.")
    # Strip 'module.' if present (DataParallel)
    new_sd = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        new_sd[nk] = v
    return new_sd


def _smart_load_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
) -> None:
    if strict:
        model.load_state_dict(state_dict, strict=True)
        return

    # When strict=False, filter out keys with shape mismatch to avoid errors
    model_sd = model.state_dict()
    filtered = {}
    dropped = []

    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            dropped.append(k)

    if dropped:
        warnings.warn(
            f"Strict=False: dropped {len(dropped)} keys due to shape mismatch "
            f"(e.g., final classifier). Keys dropped (showing up to 5): {dropped[:5]} ..."
        )
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        warnings.warn(f"Missing keys in loaded state_dict: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        warnings.warn(f"Unexpected keys in loaded state_dict: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")


def loadModel(
    name: str,
    weight_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = "auto",
    num_classes: int = DEFAULT_NUM_CLASSES,
    strict: bool = True,
) -> nn.Module:
    """
    Build and load a classification model.

    Args:
        name: Model architecture name. One of:
              'resnet50', 'densenet121', 'vit_b_16', 'resnet18_dct', 'densenet121_dct'.
        weight_path: Optional path to a .pth/.pt checkpoint. If None, returns randomly initialized model.
        device: 'auto', 'cpu', 'cuda:0', or a torch.device. Default 'auto'.
        num_classes: Number of output classes. Default 2.
        strict: If False, mismatched keys (e.g., classifier head) are ignored during loading.

    Returns:
        torch.nn.Module in eval mode, moved to the requested device.
    """
    dev = _get_device(device)
    model = _build_model(name, num_classes=num_classes)

    if weight_path is not None:
        weight_path = Path(weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        print(f"[MODEL] Loading '{name}' weights from {weight_path} ...")
        ckpt = torch.load(weight_path, map_location=dev)
        state_dict = _extract_state_dict(ckpt)
        _smart_load_state_dict(model, state_dict, strict=strict)
    else:
        print(f"[MODEL] No weights provided for '{name}'. Using randomly initialized parameters.")

    model.to(dev).eval()
    print(f"[MODEL] '{name}' ready on {dev}")
    return model
from __future__ import annotations

from .base import OpenImageModel
from .flux import FluxDevModel, FluxSchnellModel
from .qwen_image import QwenImageModel

MODEL_REGISTRY: dict[str, type[OpenImageModel]] = {
    "flux.1-dev": FluxDevModel,
    "flux.1-schnell": FluxSchnellModel,
    "qwen-image": QwenImageModel,
}


def build_model(name: str, checkpoint: str | None = None) -> OpenImageModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](checkpoint=checkpoint)

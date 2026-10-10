from __future__ import annotations

from .base import ClosedImageModel
from .gemini import Gemini31FlashImage, Gemini3ProImage, GeminiFlashImage
from .gpt_image import GPTImage1, GPTImage2

MODEL_REGISTRY: dict[str, type[ClosedImageModel]] = {
    "gemini-2.0-flash": GeminiFlashImage,
    "gemini-3.1-flash-image": Gemini31FlashImage,
    "gemini-3-pro-image": Gemini3ProImage,
    "gpt-image-1": GPTImage1,
    "gpt-image-2": GPTImage2,
}


def build_model(name: str) -> ClosedImageModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()

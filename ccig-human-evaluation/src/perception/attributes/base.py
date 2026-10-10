from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class AttributeClassifier(ABC):
    """Predicts one property (e.g. color, shape, material, size) over a fixed label
    set, given a cropped+background-neutralized object image (see perception/crop.py)."""

    name: str

    def __init__(self, property_name: str, labels: list[str], device: str | None = None) -> None:
        self.property_name = property_name
        self.labels = labels
        # Same single GPU/CPU resolution point as ObjectDetector.
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def classify(self, crop: Image.Image, context: str | None = None) -> tuple[str, float]:
        """Returns (predicted_label, confidence).

        context: an already-known attribute of the same object (e.g. its predicted
        shape) to fold into the classification prompt -- used for color, where "a
        photo of a red cube" is a stronger zero-shot query than "a photo of a red
        object". None for attributes classified independently (shape, material, size).
        """
        raise NotImplementedError

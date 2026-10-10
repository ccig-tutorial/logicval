from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float  # pixel coordinates, absolute (not normalized)
    label: str  # the class_prompts text this box matched, e.g. "cube" or "bicycle"
    score: float


class ObjectDetector(ABC):
    """Open-vocabulary zero-shot detector: text class prompts -> bounding boxes.

    No CLEVR/COCO-specific training is assumed -- the same detector instance is
    reused across domains, just given different class_prompts per call.
    """

    name: str

    def __init__(self, device: str | None = None) -> None:
        # The one place GPU-vs-CPU is decided -- subclasses only ever use self.device,
        # never branch on it themselves. Works on CPU too (just slower), so there is
        # no separate CPU code path to maintain.
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def detect(self, image: Image.Image, class_prompts: list[str]) -> list[BBox]:
        raise NotImplementedError

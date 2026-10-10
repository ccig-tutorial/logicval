from __future__ import annotations

from .base import ObjectDetector
from .grounding_dino import GroundingDinoDetector
from .owlv2 import Owlv2Detector

DETECTOR_REGISTRY: dict[str, type[ObjectDetector]] = {
    "grounding-dino": GroundingDinoDetector,
    "owlv2": Owlv2Detector,
}


def build_detector(name: str, device: str | None = None) -> ObjectDetector:
    if name not in DETECTOR_REGISTRY:
        raise ValueError(f"Unknown detector '{name}'. Available: {sorted(DETECTOR_REGISTRY)}")
    return DETECTOR_REGISTRY[name](device=device)

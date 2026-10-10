from __future__ import annotations

from .base import AttributeClassifier
from .clip_zero_shot import ClipZeroShotAttribute

# Deliberately one entry for v1 -- the registry exists so a future non-CLIP
# classifier (e.g. a small trained head) can be added without touching call sites.
ATTRIBUTE_REGISTRY: dict[str, type[AttributeClassifier]] = {
    "clip-zero-shot": ClipZeroShotAttribute,
}


def build_attribute_classifier(
    name: str, property_name: str, labels: list[str], device: str | None = None
) -> AttributeClassifier:
    if name not in ATTRIBUTE_REGISTRY:
        raise ValueError(f"Unknown attribute classifier '{name}'. Available: {sorted(ATTRIBUTE_REGISTRY)}")
    return ATTRIBUTE_REGISTRY[name](property_name=property_name, labels=labels, device=device)

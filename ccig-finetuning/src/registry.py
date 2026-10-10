from __future__ import annotations

from src.models.base import LoraTrainer
from src.models.flux import FluxDevTrainer, FluxSchnellTrainer
from src.models.qwen_image import QwenImageTrainer

TRAINER_REGISTRY: dict[str, type[LoraTrainer]] = {
    "flux.1-dev": FluxDevTrainer,
    "flux.1-schnell": FluxSchnellTrainer,
    "qwen-image": QwenImageTrainer,
}


def build_trainer(name: str) -> LoraTrainer:
    if name not in TRAINER_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(TRAINER_REGISTRY)}")
    return TRAINER_REGISTRY[name]()

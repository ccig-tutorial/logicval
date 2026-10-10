from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.common.types import TrainConfig


class LoraTrainer(ABC):
    """LoRA finetuning for one text-to-image model."""

    name: str
    hf_repo: str

    @abstractmethod
    def train(self, config: TrainConfig) -> Path:
        """Run LoRA finetuning and return the checkpoint directory written.

        The returned directory follows the convention ccig-image-generation's
        `--checkpoint` flag expects (a diffusers LoRA-weights directory).
        """
        raise NotImplementedError

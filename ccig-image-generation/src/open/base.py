from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class OpenImageModel(ABC):
    """A locally-run, open-weight text-to-image model."""

    name: str
    hf_repo: str

    def __init__(self, checkpoint: str | None = None) -> None:
        """checkpoint: optional path to a LoRA finetuned checkpoint directory
        (as produced by ccig-finetuning). If None, base weights from hf_repo are used.
        """
        self.checkpoint = checkpoint

    @abstractmethod
    def generate(self, prompt: str) -> Image.Image:
        raise NotImplementedError

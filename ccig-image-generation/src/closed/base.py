from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class ClosedImageModel(ABC):
    """A closed-source, API-based text-to-image model."""

    name: str

    @abstractmethod
    def generate(self, prompt: str) -> Image.Image:
        raise NotImplementedError

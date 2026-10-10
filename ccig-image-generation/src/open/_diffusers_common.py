from __future__ import annotations

from typing import Any, ClassVar

import torch
from PIL import Image, ImageOps

from .base import OpenImageModel

# Delivered image size. ccig-finetuning's LoRA checkpoints are trained on 1024x688
# (CLEVR's native 3:2 ratio, scaled to fit a 1024 long side, undistorted) -- see
# TrainConfig.resolution_width/resolution_height. Generation below runs the pipeline
# at that same shape (matching what the model actually learned), then pads the result
# up to a delivered DELIVERED_SIZE x DELIVERED_SIZE square, since downstream tooling
# expects square 1024x1024 files. The pad only happens to the already-generated image;
# the model itself never sees or produces blank padding.
DELIVERED_SIZE = 1024
GENERATION_WIDTH = 1024
GENERATION_HEIGHT = 688


class DiffusersImageModel(OpenImageModel):
    """Shared implementation for models built on a `diffusers` pipeline.

    Subclasses set `pipeline_cls` + `hf_repo`, and override `_call_kwargs` for
    model-specific sampling parameters (steps, guidance scale, ...). This is the
    single place that knows how to load base weights, optionally apply a LoRA
    checkpoint, and run the pipeline -- every diffusers-based model below reuses it
    instead of re-implementing `generate()`.
    """

    pipeline_cls: ClassVar[Any]
    torch_dtype: ClassVar[Any] = torch.bfloat16
    _call_kwargs: ClassVar[dict[str, Any]] = {}

    def __init__(self, checkpoint: str | None = None) -> None:
        super().__init__(checkpoint)
        self._pipe = self.pipeline_cls.from_pretrained(self.hf_repo, torch_dtype=self.torch_dtype)
        self._pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        if checkpoint is not None:
            # Convention: ccig-finetuning saves the LoRA-wrapped transformer via PEFT's
            # `save_pretrained`; load it back the same way (PeftModel.from_pretrained),
            # not diffusers' `pipe.load_lora_weights` (a different state-dict format).
            from peft import PeftModel

            self._pipe.transformer = PeftModel.from_pretrained(self._pipe.transformer, checkpoint)

    def generate(self, prompt: str) -> Image.Image:
        # height/width match ccig-finetuning's training resolution (GENERATION_WIDTH x
        # GENERATION_HEIGHT) so the LoRA generates in the geometry it actually learned,
        # instead of falling back to this pipeline's own (square) default.
        image = self._pipe(
            prompt=prompt, height=GENERATION_HEIGHT, width=GENERATION_WIDTH, **self._call_kwargs
        ).images[0]
        # Pad the undistorted 1024x688 result up to a delivered 1024x1024 square --
        # downstream tooling expects square files, but the model itself never trains on
        # or generates blank padding (see the module docstring above).
        return ImageOps.pad(
            image, (DELIVERED_SIZE, DELIVERED_SIZE), color=(255, 255, 255), centering=(0.5, 0.5)
        )

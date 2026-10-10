from __future__ import annotations

from diffusers import QwenImagePipeline

from ._diffusers_common import DiffusersImageModel


class QwenImageModel(DiffusersImageModel):
    name = "qwen-image"
    hf_repo = "Qwen/Qwen-Image"
    pipeline_cls = QwenImagePipeline
    _call_kwargs = {"num_inference_steps": 50, "true_cfg_scale": 4.0}

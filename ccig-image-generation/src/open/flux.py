from __future__ import annotations

from diffusers import FluxPipeline

from ._diffusers_common import DiffusersImageModel


class FluxBase(DiffusersImageModel):
    pipeline_cls = FluxPipeline


class FluxDevModel(FluxBase):
    name = "flux.1-dev"
    hf_repo = "black-forest-labs/FLUX.1-dev"
    _call_kwargs = {"num_inference_steps": 28, "guidance_scale": 3.5}


class FluxSchnellModel(FluxBase):
    name = "flux.1-schnell"
    hf_repo = "black-forest-labs/FLUX.1-schnell"
    # schnell is a distilled, few-step model with no classifier-free guidance.
    _call_kwargs = {"num_inference_steps": 4, "guidance_scale": 0.0}

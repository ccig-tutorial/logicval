from __future__ import annotations

from typing import Any

import torch
from diffusers import QwenImagePipeline
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from ._diffusers_common import DiffusersLoraTrainer


class QwenImageTrainer(DiffusersLoraTrainer):
    # Qwen-Image is guidance-distilled; its pipeline raises if guidance_scale is None
    # and transformer.config.guidance_embeds is True. See QwenImagePipeline.__call__'s
    # "handle guidance" block for the reference behavior this mirrors. The exact tuned
    # value for LoRA training isn't verified end-to-end here -- adjust if needed.
    guidance_scale: float = 1.0

    # SD3-paper flow-matching timestep density + loss-weighting scheme, matching the
    # FlyMyAI Qwen-Image LoRA trainer's train.py verbatim (confirmed by reading its actual
    # compute_density_for_timestep_sampling/compute_loss_weighting_for_sd3 calls, not
    # assumed): https://github.com/FlyMyAI/flymyai-lora-trainer. "none" means
    # compute_density_for_timestep_sampling samples timesteps uniformly (not logit-normal)
    # and compute_loss_weighting_for_sd3 returns all-ones (no weighting, equivalent to
    # plain MSE)
    timestep_weighting_scheme: str = "none"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.29

    name = "qwen-image"
    hf_repo = "Qwen/Qwen-Image"
    pipeline_cls = QwenImagePipeline
    frozen_module_names = ["vae", "text_encoder"]

    def _training_step(self, pipe: Any, transformer: Any, batch: dict) -> torch.Tensor:
        device = next(transformer.parameters()).device
        dtype = self.torch_dtype
        pixel_values = batch["pixel_values"].to(device, dtype=dtype)
        texts = batch["prompt"]

        # VAE + text encoder live on CPU by default (see frozen_module_names) and are
        # moved to GPU only for this block, then back -- keeps the resident, trainable
        # transformer's memory budget from having to share the GPU with the ~20B+ params
        # of frozen Qwen-Image weights at the same time.
        def _encode():
            with torch.no_grad():
                # AutoencoderKLQwenImage is a *video* VAE (see its temperal_downsample
                # config / QwenImageEncoder3d) reused for stills as 1-frame videos -- its
                # encoder hard-requires a 5D (B, C, T, H, W) input (confirmed: it slices
                # x[:, :, :1, :, :] internally, which raises on a 4D tensor). Insert a
                # size-1 temporal axis before encoding, then drop it right after --
                # everything below this line already assumes plain (B, C, H, W), and a
                # size-1 axis never reorders memory, so squeezing it back out here is
                # exactly equivalent to the reference's unsqueeze+permute dance.
                latents = pipe.vae.encode(pixel_values.unsqueeze(2)).latent_dist.sample()
                latents = latents.squeeze(2)
                # AutoencoderKLQwenImage has no scalar `scaling_factor` -- it normalizes
                # per-channel via latents_mean/latents_std instead.
                latents_mean = (
                    torch.tensor(pipe.vae.config.latents_mean, device=device, dtype=dtype)
                    .view(1, -1, 1, 1)
                )
                latents_std = 1.0 / torch.tensor(
                    pipe.vae.config.latents_std, device=device, dtype=dtype
                ).view(1, -1, 1, 1)
                latents = (latents - latents_mean) * latents_std
                bsz, num_channels, lat_h, lat_w = latents.shape

                # Use QwenImagePipeline's own `_pack_latents` (same 2x2 patch packing as
                # FLUX) instead of reimplementing it.
                packed_latents = pipe._pack_latents(latents, bsz, num_channels, lat_h, lat_w)
                noise = torch.randn_like(latents)
                packed_noise = pipe._pack_latents(noise, bsz, num_channels, lat_h, lat_w)
                # img_shapes / txt_seq_lens as built in QwenImagePipeline.__call__: one
                # (frames, h//2, w//2) tuple per batch item, and per-sample real token
                # counts (post-padding-mask) for the text sequence.
                img_shapes = [[(1, lat_h // 2, lat_w // 2)]] * bsz

                # Sample timesteps via compute_density_for_timestep_sampling (uniform, given
                # weighting_scheme="none"), then look sigma up from the scheduler's own
                # schedule (pipe.scheduler.sigmas[indices]) instead of a naive uniform
                # torch.randint + timesteps/num_train_timesteps approximation -- the sigma
                # value comes from the scheduler's actual schedule this way, so it stays
                # correct even if that schedule isn't perfectly linear.
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=self.timestep_weighting_scheme,
                    batch_size=bsz,
                    logit_mean=self.logit_mean,
                    logit_std=self.logit_std,
                    mode_scale=self.mode_scale,
                    device=device,
                )
                indices = (u * pipe.scheduler.config.num_train_timesteps).long()
                # pipe.scheduler.timesteps lives on CPU by default (schedulers aren't
                # moved to GPU by from_pretrained) -- index with a CPU copy of `indices`
                # (itself built on `device`, since compute_density_for_timestep_sampling
                # was called with device=device), then move the *result* to `device`.
                # Indexing directly with a CUDA index tensor into a CPU tensor raises.
                timesteps = pipe.scheduler.timesteps[indices.cpu()].to(device=device)
                sigma = pipe.scheduler.sigmas.to(device=device, dtype=dtype)[indices].view(-1, 1, 1)
                noisy_packed = sigma * packed_noise + (1.0 - sigma) * packed_latents


                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=self.timestep_weighting_scheme, sigmas=sigma
                )

                # pipe.encode_prompt returns prompt_embeds_mask=None when every prompt in
                # the batch needed no padding (all-ones mask) -- confirmed by reading this
                # installed diffusers version's encode_prompt source, which explicitly
                # collapses an all-ones mask to None. That's the normal case for
                # batch_size=1 (nothing else in the batch to pad against). None is a
                # valid value here, not a bug to route around: the reference pipeline's
                # own __call__ passes prompt_embeds_mask straight through to
                # encoder_hidden_states_mask unchanged, None included. This installed
                # version's QwenImageTransformer2DModel.forward also has no txt_seq_lens
                # parameter at all anymore (confirmed via inspect.signature) -- it was
                # part of an older diffusers release this file was first written against.
                prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
                    prompt=texts, device=device, num_images_per_prompt=1, max_sequence_length=1024
                )

                guidance = None
                if transformer.config.guidance_embeds:
                    guidance = torch.full((bsz,), self.guidance_scale, device=device, dtype=torch.float32)

            return (
                noisy_packed, packed_noise, packed_latents, img_shapes, timesteps,
                prompt_embeds, prompt_embeds_mask, guidance, weighting,
            )

        (
            noisy_packed, packed_noise, packed_latents, img_shapes, timesteps,
            prompt_embeds, prompt_embeds_mask, guidance, weighting,
        ) = self._run_frozen_on_cuda(pipe, _encode)

        model_pred = transformer(
            hidden_states=noisy_packed,
            timestep=timesteps.float() / 1000,
            guidance=guidance,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            return_dict=True,
        ).sample
        flow_target = packed_noise - packed_latents
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - flow_target.float()) ** 2).reshape(
                model_pred.shape[0], -1
            ),
            dim=1,
        )
        return loss.mean()

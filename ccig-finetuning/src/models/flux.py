from __future__ import annotations

from typing import Any

import torch
from diffusers import FluxPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from ._diffusers_common import DiffusersLoraTrainer


class FluxBaseTrainer(DiffusersLoraTrainer):
    pipeline_cls = FluxPipeline
    frozen_module_names = ["vae", "text_encoder", "text_encoder_2"]
    # FLUX.1-dev is guidance-distilled (transformer.config.guidance_embeds == True) and
    # requires a `guidance` tensor at every forward pass, including training; schnell is
    # not guidance-distilled and passes guidance=None (guidance_embeds is False for it,
    # so the check below resolves to None automatically -- this constant only actually
    # matters for flux.1-dev). 1.0 matches the FlyMyAI FLUX.1-dev LoRA trainer's fixed
    # training-time guidance value (https://github.com/FlyMyAI/flymyai-lora-trainer,
    # train_flux_lora.py) -- deliberately NOT FluxPipeline's own inference-time default
    # of 3.5, which is a different, unrelated choice for a different point in the
    # pipeline (sampling quality vs. what the guidance-distilled transformer was told
    # during training).
    guidance_scale: float = 1.0

    # SD3-paper flow-matching timestep density + loss-weighting scheme, matching the
    # FlyMyAI FLUX.1-dev LoRA trainer's train_flux_lora.py verbatim (confirmed by reading
    # its actual compute_density_for_timestep_sampling/compute_loss_weighting_for_sd3
    # calls). "none" means timesteps are sampled uniformly and loss weighting is a no-op
    # (all-ones) -- the real value this still adds over the previous torch.randint code
    # is sourcing sigma from the scheduler's own schedule (pipe.scheduler.sigmas[indices])
    # instead of a linear approximation (timesteps/num_train_timesteps), which stays
    # correct even if FLUX's schedule isn't perfectly linear (it uses a "shift" parameter
    # that can make it non-linear).
    timestep_weighting_scheme: str = "none"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.29

    def _training_step(self, pipe: Any, transformer: Any, batch: dict) -> torch.Tensor:
        device = next(transformer.parameters()).device
        dtype = self.torch_dtype
        pixel_values = batch["pixel_values"].to(device, dtype=dtype)
        texts = batch["prompt"]

        # VAE + text encoders live on CPU by default (see frozen_module_names) and are
        # moved to GPU only for this block, then back -- keeps the resident, trainable
        # transformer's memory budget from having to share the GPU with the ~24GB of
        # frozen FLUX weights at the same time.
        def _encode():
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                # FLUX's VAE normalizes via (latents - shift_factor) * scaling_factor --
                # confirmed via diffusers' own pipeline_flux_img2img.py encode-side
                # normalization (the reference for encoding a real image into this VAE's
                # latent space) and the FlyMyAI reference script. The previous version of
                # this file only applied `* scaling_factor`, silently dropping the
                # `- shift_factor` centering step.
                latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                bsz, num_channels, lat_h, lat_w = latents.shape

                # Use FluxPipeline's own packing helpers (`_pack_latents` converts
                # (B,C,H,W) -> (B, H//2*W//2, C*4) patches; `_prepare_latent_image_ids`
                # builds the matching RoPE position ids) instead of a hand-rolled
                # reimplementation, so this stays correct if diffusers changes the format.
                packed_latents = pipe._pack_latents(latents, bsz, num_channels, lat_h, lat_w)
                noise = torch.randn_like(latents)
                packed_noise = pipe._pack_latents(noise, bsz, num_channels, lat_h, lat_w)
                img_ids = pipe._prepare_latent_image_ids(bsz, lat_h // 2, lat_w // 2, device, dtype)

                # Sample timesteps via compute_density_for_timestep_sampling (uniform,
                # given weighting_scheme="none"), then look sigma up from the scheduler's
                # own schedule (pipe.scheduler.sigmas[indices]) instead of a naive uniform
                # torch.randint + timesteps/num_train_timesteps approximation -- sigma
                # comes from the scheduler's actual schedule this way, so it stays correct
                # even if that schedule isn't perfectly linear.
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

                # weighting_scheme="none" makes this return all-ones (no-op vs. plain MSE);
                # kept as an explicit call rather than dropped, so switching schemes later
                # is a one-line change instead of restructuring the loss.
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=self.timestep_weighting_scheme, sigmas=sigma
                )

                # encode_prompt(prompt, prompt_2=None, ...) -> (prompt_embeds, pooled_prompt_embeds,
                # text_ids); prompt_2 defaults to prompt when omitted. text_ids is already
                # unbatched (seq_len, 3), matching what the transformer expects.
                prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                    prompt=texts, device=device, num_images_per_prompt=1, max_sequence_length=512
                )

                guidance = None
                if transformer.config.guidance_embeds:
                    guidance = torch.full((bsz,), self.guidance_scale, device=device, dtype=torch.float32)

            return (
                noisy_packed, packed_noise, packed_latents, img_ids, timesteps,
                prompt_embeds, pooled_prompt_embeds, text_ids, guidance, weighting,
            )

        (
            noisy_packed, packed_noise, packed_latents, img_ids, timesteps,
            prompt_embeds, pooled_prompt_embeds, text_ids, guidance, weighting,
        ) = self._run_frozen_on_cuda(pipe, _encode)

        # timestep is normalized to [0, 1] (timestep/1000), matching FluxPipeline's own
        # denoising-loop block -- passing the raw 0-999 integer here (as this file
        # previously did) doesn't match what the transformer was calibrated to expect.
        pred_packed = transformer(
            hidden_states=noisy_packed,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            timestep=timesteps.float() / 1000,
            img_ids=img_ids,
            txt_ids=text_ids,
            guidance=guidance,
            return_dict=True,
        ).sample
        # Note: lora_target_modules ["to_k","to_q","to_v","to_out.0"] may miss FLUX's
        # double-stream text projections (add_q_proj etc.) -- revisit if convergence is slow.
        flow_target = packed_noise - packed_latents
        loss = torch.mean(
            (weighting.float() * (pred_packed.float() - flow_target.float()) ** 2).reshape(
                pred_packed.shape[0], -1
            ),
            dim=1,
        )
        return loss.mean()


class FluxDevTrainer(FluxBaseTrainer):
    name = "flux.1-dev"
    hf_repo = "black-forest-labs/FLUX.1-dev"


class FluxSchnellTrainer(FluxBaseTrainer):
    name = "flux.1-schnell"
    hf_repo = "black-forest-labs/FLUX.1-schnell"

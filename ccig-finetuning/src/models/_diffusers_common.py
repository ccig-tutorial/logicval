from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, ClassVar

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.common.dataset import load_examples
from src.common.types import TrainConfig

from .base import LoraTrainer


class _ImagePromptDataset(Dataset):
    def __init__(self, dataset_path: str, images_dir: str, width: int, height: int) -> None:
        self.examples = list(load_examples(dataset_path, images_dir))
        self.width = width
        self.height = height

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]

        # Open the image, convert to RGB, and resize to (width, height). CLEVR's images
        # are already 3:2 (480x320), matching this dataset's target ratio (e.g.
        # 1024x688), so this is a plain undistorted resize -- not a stretch to a
        # mismatched ratio, and not a letterbox pad.
        image = Image.open(ex.image_path).convert("RGB").resize((self.width, self.height))

        # Convert to [-1, 1] float32 tensor in CHW order, as expected by the VAE encoder.
        pixel_values = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0
        return {"pixel_values": pixel_values, "prompt": ex.prompt}


class DiffusersLoraTrainer(LoraTrainer):
    """Shared LoRA finetuning scaffold for models built on a `diffusers` pipeline.

    Handles the model-agnostic parts once -- pipeline loading, LoRA injection via
    `peft`, the dataset/dataloader, the optimizer loop, and checkpoint saving (via
    PEFT's `save_pretrained`, matching ccig-image-generation's `--checkpoint` loader).

    `_training_step` (forward pass + loss) is deliberately NOT given a generic
    default: diffusers architectures differ in scheduler type (flow-matching vs.
    DDPM) and transformer forward signature (single vs. packed latents, extra
    positional ids, pooled projections, ...), so a one-size-fits-all step would
    silently produce wrong gradients for at least half of these models. Each
    subclass implements its own, informed by that model's actual pipeline.
    """

    pipeline_cls: ClassVar[Any]
    torch_dtype: ClassVar[Any] = torch.bfloat16
    lora_target_modules: ClassVar[list[str]] = ["to_k", "to_q", "to_v", "to_out.0"]
    # Pipe attribute names for the frozen (non-trained) components. These stay on CPU by
    # default and are moved to GPU only transiently -- for the specific encode call that
    # needs them, via _run_frozen_on_cuda -- because these models' full bf16 weights (the
    # transformer alone is 8-20B+ params) can fill an entire 24GB GPU by themselves,
    # leaving no room for the LoRA-trainable transformer's activations/gradients/optimizer
    # state if kept resident on GPU too. Subclasses override this with their actual
    # text-encoder attribute name(s) (see FluxPipeline/StableDiffusion3Pipeline/
    # QwenImagePipeline's __init__ signatures).
    frozen_module_names: ClassVar[list[str]] = ["vae", "text_encoder"]

    def _load_pipeline(self) -> Any:
        # Load on CPU. The transformer moves to GPU explicitly in train(), after LoRA
        # injection; frozen_module_names stay on CPU (see _run_frozen_on_cuda) until
        # transiently needed.
        return self.pipeline_cls.from_pretrained(self.hf_repo, torch_dtype=self.torch_dtype)

    def _run_frozen_on_cuda(self, pipe: Any, fn: Callable[[], Any]) -> Any:
        """Temporarily move this trainer's frozen_module_names to CUDA, run fn(), then move
        them back to CPU to free VRAM for the resident, trainable transformer. No-op
        passthrough when CUDA isn't available (e.g. local CPU testing).
        """
        if not torch.cuda.is_available():
            return fn()
        modules = [getattr(pipe, name) for name in self.frozen_module_names]
        for module in modules:
            module.to("cuda")
        try:
            return fn()
        finally:
            for module in modules:
                module.to("cpu")
            torch.cuda.empty_cache()

    def train(self, config: TrainConfig) -> Path:
        torch.manual_seed(config.seed)

        # Load the diffusers pipeline and inject LoRA into the transformer.
        pipe = self._load_pipeline()

        if config.init_ckpt:
            # Continue training from a previous run's saved adapter -- loads that
            # checkpoint's own LoRA config (rank/alpha/target_modules) and weights, rather
            # than initializing fresh ones. config.lora_rank/lora_alpha are ignored here.
            transformer = PeftModel.from_pretrained(pipe.transformer, config.init_ckpt, is_trainable=True)
        else:
            # LoRA config: rank, alpha, init method, and which transformer modules to inject into.
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=self.lora_target_modules,
            )
            transformer = get_peft_model(pipe.transformer, lora_config)

        # Move only the transformer to GPU -- frozen_module_names (VAE, text encoder(s))
        # stay on CPU; see _run_frozen_on_cuda.
        pipe.transformer = transformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        transformer.to(device)
        transformer.train()

        # Freeze the non-trained components (no gradients, eval mode / dropout off).
        for name in self.frozen_module_names:
            module = getattr(pipe, name)
            module.eval()
            module.requires_grad_(False)

        # Dataset, dataloader, and optimizer setup.
        dataset = _ImagePromptDataset(
            config.dataset_path, config.images_dir, config.resolution_width, config.resolution_height
        )
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(
            (p for p in transformer.parameters() if p.requires_grad), lr=config.learning_rate
        )

        # Held-out eval set (never trained on): same loss computation as training, just
        # under no_grad with no optimizer step, so it measures generalization instead of
        # how well the model fits the current training batch.
        eval_loader = None
        if config.eval_dataset_path:
            eval_dataset = _ImagePromptDataset(
                config.eval_dataset_path, config.images_dir, config.resolution_width, config.resolution_height
            )
            eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

        # max_steps=None means either epochs*steps_per_epoch (if epochs is set) or one full
        # epoch over dataset_path (if neither is set) -- resolved here, not in TrainConfig,
        # since it depends on the actual size of whatever dataset_path points at (e.g. a
        # 1921-record batch file vs. the full dataset). TrainConfig.__post_init__ already
        # guarantees at most one of max_steps/epochs is set.
        steps_per_epoch = len(loader)
        if config.max_steps is not None:
            max_steps = config.max_steps
        elif config.epochs is not None:
            max_steps = config.epochs * steps_per_epoch
            print(
                f"[{self.name}] epochs={config.epochs} -> max_steps={max_steps} "
                f"({steps_per_epoch} steps/epoch, {len(dataset)} examples)"
            )
        else:
            max_steps = steps_per_epoch
            print(f"[{self.name}] max_steps/epochs not set -- defaulting to one epoch: {max_steps} steps over {len(dataset)} examples")

        def _cycle(dl):
            while True:
                yield from dl

        # Training loop: forward pass, loss, backward, optimizer step, logging, checkpointing.
        # tqdm gives a live progress bar with steps/sec and estimated time remaining, which
        # reads cleanly through `tee` into a log file (as start_tmux.sh redirects to), same as
        # the download progress bars already seen from huggingface_hub/diffusers.
        #
        # Checkpoint retention: only ONE checkpoint is ever kept on disk, but which one is
        # decided by eval loss, not recency -- best_ckpt_dir/best_eval_loss track the best
        # checkpoint seen so far, and a newly-saved checkpoint is discarded immediately
        # (instead of replacing the kept one) if its eval loss is worse. Without this, a
        # model that peaks mid-training and then overfits/drifts on later steps would have
        # its best checkpoint silently deleted in favor of a worse, merely-more-recent one.
        # When no eval_dataset_path is configured there's no metric to compare against, so
        # this falls back to the old keep-only-latest behavior.
        #
        # This state also needs to survive ACROSS separate runs (e.g. batch1's process
        # exiting, then batch2 starting fresh via --init-ckpt) -- otherwise a batch whose
        # own steps all happen to be worse than an earlier batch's best would still keep
        # its own (worse) checkpoint, silently regressing what's on disk. best_state.json,
        # written next to the checkpoints in output_dir, is how that carries over: it's
        # read here to seed best_ckpt_dir/best_eval_loss instead of starting fresh at None.
        best_state_path = Path(config.output_dir) / "best_state.json"
        best_ckpt_dir: Path | None = None
        best_eval_loss: float | None = None
        if best_state_path.exists():
            try:
                prior_best = json.loads(best_state_path.read_text(encoding="utf-8"))
                prior_ckpt_dir = Path(prior_best["checkpoint_dir"])
                if prior_ckpt_dir.exists():
                    best_ckpt_dir = prior_ckpt_dir
                    best_eval_loss = prior_best["eval_loss"]
                    print(
                        f"[{self.name}] seeding checkpoint retention from a previous run: "
                        f"best so far is {best_ckpt_dir} (eval_loss={best_eval_loss:.4f})"
                    )
            except (json.JSONDecodeError, KeyError, OSError):
                pass  # Corrupt or unreadable state file -- fall back to starting fresh.
        last_eval_loss: float | None = None
        step = 0
        with tqdm(total=max_steps, initial=0, desc=f"[{self.name}]", unit="step") as pbar:
            for batch in _cycle(loader):
                if step >= max_steps:
                    break
                loss = self._training_step(pipe, transformer, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

                is_ckpt_step = step % config.checkpoint_every == 0 or step == max_steps
                # Also (re-)compute eval loss at every checkpoint step, even if it doesn't
                # land on an eval_every boundary -- the retention decision below needs a
                # loss value that actually corresponds to *this* checkpoint's weights, not
                # a possibly-stale one from a few steps earlier.
                if eval_loader is not None and (step % config.eval_every == 0 or is_ckpt_step):
                    last_eval_loss = self._compute_eval_loss(pipe, transformer, eval_loader)
                    print(
                        f"[{self.name}] step {step}/{max_steps} "
                        f"eval_loss={last_eval_loss:.4f} (n={len(eval_loader.dataset)})"
                    )

                postfix = {"loss": f"{loss.item():.4f}"}
                if last_eval_loss is not None:
                    postfix["eval_loss"] = f"{last_eval_loss:.4f}"
                pbar.set_postfix(**postfix)
                pbar.update(1)

                if is_ckpt_step:
                    ckpt_dir = self._save_checkpoint(transformer, config, step)
                    if eval_loader is None:
                        # No eval configured -- nothing to compare against, keep only the latest.
                        if best_ckpt_dir is not None and best_ckpt_dir != ckpt_dir:
                            shutil.rmtree(best_ckpt_dir, ignore_errors=True)
                        best_ckpt_dir = ckpt_dir
                    elif best_eval_loss is None or last_eval_loss <= best_eval_loss:
                        if best_ckpt_dir is not None and best_ckpt_dir != ckpt_dir:
                            shutil.rmtree(best_ckpt_dir, ignore_errors=True)
                        best_ckpt_dir = ckpt_dir
                        best_eval_loss = last_eval_loss
                        best_state_path.write_text(
                            json.dumps({"checkpoint_dir": str(best_ckpt_dir), "eval_loss": best_eval_loss}),
                            encoding="utf-8",
                        )
                        print(f"[{self.name}] step {step}: new best checkpoint (eval_loss={last_eval_loss:.4f}) -> {ckpt_dir}")
                    else:
                        # Worse than the checkpoint already kept (whether from this run or a
                        # previous one) -- discard this one immediately; best_state.json is
                        # left untouched, still pointing at the real best.
                        shutil.rmtree(ckpt_dir, ignore_errors=True)
                        print(
                            f"[{self.name}] step {step}: checkpoint discarded "
                            f"(eval_loss={last_eval_loss:.4f} > best {best_eval_loss:.4f} at {best_ckpt_dir})"
                        )

        return best_ckpt_dir

    def _compute_eval_loss(self, pipe: Any, transformer: Any, eval_loader: DataLoader) -> float:
        # Same _training_step loss computation as training, just under no_grad (no autograd
        # graph, no backward) and with no optimizer step -- this measures how well the
        # current weights generalize to images the model has never been trained on.
        # A nested progress bar here matters: the outer training bar doesn't move for the
        # whole duration of this loop (it only advances once per *training* step), so without
        # this, evaluating ~200 held-out examples looks like training silently froze.
        transformer.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"[{self.name}] eval", unit="ex", leave=False):
                loss = self._training_step(pipe, transformer, batch)
                total_loss += loss.item()
                n += 1
        transformer.train()
        return total_loss / max(n, 1)

    def _save_checkpoint(self, transformer: Any, config: TrainConfig, step: int) -> Path:
        # Save the LoRA weights in a directory that ccig-image-generation can load with its
        # `--checkpoint` flag. Whether this checkpoint or a previously-saved one is kept
        # around afterward is decided by the caller (train()), based on eval loss.
        ckpt_dir = config.checkpoint_dir_for_step(step)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        transformer.save_pretrained(str(ckpt_dir))
        print(f"[{self.name}] saved checkpoint at step {step} -> {ckpt_dir}")
        return ckpt_dir

    def _training_step(self, pipe: Any, transformer: Any, batch: dict) -> torch.Tensor:
        raise NotImplementedError

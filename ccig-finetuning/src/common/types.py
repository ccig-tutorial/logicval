from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainExample:
    id: int
    image_path: Path
    prompt: str


@dataclass
class TrainConfig:
    dataset_path: str
    images_dir: str
    output_dir: str
    run_name: str = "run1"
    # CLEVR's native images are 480x320 (3:2 aspect ratio). Training at a plain square
    # resolution (the old `resolution: int` field) force-resized every image, stretching
    # the scene. These two fields instead resize to the native ratio scaled up so the
    # long side is 1024 and the short side is a multiple of 16 (required by the VAE's
    # downsampling + transformer patching) -- 1024x688 -- so training sees undistorted
    # images. ccig-image-generation generates at this same shape (matching what the LoRA
    # actually learned) and pads the result up to a delivered 1024x1024 file afterward;
    # see its open/postprocess.py.
    resolution_width: int = 1024
    resolution_height: int = 688
    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 16
    batch_size: int = 1
    # None (the default) means "one full epoch over dataset_path" -- DiffusersLoraTrainer.train()
    # resolves this to len(dataloader) once the dataset is loaded, since that depends on
    # dataset_path's actual size. Set an explicit number of steps to override (e.g. to stop
    # partway through an epoch). Set at most one of max_steps/epochs.
    max_steps: int | None = None
    # Alternative to max_steps: train for this many full passes over dataset_path instead of a
    # raw step count -- max_steps = epochs * steps_per_epoch, resolved once the dataset is
    # loaded (see DiffusersLoraTrainer.train()). More portable than max_steps across datasets
    # of different sizes (e.g. batch_001.json vs. batch_002.json). Leaving both max_steps and
    # epochs unset defaults to a single epoch, same as epochs=1.
    epochs: int | None = None
    seed: int = 42
    # Save a checkpoint every N steps (and always at max_steps), deleting the previous
    # one -- see DiffusersLoraTrainer._save_checkpoint. Keeps disk usage to one
    # checkpoint at a time instead of one per save.
    checkpoint_every: int = 500
    # Path to a previous run's checkpoint directory (as returned by train(), e.g.
    # ../outputs/checkpoints-finetuning/flux.1-dev/batch1-step002000) to continue LoRA
    # training from, instead of initializing fresh LoRA weights. lora_rank/lora_alpha are
    # ignored when set -- the saved adapter's own config is used. See
    # DiffusersLoraTrainer.train().
    init_ckpt: str | None = None
    # Path to a held-out dataset (same schema as dataset_path) never used for training --
    # e.g. data/clevr-dataset/finetuning-data/eval_holdout.json. When set, eval loss is computed on the
    # whole eval set every eval_every steps (same loss formula as training, just under
    # no_grad with no optimizer step) and logged alongside training loss, so you can tell
    # whether the model is generalizing or just memorizing the current training batch. None
    # (the default) disables eval entirely.
    eval_dataset_path: str | None = None
    eval_every: int = 50

    def __post_init__(self) -> None:
        if self.max_steps is not None and self.epochs is not None:
            raise ValueError("TrainConfig: set only one of max_steps or epochs, not both.")

    def checkpoint_dir_for_step(self, step: int) -> Path:
        return Path(self.output_dir) / f"{self.run_name}-step{step:06d}"

from __future__ import annotations

import argparse

import transformers
import yaml

from src.common.types import TrainConfig
from src.registry import TRAINER_REGISTRY, build_trainer

# Silences transformers' per-call "input was truncated because CLIP can only handle
# sequences up to 77 tokens" warning (and other INFO/WARNING-level transformers log
# spam). Harmless to suppress here: only CLIP's secondary pooled embedding truncates,
# not the T5 embedding that actually carries the full prompt to the transformer.
# Errors still surface -- this only raises the verbosity floor to ERROR.
transformers.logging.set_verbosity_error()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA finetune an open-source text-to-image model on the CCIG dataset."
    )
    parser.add_argument("--model", required=True, choices=sorted(TRAINER_REGISTRY))
    parser.add_argument("--config", required=True, help="Path to a configs/<model>.yaml file")
    parser.add_argument("--run-name", default=None, help="Overrides run_name from the config")
    parser.add_argument("--max-steps", type=int, default=None, help="Overrides max_steps from the config")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Train for this many full passes over the dataset instead of --max-steps "
        "(more portable across datasets of different sizes); set at most one of the two",
    )
    parser.add_argument("--dataset", default=None, help="Overrides dataset_path from the config")
    parser.add_argument(
        "--init-ckpt",
        default=None,
        help="Path to a previous run's checkpoint directory to continue LoRA training from",
    )
    parser.add_argument(
        "--eval-dataset", default=None,
        help="Path to a held-out dataset (same schema as --dataset) to compute eval loss "
        "against periodically during training; overrides eval_dataset_path from the config",
    )
    parser.add_argument(
        "--eval-every", type=int, default=None,
        help="Compute eval loss every N steps (only used when an eval dataset is set); "
        "overrides eval_every from the config",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if args.run_name is not None:
        raw_config["run_name"] = args.run_name
    if args.max_steps is not None:
        raw_config["max_steps"] = args.max_steps
    if args.epochs is not None:
        raw_config["epochs"] = args.epochs
    if args.dataset is not None:
        raw_config["dataset_path"] = args.dataset
    if args.init_ckpt is not None:
        raw_config["init_ckpt"] = args.init_ckpt
    if args.eval_dataset is not None:
        raw_config["eval_dataset_path"] = args.eval_dataset
    if args.eval_every is not None:
        raw_config["eval_every"] = args.eval_every

    config = TrainConfig(**raw_config)
    trainer = build_trainer(args.model)
    checkpoint_dir = trainer.train(config)
    print(f"[{args.model}] LoRA checkpoint written to {checkpoint_dir}")


if __name__ == "__main__":
    main()

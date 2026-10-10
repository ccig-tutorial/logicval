from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.closed.registry import MODEL_REGISTRY as CLOSED_MODEL_REGISTRY
from src.closed.registry import build_model as build_closed_model
from src.common.io import append_manifest, load_prompts
from src.common.scene_setup import scene_setup_text, scene_unsat_text
from src.common.types import GenerationRecord
from src.open.registry import MODEL_REGISTRY as OPEN_MODEL_REGISTRY
from src.open.registry import build_model as build_open_model

ALL_MODELS = sorted({**CLOSED_MODEL_REGISTRY, **OPEN_MODEL_REGISTRY})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate images for CCIG eval prompts using a text-to-image model."
    )
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument(
        "--dataset", help="Path to a ccig_eval_dataset_{SAT,UNSAT}.jsonl file",
        default="../data/ccig_eval_dataset_SAT.jsonl"
    )
    parser.add_argument("--prompt-field", default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--out", default="../data/generated_images")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--start-id", type=int, default=None,
        help="Only process prompts with id >= this value -- e.g. to resume a run that failed "
        "partway through (check the manifest for the last successful id).",
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle prompts before applying --limit, for a random subset instead of the first N",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (only used with --shuffle)")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a LoRA finetuned checkpoint directory (open-source models only), "
        "as produced by ccig-finetuning.",
    )
    args = parser.parse_args()

    is_closed_model = args.model in CLOSED_MODEL_REGISTRY
    if args.model in OPEN_MODEL_REGISTRY:
        model = build_open_model(args.model, checkpoint=args.checkpoint)
    else:
        if args.checkpoint is not None:
            parser.error(f"--checkpoint is not supported for closed model '{args.model}'")
        model = build_closed_model(args.model)
    # For open-source models run against a LoRA checkpoint, fold the checkpoint's directory
    # name into the output folder name -- otherwise different checkpoints (e.g. batch1 vs.
    # batch2) would all write into the same "<model>/" folder and overwrite each other's
    # images/manifest. Same idea for closed models that expose a `quality` or `image_size`
    # setting (e.g. gpt-image-2 low vs. medium) -- fold that into the folder name too, so
    # different settings don't clobber each other.
    variant = getattr(model, "quality", None) or getattr(model, "image_size", None)
    out_dir_name = args.model
    if args.checkpoint is not None:
        out_dir_name = f"{out_dir_name}-{Path(args.checkpoint).name}"
    if variant is not None:
        out_dir_name = f"{out_dir_name}-{variant}"
    # Group by dataset within the model's folder (e.g. "gpt-image-2-low/clevr_3_scenes_SAT/")
    # so images from different eval files don't mix together in one flat folder.
    dataset_name = Path(args.dataset).stem
    out_dir = Path(args.out) / out_dir_name / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    prompts = list(load_prompts(args.dataset, args.prompt_field))
    if args.start_id is not None:
        prompts = [p for p in prompts if int(p.id) >= args.start_id]
    if args.shuffle:
        random.Random(args.seed).shuffle(prompts)

    for i, item in enumerate(prompts):
        if args.limit is not None and i >= args.limit:
            break

        # check if outdir exists... if not, create it
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)


        image_path = out_dir / f"{item.id}-{args.prompt_field}.png"
        setup_text = scene_setup_text(
            item.number_of_objects, item.domain, with_background=is_closed_model
        )
        # Closed models get the unsat-handling sentence appended; open models keep the
        # domain module's default (currently off) for now.
        unsat_text = scene_unsat_text(item.domain, with_unsat=is_closed_model)
        full_prompt = f"{setup_text} {item.text}{unsat_text}"
        try:
            image = model.generate(full_prompt)
            image.save(image_path)
            record = GenerationRecord(
                id=item.id,
                model=args.model,
                prompt=item.text,
                prompt_field=args.prompt_field,
                scene_generation_setup=f"{setup_text}{unsat_text}",
                image_path=str(image_path),
                success=True,
                error=None,
                variant=variant,
            )
            print(f"[ok]   {item.id}")
        except Exception as e:
            record = GenerationRecord(
                id=item.id,
                model=args.model,
                prompt=item.text,
                prompt_field=args.prompt_field,
                scene_generation_setup=f"{setup_text}{unsat_text}",
                image_path=None,
                success=False,
                error=repr(e),
                variant=variant,
            )
            print(f"[fail] {item.id}: {e}")

        append_manifest(manifest_path, record)


if __name__ == "__main__":
    main()

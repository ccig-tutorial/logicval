"""Subsample finetune_prompts_clevr_train_filtered.json to one record per image.

The source file has ~77-171 records per image (every grounded class/variant/style
combination), with a heavily class-imbalanced overall distribution (e.g. C1: 669k rows
vs. C9: 120k rows). Sampling one row per image uniformly at random would inherit that
imbalance. Instead, for each image: pick a class uniformly at random from the classes
available for that image, then pick one random record within that (image, class) group.
That gives one sample per image while spreading class selection evenly.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../data/clevr-dataset/finetune_prompts_clevr_train_filtered.json")
    parser.add_argument("--output", default="../data/clevr-dataset/finetune-dataset-one-per-image.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with Path(args.input).open("r", encoding="utf-8") as f:
        records = json.load(f)

    by_image_then_class: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        by_image_then_class[rec["image_id"]][rec["class"]].append(rec)

    sampled = []
    for image_id in sorted(by_image_then_class):
        classes_for_image = by_image_then_class[image_id]
        chosen_class = rng.choice(sorted(classes_for_image))
        chosen_record = rng.choice(classes_for_image[chosen_class])
        sampled.append(chosen_record)

    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2)

    print(f"Sampled {len(sampled)} records (1 per image) from {len(records)} -> {args.output}")


if __name__ == "__main__":
    main()

"""Build one fixed, class-balanced held-out eval set, excluding every image used in any
existing training batch (batch_*.json).

Reads the already-generated batch_*.json files directly (not a reconstruction of how they were
built) and excludes any image_id appearing in any of them -- for any class/variant, not just the
one it was originally paired with -- so held-out images are completely unseen by training,
regardless of which batch or which constraint. From the remaining pool, draws one record per
image for a class-balanced set of eval examples: no image repeats within the eval set either.

Build this once. Because fine-tuning here is incremental (batch1 -> checkpoint -> batch2 ->
checkpoint -> ...), the same fixed eval set stays valid and comparable across every future
stage, as long as no later batch is built from images already in it.
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
    parser.add_argument("--batches-dir", default="../data/clevr-dataset/finetuning-data/batches")
    parser.add_argument("--output", default="../data/clevr-dataset/finetuning-data/eval_holdout.json")
    parser.add_argument("--eval-size", type=int, default=200, help="Total size of the held-out eval set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    batch_files = sorted(Path(args.batches_dir).glob("batch_*.json"))
    if not batch_files:
        raise SystemExit(f"No batch_*.json files found under {args.batches_dir}")

    used_image_ids: set[int] = set()
    for bf in batch_files:
        with bf.open("r", encoding="utf-8") as f:
            for rec in json.load(f):
                used_image_ids.add(rec["image_id"])
    print(f"Excluding {len(used_image_ids)} image_ids used across {len(batch_files)} batch file(s): "
          f"{', '.join(p.name for p in batch_files)}")

    with Path(args.input).open("r", encoding="utf-8") as f:
        records = json.load(f)

    # One record per (class, image_id), among images not already used in any batch.
    by_class_image: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec["image_id"] in used_image_ids:
            continue
        by_class_image[rec["class"]][rec["image_id"]].append(rec)

    classes = sorted(by_class_image)
    per_class = max(1, args.eval_size // len(classes))

    eval_set = []
    chosen_image_ids: set[int] = set()  # no image repeats within the eval set, even across classes
    shortfalls = []
    for cls in classes:
        image_ids = [iid for iid in by_class_image[cls] if iid not in chosen_image_ids]
        rng.shuffle(image_ids)
        take_ids = image_ids[:per_class]
        if len(take_ids) < per_class:
            shortfalls.append(f"{cls}({len(take_ids)}/{per_class})")
        for iid in take_ids:
            eval_set.append(rng.choice(by_class_image[cls][iid]))
            chosen_image_ids.add(iid)

    rng.shuffle(eval_set)

    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2)

    print(f"{len(classes)} classes; target {per_class} instances per class")
    print(f"Held-out eval set: {len(eval_set)} records, {len(chosen_image_ids)} unique images -> {args.output}")
    if shortfalls:
        print(f"Classes short of target (ran out of unused images): {', '.join(shortfalls)}")

    overlap = sum(1 for rec in eval_set if rec["image_id"] in used_image_ids)
    print(f"Overlap check with training batches: {overlap} (should be 0)")


if __name__ == "__main__":
    main()

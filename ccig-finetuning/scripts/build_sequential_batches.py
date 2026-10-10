"""Split finetune_prompts_clevr_train_filtered.json into sequential, (class, variant)-balanced
batches.

Each batch aims for the same number of instances from every (class, variant) pair (e.g.
C1/1prop, C8/2prop_exact_neg, ...) -- not just the same number per class. When a pair doesn't
have enough remaining images left to hit that per-batch target, it contributes whatever it has
left instead (never more than the target). Batches are non-overlapping: once an image has been
used for a given (class, variant) pair, it's never reused for that same pair in a later batch.

Because classes have very different numbers of variants (2-18) and per-variant image
availability varies just as widely (as few as 1 image, as many as 21,083), batch sizes and
per-class totals will NOT be equal across classes -- that's the direct, accepted trade-off of
guaranteeing every variant gets sampled rather than only balancing at the class level.
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
    parser.add_argument("--output-dir", default="../data/clevr-dataset/finetuning-data/batches")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Target total instances per batch; divided evenly across all (class, variant) pairs",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=10,
        help="Stop after this many batches even if some pairs still have data left",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with Path(args.input).open("r", encoding="utf-8") as f:
        records = json.load(f)

    # Group by (class, variant, image_id): a given (class, variant) pair can have up to 3
    # records per image (one per prompt style -- short/medium/long, verbalizing the same
    # grounded constraint); pick one at random per image so each image contributes at most
    # once to a given pair.
    by_pair_image: dict[tuple[str, str], dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        by_pair_image[(rec["class"], rec["variant"])][rec["image_id"]].append(rec)

    pairs = sorted(by_pair_image)
    pools: dict[tuple[str, str], list[dict]] = {}
    for pair in pairs:
        pool = [rng.choice(recs) for recs in by_pair_image[pair].values()]
        rng.shuffle(pool)
        pools[pair] = pool

    target_per_pair = max(1, args.batch_size // len(pairs))
    print(f"{len(pairs)} (class, variant) pairs; target {target_per_pair} instances per pair per batch\n")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cursor = {pair: 0 for pair in pairs}
    batch_idx = 0
    while batch_idx < args.max_batches:
        batch = []
        pair_counts: dict[tuple[str, str], int] = {}
        for pair in pairs:
            pool = pools[pair]
            start = cursor[pair]
            take = pool[start : start + target_per_pair]
            cursor[pair] = start + len(take)
            pair_counts[pair] = len(take)
            batch.extend(take)

        if not batch:
            print(f"All {len(pairs)} pairs exhausted -- stopping after {batch_idx} batches.")
            break

        rng.shuffle(batch)
        batch_idx += 1
        out_path = out_dir / f"batch_{batch_idx:03d}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(batch, f, indent=2)

        short_pairs = [f"{c}/{v}({n})" for (c, v), n in pair_counts.items() if n < target_per_pair]
        print(f"batch_{batch_idx:03d}: {len(batch)} records -> {out_path}")
        if short_pairs:
            print(f"    below target ({target_per_pair}): {', '.join(short_pairs)}")

    print("\nRemaining images per pair not yet used in any batch:")
    any_remaining = False
    for pair in pairs:
        remaining = len(pools[pair]) - cursor[pair]
        if remaining > 0:
            any_remaining = True
            print(f"    {pair[0]}/{pair[1]}: {remaining} left")
    if not any_remaining:
        print("    (none -- every pair fully consumed)")


if __name__ == "__main__":
    main()

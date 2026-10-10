from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .types import TrainExample


def load_examples(dataset_path: str | Path, images_dir: str | Path) -> Iterator[TrainExample]:
    """Read data/clevr-dataset/finetune_prompts_clevr_train_filtered.json and yield one
    training example per record, pairing each CLEVR image with its already-verbalized prompt
    (the prompt style -- short/medium/long -- was chosen at generation time, one per record).
    """
    images_dir = Path(images_dir)
    with Path(dataset_path).open("r", encoding="utf-8") as f:
        records = json.load(f)

    for rec in records:
        yield TrainExample(
            id=rec["id"],
            image_path=images_dir / rec["image_filename"],
            prompt=rec["prompt"],
        )

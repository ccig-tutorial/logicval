from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .types import GenerationRecord, PromptRecord


def load_prompts(path: str | Path, prompt_field: str = "medium") -> Iterator[PromptRecord]:
    """Read a ccig_eval_dataset_{SAT,UNSAT}.jsonl file and yield one prompt per record."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield PromptRecord(
                id=rec["id"],
                text=rec["prompts"][prompt_field],
                number_of_objects=rec["number_of_objects"],
                domain=rec["domain"],
                status=rec["status"],
                complexity_class=rec["complexity_class"],
            )


def append_manifest(path: str | Path, record: GenerationRecord) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_json()) + "\n")

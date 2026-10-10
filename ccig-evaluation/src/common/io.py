from __future__ import annotations

import json
import re
from pathlib import Path

from .types import MatchedItem, PromptRecord

# ccig-image-generation writes generated images as "{id}-{prompt_field}.png". Some
# older manifest rows predate the prompt_field suffix and were saved as bare "{id}.png" --
# those are skipped (not matchable to a specific prompt field) rather than guessed at.
_IMAGE_NAME_RE = re.compile(r"^(?P<id>.+)-(?P<field>short|medium|long)\.png$")


def discover_images(images_dir: str | Path) -> list[tuple[str, str, Path]]:
    """Scan images_dir for '{id}-{prompt_field}.png' files.

    Returns a list of (id, prompt_field, path), skipping (with a printed warning)
    any .png that doesn't match the naming convention.
    """
    images_dir = Path(images_dir)
    found: list[tuple[int, str, Path]] = []
    for path in sorted(images_dir.glob("*.png")):
        id_im = int (path.name.split('-')[0])
        prompt = path.name.split('-')[1].split('.')[0]
        found.append((id_im, prompt, path))
        
    return found


def load_prompt_records(prompts_file: str | Path) -> dict[str, PromptRecord]:
    """Read a ccig_eval_dataset_{SAT,UNSAT}.jsonl file -> {id: PromptRecord}."""
    records: dict[str, PromptRecord] = {}
    with Path(prompts_file).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["id"]] = PromptRecord(
                id=rec["id"],
                domain=rec["domain"],
                complexity_class=rec["complexity_class"],
                constraint_family=rec["constraint_family"],
                prompts=rec["prompts"],
                instantiated_rule=rec["instantiated_rule"],
                asp_template_file=rec["asp_template_file"],
                status=rec["status"],
                number_of_objects=rec["number_of_objects"],
                # .get(): absent on datasets generated before the subqa field existed --
                # those just can't be scored by --method soft-tifa (see its README).
                subqa=rec.get("subqa", {}),
            )
    return records


def match_images_to_prompts(images_dir: str | Path, prompts_file: str | Path) -> list[MatchedItem]:
    """Join discovered images to their prompt record by id. Unmatched ids on either
    side are warned about and skipped, not treated as fatal -- a partial dataset
    (e.g. a generation run that failed on some prompts) should still be evaluable.
    """
    records = load_prompt_records(prompts_file)
    
    matched: list[MatchedItem] = []
    
    for image_id, prompt_field, path in discover_images(images_dir):
        record = records.get(image_id)
        if record is None:
            print(f"[warn] no prompt record for image id '{image_id}', skipping")
            continue
        matched.append(MatchedItem(id=image_id, prompt_field=prompt_field, image_path=path, record=record))
    
    return matched


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

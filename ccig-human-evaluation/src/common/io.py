from __future__ import annotations

import json
import os
import re
import tempfile
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
    found: list[tuple[str, str, Path]] = []
    for path in sorted(images_dir.glob("*.png")):
        m = _IMAGE_NAME_RE.match(path.name)
        if not m:
            print(f"[warn] skipping '{path.name}': does not match '{{id}}-{{short,medium,long}}.png'")
            continue
        found.append((m.group("id"), m.group("field"), path))
    return found


def _flatten(value):
    """combo=1 dataset files wrap several fields (complexity_class,
    constraint_family, instantiated_rule, asp_template_file) in one-element
    lists, and use an int id -- while image filenames (and this module's own
    discover_images) use plain strings. Same workaround as
    perception/run_single_example.py::_flatten / load_one_record, applied
    here so the general batch-matching path (match_images_to_prompts) isn't
    silently broken for these files the way it is in the upstream copy this
    was taken from."""
    if isinstance(value, list):
        return "\n".join(str(v) for v in value) if len(value) != 1 else str(value[0])
    return value


def load_prompt_records(prompts_file: str | Path) -> dict[str, PromptRecord]:
    """Read a ccig_eval_dataset_{SAT,UNSAT}.jsonl file -> {str(id): PromptRecord}."""
    records: dict[str, PromptRecord] = {}
    with Path(prompts_file).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            record_id = str(rec["id"])
            records[record_id] = PromptRecord(
                id=record_id,
                domain=rec["domain"],
                complexity_class=_flatten(rec["complexity_class"]),
                constraint_family=_flatten(rec["constraint_family"]),
                prompts=rec["prompts"],
                instantiated_rule=_flatten(rec["instantiated_rule"]),
                asp_template_file=_flatten(rec["asp_template_file"]),
                status=rec["status"],
                number_of_objects=rec["number_of_objects"],
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
    """Write via a temp file + atomic os.replace, never directly into `path`.

    A direct `open(path, "w")` truncates the file immediately and then streams
    content into it over time -- if a second process does the same thing to the
    same path around the same time (e.g. two `python -m src.app` instances both
    pointed at the same batch), their writes can interleave mid-file and leave
    behind invalid JSON that's silently corrupted, not just out of date. This
    happened for real once already (see human-perception.json recovery in git
    history). os.replace is atomic on the same filesystem, so any reader (or
    another writer) only ever sees the fully-old or fully-new file -- concurrent
    writers can still race on *which* version wins, but the file itself is never
    torn."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise

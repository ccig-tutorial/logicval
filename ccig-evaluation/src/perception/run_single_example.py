#!/usr/bin/env python3
"""Run the perception pipeline on exactly one image.

Standalone -- doesn't touch the rest of ccig-evaluation's source. It reuses the
real pipeline (run.py::run_perception) for everything, and only works around
the current id-type / instantiated_rule-list mismatch between the
filename-derived id (a string) and the ccig_eval_dataset records (id as an
int, instantiated_rule/complexity_class/etc. as single-element lists) --
see README.md, "Running it" section, for why that mismatch exists.

Usage:
    python -m src.perception.run_single_example <image_path> <prompts_file> \
        [--domain clevr] [--detector owlv2] [--device cpu] [--out result.json]

Example (from ccig-evaluation/):
    python -m src.perception.run_single_example \
        ../data/generated_images/flux.1-dev-batch1-step001921/26-medium.png \
        ../data/ccig_eval_dataset/clevr_1_scenes_SAT.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Make `import src...` resolve the same way it does for `python -m src.run`,
# regardless of the caller's cwd.
_CCIG_EVALUATION_DIR = Path(__file__).resolve().parents[2]
if str(_CCIG_EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(_CCIG_EVALUATION_DIR))

from src.common.types import MatchedItem, PromptRecord  # noqa: E402
from src.perception.run import run_perception  # noqa: E402

# Same filename convention as common/io.py::discover_images.
_IMAGE_NAME_RE = re.compile(r"^(?P<id>.+)-(?P<field>short|medium|long)\.png$")


def _flatten(value):
    """The dataset's list-wrapped fields (id/complexity_class/instantiated_rule/
    asp_template_file/constraint_family) collapse to their single element for a
    combo=1 record; join multi-element lists with newlines/commas just in case."""
    if isinstance(value, list):
        return "\n".join(str(v) for v in value) if len(value) != 1 else str(value[0])
    return value


def load_one_record(prompts_file: Path, image_id: str) -> PromptRecord:
    """Same job as common/io.py::load_prompt_records, but matches by str(id)
    (the file's id is an int) and flattens list-wrapped fields to plain strings
    (the file wraps them in one-element lists) so the ASP program built later
    is valid text, not a Python list repr."""
    with prompts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if str(rec["id"]) != image_id:
                continue
            return PromptRecord(
                id=str(rec["id"]),
                domain=rec["domain"],
                complexity_class=_flatten(rec["complexity_class"]),
                constraint_family=_flatten(rec["constraint_family"]),
                prompts=rec["prompts"],
                instantiated_rule=_flatten(rec["instantiated_rule"]),
                asp_template_file=_flatten(rec["asp_template_file"]),
                status=rec["status"],
                number_of_objects=rec["number_of_objects"],
            )
    raise ValueError(f"No record with id '{image_id}' found in {prompts_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run perception on a single image.")
    parser.add_argument("image_path")
    parser.add_argument("prompts_file")
    parser.add_argument("--domain", default="clevr", choices=["clevr", "coco"])
    parser.add_argument("--detector", default="owlv2", choices=["grounding-dino", "owlv2"])
    parser.add_argument("--attribute-classifier", default="clip-zero-shot")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="single_example_result.json")
    args = parser.parse_args()

    image_path = Path(args.image_path).resolve()
    m = _IMAGE_NAME_RE.match(image_path.name)
    if not m:
        raise ValueError(f"'{image_path.name}' doesn't match '{{id}}-{{short,medium,long}}.png'")
    image_id, prompt_field = m.group("id"), m.group("field")

    record = load_one_record(Path(args.prompts_file).resolve(), image_id)
    item = MatchedItem(id=image_id, prompt_field=prompt_field, image_path=image_path, record=record)

    print(f"Matched image id={image_id!r} field={prompt_field!r} -> status={record.status!r}")
    print(f"instantiated_rule: {record.instantiated_rule}")

    run_perception([item], args.domain, args.detector, args.attribute_classifier, args.device, Path(args.out))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

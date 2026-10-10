from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.closed.registry import MODEL_REGISTRY as CLOSED_MODEL_REGISTRY
from src.closed.registry import build_model as build_closed_model
from src.common.io import load_prompts
from src.common.scene_setup import scene_setup_text, scene_unsat_text
from src.common.types import GenerationRecord
from src.open.registry import MODEL_REGISTRY as OPEN_MODEL_REGISTRY
from src.open.registry import build_model as build_open_model

ALL_MODELS = sorted({**CLOSED_MODEL_REGISTRY, **OPEN_MODEL_REGISTRY})


def _iter_manifest_records(manifest_path: Path):
    """Yield each JSON record in a manifest.jsonl file.

    Some earlier partial-run manifests have two records concatenated on one physical line
    with no newline between them (e.g. a process that died mid-write, then got appended to
    again) -- a plain line-by-line json.loads() chokes on those with "Extra data". Parse the
    whole file as a stream of back-to-back JSON values instead, which tolerates that.
    """
    text = manifest_path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    pos = 0
    length = len(text)
    while pos < length:
        while pos < length and text[pos].isspace():
            pos += 1
        if pos >= length:
            break
        obj, end = decoder.raw_decode(text, pos)
        yield obj
        pos = end


def _is_valid_image(path: Path) -> bool:
    """True if `path` decodes as a real image, not just "exists with some bytes".

    Catches the case where an earlier attempt's image.save() failed partway through and left
    a truncated/corrupt file behind -- a plain size-check would wrongly treat that as
    recoverable and skip regenerating it.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def _resolve_out_dir(
    model, model_name: str, dataset_path: str, out_root: str, checkpoint: str | None
) -> tuple[Path, str | None]:
    # Same naming convention as src/run.py -- must match exactly, or this looks at the
    # wrong manifest.jsonl entirely.
    variant = getattr(model, "quality", None) or getattr(model, "image_size", None)
    out_dir_name = model_name
    if checkpoint is not None:
        out_dir_name = f"{out_dir_name}-{Path(checkpoint).name}"
    if variant is not None:
        out_dir_name = f"{out_dir_name}-{variant}"
    dataset_name = Path(dataset_path).stem
    return Path(out_root) / out_dir_name / dataset_name, variant


def repair_manifest(
    model_name: str,
    dataset_path: str,
    prompt_fields: list[str] | None = None,
    out_root: str = "../data/generated_images",
    checkpoint: str | None = None,
) -> None:
    """Fill in missing/errored rows of a manifest.jsonl by retrying generation once each.

    For every (id, prompt_field) pair expected for this dataset -- every dataset id crossed
    with `prompt_fields` -- checks the existing manifest:
    - a good row (success=True, error=None) is left alone; prints "[exists]".
    - a missing row, or one with success=False / error set, gets exactly one retry via
      model.generate(); the row is replaced with whatever that retry produced (success or
      another failure), in the same shape src/run.py writes.

    Rewrites manifest.jsonl after EVERY row it touches (not once at the end) -- so a crash,
    Ctrl-C, or an API outage partway through never loses progress already made, and anyone
    checking the file mid-run sees the true current state, not stale pre-run data. Rows for
    prompt_field values NOT in `prompt_fields` (e.g. an earlier run.py pass wrote "short" rows
    and this call only repairs "long") are carried through unchanged in every write -- this
    function must never be the reason a field's rows disappear.

    If a row is missing/errored in the manifest but its PNG already exists on disk (e.g. the
    manifest lost track of it, rather than generation never having happened), that's treated
    as recovered without calling the API again -- see "[recovered]" in the printed output.
    """
    is_closed_model = model_name in CLOSED_MODEL_REGISTRY
    if model_name in OPEN_MODEL_REGISTRY:
        model = build_open_model(model_name, checkpoint=checkpoint)
    else:
        if checkpoint is not None:
            raise ValueError(f"checkpoint is not supported for closed model '{model_name}'")
        model = build_closed_model(model_name)

    out_dir, variant = _resolve_out_dir(model, model_name, dataset_path, out_root, checkpoint)
    manifest_path = out_dir / "manifest.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    # manifest_state is the single source of truth for what gets written. It starts as
    # everything already on disk (any field, any id) and is only ever updated in place --
    # dict insertion order means keys we don't touch keep their original position, and keys
    # we do touch get their value replaced without disturbing anything else.
    manifest_state: dict[tuple[str, str], dict] = {}
    if manifest_path.exists():
        for row in _iter_manifest_records(manifest_path):
            manifest_state[(str(row["id"]), row["prompt_field"])] = row  # last one wins

    if prompt_fields is None:
        prompt_fields = sorted({field for (_id, field) in manifest_state})
        if not prompt_fields:
            raise ValueError(
                f"{manifest_path} is empty/missing and no prompt_fields were given -- pass "
                "prompt_fields explicitly, e.g. ['short', 'long']"
            )

    def _write_manifest() -> None:
        with manifest_path.open("w", encoding="utf-8") as f:
            for row in manifest_state.values():
                f.write(json.dumps(row) + "\n")

    seen_keys: set[tuple[str, str]] = set()
    n_exists = n_fixed = n_still_failing = n_recovered = 0
    for field in prompt_fields:
        for item in load_prompts(dataset_path, field):
            key = (str(item.id), field)
            if key in seen_keys:
                # A duplicate id within the dataset file itself -- not a manifest problem,
                # a data problem. Fail loudly rather than silently writing two rows for it.
                raise ValueError(
                    f"duplicate id {item.id!r} for prompt_field={field!r} in {dataset_path} "
                    "-- fix the dataset file, this would otherwise produce two manifest rows "
                    "for the same (id, prompt_field)"
                )
            seen_keys.add(key)
            row = manifest_state.get(key)
            if row is not None and row.get("success") and not row.get("error"):
                print(f"[exists] {item.id}-{field}")
                n_exists += 1
                continue

            image_path = out_dir / f"{item.id}-{field}.png"
            setup_text = scene_setup_text(
                item.number_of_objects, item.domain, with_background=is_closed_model
            )
            unsat_text = scene_unsat_text(item.domain, with_unsat=is_closed_model)

            # The manifest row is missing/bad, but the image itself may still be on disk --
            # e.g. an earlier manifest write got clobbered, not generation never happening.
            # Don't burn an API call re-generating something we already have -- but verify it
            # actually decodes first; a truncated/corrupt file (a prior partial write) should
            # still be regenerated, not blindly trusted.
            if _is_valid_image(image_path):
                print(f"[recovered] {item.id}-{field} (manifest row lost, image on disk)")
                manifest_state[key] = GenerationRecord(
                    id=item.id, model=model_name, prompt=item.text, prompt_field=field,
                    scene_generation_setup=f"{setup_text}{unsat_text}", image_path=str(image_path),
                    success=True, error=None, variant=variant,
                ).to_json()
                n_recovered += 1
                _write_manifest()
                continue

            reason = "missing" if row is None else f"error={row.get('error')!r}"
            print(f"[retry]  {item.id}-{field} ({reason})")

            full_prompt = f"{setup_text} {item.text}{unsat_text}"
            try:
                image = model.generate(full_prompt)
                image.save(image_path)
                record = GenerationRecord(
                    id=item.id, model=model_name, prompt=item.text, prompt_field=field,
                    scene_generation_setup=f"{setup_text}{unsat_text}", image_path=str(image_path),
                    success=True, error=None, variant=variant,
                )
                print(f"[fixed]  {item.id}-{field}")
                n_fixed += 1
            except Exception as e:
                record = GenerationRecord(
                    id=item.id, model=model_name, prompt=item.text, prompt_field=field,
                    scene_generation_setup=f"{setup_text}{unsat_text}", image_path=None,
                    success=False, error=repr(e), variant=variant,
                )
                print(f"[still-failing] {item.id}-{field}: {e}")
                n_still_failing += 1

            manifest_state[key] = record.to_json()
            _write_manifest()

    # Verify the invariant this function exists to guarantee -- exactly one row per
    # (id, prompt_field) -- rather than just trusting the loop above got it right. A plain
    # dict can't itself hold duplicate keys, but this also catches any row whose stored
    # "id"/"prompt_field" fields don't match the key it was filed under.
    for key, row in manifest_state.items():
        actual_key = (str(row["id"]), row["prompt_field"])
        assert actual_key == key, (
            f"manifest_state[{key!r}] has mismatched id/prompt_field {actual_key!r} -- "
            "refusing to write a manifest that violates the one-row-per-id+field invariant"
        )

    _write_manifest()

    print(
        f"\n{manifest_path}: {len(manifest_state)} rows total "
        f"({n_exists} already good, {n_recovered} recovered from disk, {n_fixed} fixed, "
        f"{n_still_failing} still failing)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retry missing/errored rows in a manifest.jsonl, in place, one retry each."
    )
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--prompt-field", nargs="+", choices=["short", "medium", "long"], default=None,
        dest="prompt_fields",
        help="Which prompt field(s) to check/repair. Defaults to whatever fields are already "
        "present in the existing manifest.",
    )
    parser.add_argument("--out", default="../data/generated_images")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a LoRA finetuned checkpoint directory (open-source models only).",
    )
    args = parser.parse_args()
    repair_manifest(
        model_name=args.model,
        dataset_path=args.dataset,
        prompt_fields=args.prompt_fields,
        out_root=args.out,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()

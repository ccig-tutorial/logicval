from __future__ import annotations

import json
from pathlib import Path

from ..common.dataset_gen import load_domain
from ..common.io import write_json
from ..common.types import MatchedItem, PerceptionResult
from .attributes.registry import build_attribute_classifier
from .crop import crop_and_neutralize
from .detectors.registry import build_detector
from .regions import bbox_center, region_of
from .scene_graph import to_graph_dict
from .types import DetectedObject


def _build_property_classifiers(domain_module, domain: str, attribute_classifier: str, device: str | None) -> dict:
    """One classifier instance per property, built once per run (not per object --
    reloading a CLIP checkpoint per crop would be prohibitively slow). Properties
    come from domain_module.PROPERTIES, never hardcoded, so this generalizes to
    whatever properties a domain defines -- except "size", deliberately excluded:
    neither shown, classified, nor saved (see _classify_object_properties)."""
    if domain == "clevr":
        properties = ["shape", "color", "material"]
    else:  # coco: category comes from the detector label itself, only color is classified
        properties = ["color"]
    return {
        prop: build_attribute_classifier(attribute_classifier, prop, domain_module.PROPERTIES[prop], device)
        for prop in properties
    }


def _classify_object_properties(domain: str, crop, classifiers: dict) -> dict[str, str]:
    """Classification order: shape first (CLEVR only, color's prompt is
    shape-conditioned, see attributes/README.md), then color, then the rest
    independently. "size" is deliberately never classified or included here --
    it's excluded from both automated- and human-perception, per user request."""
    properties: dict[str, str] = {}
    if domain == "clevr":
        predicted_shape, _ = classifiers["shape"].classify(crop)
        properties["shape"] = predicted_shape
        predicted_color, _ = classifiers["color"].classify(crop, context=predicted_shape)
        properties["color"] = predicted_color
        predicted_material, _ = classifiers["material"].classify(crop)
        properties["material"] = predicted_material
    else:
        predicted_color, _ = classifiers["color"].classify(crop)
        properties["color"] = predicted_color
    return properties


def _perceive_scene(image, domain: str, domain_module, detector, classifiers: dict) -> list[DetectedObject]:
    class_prompts = domain_module.PROPERTIES["shape"]  # CLEVR: cube/sphere/cylinder; COCO: bicycle/suitcase/chair
    boxes = detector.detect(image, class_prompts)

    objects: list[DetectedObject] = []
    for obj_id, bbox in enumerate(boxes):
        crop = crop_and_neutralize(image, bbox)
        properties = _classify_object_properties(domain, crop, classifiers)
        if domain == "coco":
            properties["shape"] = bbox.label  # detector's matched class prompt *is* the category
        cx, cy = bbox_center(bbox)
        region = region_of(cx, cy, image.width, image.height)
        objects.append(DetectedObject(obj_id=obj_id, bbox=bbox, properties=properties, region=region))
    return objects


def _load_prior_results(out_path: Path, detector_name: str, attribute_classifier: str) -> dict[tuple[str, str], dict]:
    """Resume support: reuse every *successful* result already sitting in out_path from
    an earlier (possibly interrupted) run, keyed by (id, prompt_field), instead of
    redoing the detector + classifier work for it. A "success": false entry isn't a
    completed result, so it's retried, not reused. If the prior run used a different
    detector/attribute_classifier, none of it is reused -- those results aren't
    comparable to what this run would produce, so silently mixing them in would be
    wrong, not just suboptimal. A missing or corrupt out_path (e.g. from a hard kill
    mid-write) just means starting fresh, not an error."""
    if not out_path.is_file():
        return {}
    try:
        with out_path.open("r", encoding="utf-8") as f:
            prior = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if prior.get("detector") != detector_name or prior.get("attribute_classifier") != attribute_classifier:
        print(
            f"[resume] out_path was written with detector={prior.get('detector')!r}/"
            f"attribute_classifier={prior.get('attribute_classifier')!r}, this run uses "
            f"{detector_name!r}/{attribute_classifier!r} -- not reusing any of it"
        )
        return {}
    return {
        (r["id"], r["prompt_field"]): r
        for r in prior.get("results", [])
        if r.get("success") and r.get("scene_graph") is not None
    }


def run_perception(
    items: list[MatchedItem],
    domain: str,
    detector_name: str,
    attribute_classifier: str,
    device: str | None,
    out_path: str | Path,
    on_item_done=None,  # optional callback(index_done: int, total: int, item_id: str) -- e.g. for
    # a caller driving a progress bar. This whole function used to be a black box until every
    # item finished; ccig-human-evaluation's "Run perception" button looked hung for long CPU
    # runs because of that, so progress is now reported incrementally instead of only at the end.
) -> None:
    from PIL import Image

    out_path = Path(out_path)
    domain_module = load_domain(domain)
    already_done = _load_prior_results(out_path, detector_name, attribute_classifier)

    # Detector + classifiers are real model weights, expensive to load -- built lazily,
    # only once at least one item actually needs (re)processing. A full resume where
    # every item is already done should cost nothing, not reload every model just to
    # immediately skip everything.
    detector = None
    classifiers = None

    def _ensure_models() -> None:
        nonlocal detector, classifiers
        if detector is None:
            detector = build_detector(detector_name, device=device)
            classifiers = _build_property_classifiers(domain_module, domain, attribute_classifier, device)

    results: list[dict] = []
    for item in items:
        key = (item.id, item.prompt_field)
        if key in already_done:
            results.append(already_done[key])
            print(f"[skip] {item.id}: already processed, reusing saved result")
        else:
            try:
                _ensure_models()
                image = Image.open(item.image_path).convert("RGB")
                objects = _perceive_scene(image, domain, domain_module, detector, classifiers)

                results.append(
                    PerceptionResult(
                        id=item.id,
                        prompt_field=item.prompt_field,
                        image_path=str(item.image_path),
                        prompt=item.prompt_text,
                        instantiated_rule=item.record.instantiated_rule,
                        status=item.record.status,
                        number_of_objects=len(objects),
                        scene_graph=to_graph_dict(objects),
                        success=True,
                        error=None,
                    ).to_json()
                )
                print(f"[ok]   {item.id}: {len(objects)} object(s) detected")
            except Exception as e:  # noqa: BLE001 -- one bad image must not abort the whole batch
                results.append(
                    PerceptionResult(
                        id=item.id,
                        prompt_field=item.prompt_field,
                        image_path=str(item.image_path),
                        prompt=item.prompt_text,
                        instantiated_rule=item.record.instantiated_rule,
                        status=item.record.status,
                        number_of_objects=None,
                        scene_graph=None,
                        success=False,
                        error=repr(e),
                    ).to_json()
                )
                print(f"[fail] {item.id}: {e}")

        # Write after every item, not just at the end -- so a long CPU run that's killed or
        # crashes partway through still leaves a partial, valid out_path behind, and so a
        # caller polling out_path's mtime (or using on_item_done) sees real progress instead
        # of nothing until the very last item.
        write_json(
            out_path,
            {
                "method": "perception",
                "domain": domain,
                "detector": detector_name,
                "attribute_classifier": attribute_classifier,
                "results": results,
            },
        )
        if on_item_done is not None:
            on_item_done(len(results), len(items), item.id)

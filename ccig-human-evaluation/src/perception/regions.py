from __future__ import annotations

from .detectors.base import BBox

# Transcribed from ccig-dataset-gen/src/eval_dataset_gen/asp_background/background.lp
# (not imported -- it's a .lp file, not a Python symbol). This is the same fixed 2x2
# grid adjacency ccig-dataset-gen's own clingo-based pipeline uses to derive
# left/right/front/behind between regions; kept as a static Python lookup here since
# it's small and this pipeline never calls clingo at all (see common/dataset_gen.py).
_RIGHT_OF: dict[str, set[str]] = {
    "r0": {"r1", "r3"},  # r1, r3 are right of r0
    "r2": {"r1", "r3"},  # r1, r3 are right of r2
}
_FRONT_OF: dict[str, set[str]] = {
    "r0": {"r2", "r3"},  # r2, r3 are in front of r0
    "r1": {"r2", "r3"},  # r2, r3 are in front of r1
}


def bbox_center(bbox: BBox) -> tuple[float, float]:
    return (bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2


def region_of(cx: float, cy: float, image_w: int, image_h: int) -> str:
    """2x2 grid: r0=top-left, r1=top-right, r2=bottom-left, r3=bottom-right,
    matching REGION_LAYOUT in ccig-dataset-gen's domain_clevr.py/domain_coco.py."""
    left = cx < image_w / 2
    top = cy < image_h / 2
    if top and left:
        return "r0"
    if top and not left:
        return "r1"
    if not top and left:
        return "r2"
    return "r3"


def pairwise_relations(regions: dict[int, str]) -> list[tuple[int, int, str]]:
    """regions: {obj_id: region}. Returns (from_id, to_id, direction) for every
    ordered pair whose regions are related by the background.lp adjacency table."""
    relations: list[tuple[int, int, str]] = []
    for id_a, region_a in regions.items():
        for id_b, region_b in regions.items():
            if id_a == id_b:
                continue
            if region_b in _RIGHT_OF.get(region_a, ()):
                # region_b is right of region_a => object b is right of object a
                relations.append((id_b, id_a, "right"))
                relations.append((id_a, id_b, "left"))
            if region_b in _FRONT_OF.get(region_a, ()):
                relations.append((id_b, id_a, "front"))
                relations.append((id_a, id_b, "behind"))
    return relations

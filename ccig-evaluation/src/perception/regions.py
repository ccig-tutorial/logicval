from __future__ import annotations

from .detectors.base import BBox


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

def pairwise_relations(objects: list[DetectedObject]) -> list[tuple[int, int, str]]:
    """objects: from bb. Returns (from_id, to_id, direction) for every
    ordered pair whose regions are related by the background.lp adjacency table."""
    relations: list[tuple[int, int, str]] = []
    for obj1 in objects:
        id_a = obj1.obj_id
        c1_x, c1_y = bbox_center(obj1.bbox)
        for obj2 in objects:
            id_b = obj2.obj_id
            if id_a == id_b:
                continue
            c2_x, c2_y = bbox_center(obj2.bbox)
            if c1_x<c2_x:
                relations.append((id_a, id_b, "left"))
                relations.append((id_b, id_a, "right"))
            if c1_y<c2_y:
                relations.append((id_a, id_b, "behind"))
                relations.append((id_b, id_a, "front"))
    return relations
    



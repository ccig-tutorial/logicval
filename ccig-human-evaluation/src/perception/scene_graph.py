from __future__ import annotations

from .regions import pairwise_relations
from .types import DetectedObject


def to_graph_dict(objects: list[DetectedObject]) -> dict:
    """Human/JSON-friendly scene graph for the output file. ccig-human-evaluation
    deliberately never builds the ASP-fact-atom form ccig-evaluation's own perception
    pipeline uses to feed clingo -- this pipeline never runs the constraint check."""
    regions = {obj.obj_id: obj.region for obj in objects}
    return {
        "objects": {
            str(obj.obj_id): {
                **obj.properties,
                "region": obj.region,
                "bbox": [obj.bbox.x0, obj.bbox.y0, obj.bbox.x1, obj.bbox.y1],
                "det_score": obj.bbox.score,
            }
            for obj in objects
        },
        "relations": [
            {"from": id_a, "to": id_b, "direction": direction}
            for id_a, id_b, direction in pairwise_relations(regions)
        ],
    }

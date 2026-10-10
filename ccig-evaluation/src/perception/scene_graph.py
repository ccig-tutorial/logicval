from __future__ import annotations

from .regions import pairwise_relations
from .types import DetectedObject


def build_scene_facts(objects: list[DetectedObject]) -> str:
    """Emit ASP facts in the exact atom shapes ccig-dataset-gen's format_scene()
    parses (object/1, hasProperty/3, hasRelationship/3), so the same clingo program
    shape works for both the original symbolic scenes and these perceived ones."""
    lines = [f"object({obj.obj_id})." for obj in objects]
    for obj in objects:
        for prop, val in obj.properties.items():
            lines.append(f"hasProperty({obj.obj_id},{prop},{val}).")
        lines.append(f"hasProperty({obj.obj_id},region,{obj.region}).")

    regions = {obj.obj_id: obj.region for obj in objects}
    for id_a, id_b, direction in pairwise_relations(objects):
        lines.append(f"hasRelationship({id_a},{id_b},{direction}).")

    return "\n".join(lines)




def to_graph_dict(objects: list[DetectedObject]) -> dict:
    """Human/JSON-friendly scene graph for the output file (separate from the ASP
    fact string, which clingo consumes)."""
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
            for id_a, id_b, direction in pairwise_relations(objects)
        ],
    }

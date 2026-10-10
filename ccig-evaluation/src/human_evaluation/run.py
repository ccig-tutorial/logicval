from __future__ import annotations

from pathlib import Path

from ..common.dataset_gen import load_domain, solve
from ..common.io import write_json
from ..common.types import MatchedItem, HumanResult
import json
import numpy as np

def bbox_center(bbox:list) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

def get_ground_item(ground_items, item_id, item_prompt_field):
    for item in ground_items:
        if int(item.id) == int(item_id) and item.prompt_field.strip() == item_prompt_field.strip():
            return item

def pairwise_relations(objects: dict) -> list[tuple[int, int, str]]:
    """objects: from bb. Returns (from_id, to_id, direction) for every
    ordered pair whose regions are related by the background.lp adjacency table."""
    relations: list[tuple[int, int, str]] = []
    for obj1 in objects:
        props1 = objects[obj1]
        id_a = obj1
        c1_x, c1_y = bbox_center(props1['bbox'])
        for obj2 in objects:
            props2 = objects[obj2]
            id_b = obj2
            if id_a == id_b:
                continue
            c2_x, c2_y = bbox_center(props2['bbox'])
            if c1_x<c2_x:
                relations.append((id_a, id_b, "left"))
                relations.append((id_b, id_a, "right"))
            if c1_y<c2_y:
                relations.append((id_a, id_b, "behind"))
                relations.append((id_b, id_a, "front"))
    return relations
    
def build_scene_facts_human(objects):
    lines = [f"object({obj})." for obj in objects]
    for obj in objects:
        props = objects[obj]
        lines.append(f"hasProperty({obj},color,{props['color']}).")
        lines.append(f"hasProperty({obj},shape,{props['shape']}).")
        lines.append(f"hasProperty({obj},material,{props['material']}).")
        lines.append(f"hasProperty({obj},region,{props['region']}).")
    
    relations = pairwise_relations(objects)
    for id_a, id_b, direction in pairwise_relations(objects):
        lines.append(f"hasRelationship({id_a},{id_b},{direction}).")
    return "\n".join(lines)


def run_human_eval(
    ground_items: list[MatchedItem],
    domain: str,
    annotation_file: str | Path,
    out_path: str | Path,
    manifest:str|Path, 
) -> None:
    from PIL import Image

    domain_module = load_domain(domain)
    
    results: list[HumanResult] = []
    with open(manifest, "r") as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    #items with no image generated
    for item in manifest:
        if item["error"] is not None:
            print('No image generated:', item['id'])
            results.append(
                    HumanResult(
                    id=item['id'],
                    prompt_field=item['prompt_field'],
                    image_path=None,
                    instantiated_rule=None,
                    dataset_status = None,
                    predicted_status=None,
                    score = 0,
                    objects = None,
                    pred_number_of_objects = None,
                    actual_number_of_objects = None, 
                    scene_graph=None,
                    clingo_program=None,
                    success=False,
                    error='No image generated',
                ))

    #items with images generated
    with open(annotation_file, "r") as f:
        file_annotations = json.load(f)
    annotations = file_annotations['annotations']
    for item in annotations:
            g_item = get_ground_item(ground_items, item['id'], item['prompt_field'])
            print('Processing:', item['id'], flush=True)
            if item['number_of_objects'] == 0: 
                print('white image generated:', g_item.id)
                if g_item.record.status == 'SAT':
                    results.append(
                    HumanResult(
                    id=item['id'],
                    prompt_field=item['prompt_field'],
                    image_path=str(g_item.image_path),
                    instantiated_rule=g_item.record.instantiated_rule,
                    dataset_status = g_item.record.status,
                    predicted_status='UNSAT',
                    score = 0,
                    objects = None,
                    pred_number_of_objects = 0,
                    actual_number_of_objects = g_item.record.number_of_objects,
                    scene_graph=None,
                    clingo_program=None,
                    success=True,
                    error=None,
                ))
                else:#UNSAT
                    results.append(
                    HumanResult(
                    id=item['id'],
                    prompt_field=item['prompt_field'],
                    image_path=str(g_item.image_path),
                    instantiated_rule=g_item.record.instantiated_rule,
                    dataset_status = g_item.record.status,
                    predicted_status='UNSAT',
                    score = 1,
                    objects = None,
                    pred_number_of_objects = 0,
                    actual_number_of_objects = 0,
                    scene_graph=None,
                    clingo_program=None,
                    success=True,
                    error=None,
                ))
                continue
            
            objects = item['scene_graph']['objects']
            
            
            facts = build_scene_facts_human(objects)
            # instantiated_rule uses free ASP variables (X, Y, ...) bound over object(X),
            # not literal object ids -- it applies unchanged no matter how perception
            # numbered the detected objects here.
            program = f"{facts}\n" + "\n".join(g_item.record.instantiated_rule)

            predicted_status, _ = solve(program, n_models=1, time_limit=10)
        
            
            if g_item.record.status == 'SAT':
                if predicted_status == g_item.record.status:
                    if item['number_of_objects'] == g_item.record.number_of_objects: 
                        score = 1
                    else:
                        score = 0
                else:
                    score = 0
            else:
                score = 0
           
            
            results.append(
                HumanResult(
                    id=item['id'],
                    prompt_field=item['prompt_field'],
                    image_path=str(g_item.image_path),
                    instantiated_rule=g_item.record.instantiated_rule,
                    dataset_status = g_item.record.status,
                    predicted_status=predicted_status,
                    score = score,
                    objects = objects,
                    pred_number_of_objects = item['number_of_objects'],
                    actual_number_of_objects = g_item.record.number_of_objects,
                    scene_graph=item['scene_graph'],
                    clingo_program=program,
                    success=True,
                    error=None,
                )
            )
               
    write_json(
        out_path,
        {
            "method": "human_eval",
            "domain": domain,
            "results": [r.to_json() for r in results],
        },
    )

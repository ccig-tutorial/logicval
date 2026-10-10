from __future__ import annotations

from pathlib import Path
import json, argparse, os
from ..common.dataset_gen import load_domain, solve
from ..common.io import write_json
from ..common.types import MatchedItem, PerceptionResult
from .attributes.registry import build_attribute_classifier
from .crop import crop_and_neutralize, make_object_views
from .detectors.registry import build_detector
from .regions import bbox_center, region_of
from .scene_graph import build_scene_facts, to_graph_dict
from .types import DetectedObject
from .regions import bbox_center, region_of
import numpy as np

#for hungarian matchingype=float)
from scipy.optimize import linear_sum_assignment


def get_pred_center(det_obj):
    """Get (x, y) center from a DetectedObject bounding box."""
    bbox = det_obj.bbox

    x = (bbox.x0 + bbox.x1) / 2.0
    y = (bbox.y0 + bbox.y1) / 2.0

    return np.array([x, y], dtype=float)


def get_gt_position(gt_obj):
    """Get (x, y) position from a ground-truth object."""
    return np.array([gt_obj["x"], gt_obj["y"]], dtype=float)


def hungarian_object_matching(pred_objects, gt_objects):
    """
    Match predicted objects to ground-truth objects using their (x, y)
    Euclidean distance.

    Matching is performed only if the number of predicted and GT objects
    is identical.

    Args:
        pred_objects: list of DetectedObject objects.
        gt_objects: dict of the form:
            {
                "o_0": {"x": 10.6, "y": 90, ...},
                "o_1": {"x": 50.2, "y": 30, ...},
                ...
            }

    Returns:
        pred_to_gt: dict mapping predicted obj_id -> ground-truth object id.

        Example:
            {
                0: "o_2",
                1: "o_0",
                2: "o_1"
            }

        Also returns the similarity/distance matrix and total matching cost.
    """
    n = len(pred_objects)

    
    # Preserve GT IDs
    gt_ids = list(gt_objects.keys())

    # ---------------------------------------------------------
    # 2. Extract positions
    # ---------------------------------------------------------
    pred_positions = np.array([
        get_pred_center(obj)
        for obj in pred_objects
    ])

    gt_positions = np.array([
        get_gt_position(gt_objects[obj_id])
        for obj_id in gt_ids
    ])

    # ---------------------------------------------------------
    # 3. Create pairwise distance matrix
    #
    # distance_matrix[i, j] =
    #     distance between predicted object i
    #     and GT object j
    # ---------------------------------------------------------
    distance_matrix = np.linalg.norm(
        pred_positions[:, None, :] - gt_positions[None, :, :],
        axis=2
    )

    # ---------------------------------------------------------
    # 4. Hungarian matching
    # ---------------------------------------------------------
    pred_indices, gt_indices = linear_sum_assignment(distance_matrix)

    # ---------------------------------------------------------
    # 5. Build pred_id -> GT_id dictionary
    # ---------------------------------------------------------
    pred_to_gt = {}

    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        pred_id = pred_objects[pred_idx].obj_id
        gt_id = gt_ids[gt_idx]

        pred_to_gt[pred_id] = gt_id

    # Total cost of the optimal matching
    total_cost = distance_matrix[pred_indices, gt_indices].sum()

    return pred_to_gt #, distance_matrix, total_cost

def reconstruct_scene(image, scene):
    #return a dict:{
    #  "objects": {"o_0": {"color": "blue", "size": "large", "material": "rubber",
    #                       "shape": "cube", "region": "r1"}, ...},
    #  "relations": [{"from": "o_0", "to": "o_2", "direction": "right"}, ...],
    #}
    reconstructed_scene = {}
    objects_scene = scene['objects']
    num= len(objects_scene)
    relationships_scene = scene['relationships']
    objects = {}
    relationships = []
    for idx, o in enumerate(objects_scene):
        pos = o['pixel_coords']
        x=  pos[0]
        y = pos[1]
        region = region_of(x, y, image.width, image.height)
        objects['o_'+str(idx)] = {"color":o["color"], "size":o["size"], "material": o["material"] , "shape": o["shape"], "region": region, "x":x, "y": y }
        
    for idx, rr in enumerate(relationships_scene['right']): ## rr is the list of objects to the right of idx
        for r, o in enumerate(rr):
            relation ={}
            relation['from'] = 'o_'+str(o)
            relation['to'] = 'o_'+str(idx)
            relation['direction'] = 'right'
            relationships.append(relation)

            relation_inv ={}
            relation_inv['from'] = 'o_'+str(idx)
            relation_inv['to'] = 'o_'+str(o)
            relation_inv['direction'] = 'left'
            relationships.append(relation_inv)
    
    for idx, rr in enumerate(relationships_scene['front']): ## rr is the list of objects to the front of idx
        for r, o in enumerate(rr):
            relation ={}
            relation['from'] = 'o_'+str(o)
            relation['to'] = 'o_'+str(idx)
            relation['direction'] = 'front'
            relationships.append(relation)

            relation_inv ={}
            relation_inv['from'] = 'o_'+str(idx)
            relation_inv['to'] = 'o_'+str(o)
            relation_inv['direction'] = 'behind'
            relationships.append(relation_inv)
    return {"objects": objects, "relations": relationships}


def _build_property_classifiers(domain_module, domain: str, attribute_classifier: str, device: str | None) -> dict:
    """One classifier instance per property, built once per run (not per object --
    reloading a CLIP checkpoint per crop would be prohibitively slow). Properties
    come from domain_module.PROPERTIES, never hardcoded, so this generalizes to
    whatever properties a domain defines (e.g. CLEVR's "size", which some
    constraints reference despite the sibling pipeline's ASP generator excluding it
    from its own choice rules)."""
    if domain == "clevr":
        properties = ["shape", "color", "material", "size"]
    else:  # coco: category comes from the detector label itself, only color is classified
        properties = ["color"]
    return {
        prop: build_attribute_classifier(attribute_classifier, prop, domain_module.PROPERTIES[prop], device)
        for prop in properties
    }

from PIL import Image


def _classify_object_properties(
    domain: str,
    views: dict[str, Image.Image],
    classifiers: dict,
) -> dict[str, str]:
    """
    Classification order for CLEVR:
      1. shape
      2. color conditioned on predicted shape
      3. material
      4. size

    Different attributes use different image views.
    """

    properties: dict[str, str] = {}

    if domain == "clevr":

        # Shape: emphasize geometry with tighter views.
        predicted_shape, _ = classifiers["shape"].classify_ensemble(
            [
                views["tight"],
                views["medium"],
            ]
        )
        properties["shape"] = predicted_shape

        # Color: use multiple views, conditioned on predicted shape.
        # Example competing prompts:
        #   "a red cube"
        #   "a blue cube"
        #   "a green cube"
        #   ...
        predicted_color, _ = classifiers["color"].classify_ensemble(
            [
                views["tight"],
                views["medium"],
                views["wide"],
            ]
            context=predicted_shape,
        )
        properties["color"] = predicted_color

        # Material: preserve RGB information because lighting,
        # highlights and reflections can be useful.
        predicted_material, _ = classifiers["material"].classify_ensemble(
            [
                views["rgb"],
                #views["medium"],
            ]
        )
        properties["material"] = predicted_material

        # Size: use medium/wide views.
        predicted_size, _ = classifiers["size"].classify_ensemble(
            [
                views["medium"],
                views["wide"],
            ]
        )
        properties["size"] = predicted_size

    else:
        # Non-CLEVR: only color is classified here.
        predicted_color, _ = classifiers["color"].classify_ensemble(
            [
                views["tight"],
                views["medium"],
                views["wide"],
            ]
        )
        properties["color"] = predicted_color

    return properties

'''
def _classify_object_properties(domain: str, crop, views: dict[str, Image.Image], classifiers: dict) -> dict[str, str]:
    """Classification order: shape first (CLEVR only, color's prompt is
    shape-conditioned, see attributes/README.md), then color, then the rest
    independently."""
    properties: dict[str, str] = {}
    if domain == "clevr":
        predicted_shape, _ = classifiers["shape"].classify(crop)
        properties["shape"] = predicted_shape
        predicted_color, _ = classifiers["color"].classify(crop, context=predicted_shape)
        properties["color"] = predicted_color
        for prop in ("material", "size"):
            predicted, _ = classifiers[prop].classify(crop)
            properties[prop] = predicted
    else:
        predicted_color, _ = classifiers["color"].classify(crop)
        properties["color"] = predicted_color
    return properties
'''

def _perceive_scene(
    image,
    domain: str,
    domain_module,
    detector,
    classifiers: dict,
) -> list[DetectedObject]:

    
    class_prompts = domain_module.PROPERTIES["shape"]

    boxes = detector.detect(image, class_prompts)

    objects: list[DetectedObject] = []

    for obj_id, bbox in enumerate(boxes):

        # Generate multiple representations of the detected object.
        views = make_object_views(image, bbox)

        properties = _classify_object_properties(
            domain,
            views,
            classifiers,
        )

        if domain == "coco":
            # Detector's matched class prompt is the category.
            properties["shape"] = bbox.label

        cx, cy = bbox_center(bbox)
        region = region_of(
            cx,
            cy,
            image.width,
            image.height,
        )

        objects.append(
            DetectedObject(
                obj_id=obj_id,
                bbox=bbox,
                properties=properties,
                region=region,
            )
        )

    return objects

'''    
def _perceive_scene(image, domain: str, domain_module, detector, classifiers: dict) -> list[DetectedObject]:
    class_prompts = domain_module.PROPERTIES["shape"]  # CLEVR: cube/sphere/cylinder; COCO: bicycle/suitcase/chair
    #print('Class prompts:', class_prompts)
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
'''
def find_closest_match(bbdet, objects):
    if not objects:
        return None

    return min(
        objects,
        key=lambda o: (objects[o]["x"] - bbdet[0]) ** 2
                   + (objects[o]["y"] - bbdet[1]) ** 2
    )
'''
def find_closest_match(bbdet, objects):
    bbdet_array = np.array(bbdet)
    Objid_o = {}
    pixel_coords = []
    o_id = 0 
    for o in objects:
        print('O:', o)
        pixel_coord = []
        pixel_coord.append(o['x'])
        pixel_coord.append(o['y'])
        Objid_o[o_id] = o
        pixel_coords.append(pixel_coord)
        o_id = o_id+1
    # convert to numpy array for convenience
    pixel_coords_array = np.array(pixel_coords)
    # compute Euclidean distances
    distances = np.linalg.norm(pixel_coords_array - bbdet_array, axis=1)
    # find index of closest pixel
    closest_idx = np.argmin(distances)
    # get the closest object coords
    closest_pixel = pixel_coords[closest_idx]
    #get matching object
    match_o = Objid_o[closest_idx]
    return match_o
'''

def process_prediction(objects, scene, neq, acc_scene, acc, total_objects):
    from scipy.optimize import linear_sum_assignment
    #return a dict:{
    #  "objects": {"o_0": {"color": "blue", "size": "large", "material": "rubber",
    #                       "shape": "cube", "region": "r1"}, ...},
    #  "relations": [{"from": "o_0", "to": "o_2", "direction": "right"}, ...],
    #}
    pred_objects = {'objects':{}, 'relations': []}
    if len(objects) != len(scene['objects']): 
        neq = neq+1
        return neq, acc_scene, acc, total_objects
    flag = True
    pairs = hungarian_object_matching(objects, scene['objects'])
    #print('Pairs:', pairs)        
    for do in objects:
        o_id = do.obj_id
        #bb = do.bbox 
        #(cx, cy) = bbox_center(bb) 
        o_id_pred = pairs[o_id]#find_closest_match([cx, cy], scene['objects'])
        #print('Closest match:', o_id_pred)
        #print('Obj pred:', o_id, cx,cy, scene['objects'][o_id_pred]['x'], scene['objects'][o_id_pred]['y'])
        
        prop_pred = do.properties
        prop_gnd = scene['objects'][o_id_pred]
        #print('Predicted prop:', prop_pred)
        #print('Grounf prop:', prop_gnd)
        total_objects = total_objects + 1
        for prop in prop_pred:
            if prop_pred[prop] == prop_gnd[prop]:
                acc[prop] = acc[prop] + 1
            else:
                if prop!='size':
                    flag = False
        if do.region == prop_gnd['region']:
            acc['region'] = acc['region'] + 1
        else:
            flag = False
    if flag:
        acc_scene = acc_scene+1
    return neq, acc_scene, acc, total_objects

def run_perception_acc(
    domain:str,
    detector_name: str,
    attribute_classifier: str,
    device: str | None,
) -> None:
    
    from PIL import Image

    domain_module = load_domain(domain)
    #print(domain_module)
    detector = build_detector(detector_name, device=device)
    classifiers = _build_property_classifiers(domain_module, domain, attribute_classifier, device)

    results: list[PerceptionResult] = []
    clevr_test_image = '/users/sbsh670/data/clevr/CLEVR_v1.0/image_generative_model_training/images/val'
    clevr_val_scenes = '/users/sbsh670/data/clevr/CLEVR_v1.0/scenes/CLEVR_val_scenes.json'
    acc ={'shape':0, 'color':0, 'size': 0, 'material':0, 'region':0}
    acc_scene = 0
    neq = 0
    total = 0
    total_objects = 0
    with open(clevr_val_scenes, 'r') as f:
        data = json.load(f)
    scene_records = data['scenes']
    all_predictions = {}
 
    for rec in scene_records:
            image_id = rec["image_index"]
            image_filename = rec["image_filename"]
            image_path = os.path.join(clevr_test_image, image_filename)
            image = Image.open(image_path).convert("RGB")
            objects = _perceive_scene(image, domain, domain_module, detector, classifiers)
            #print('Predicted objects:', objects)
            all_predictions[str(image_id)] = {
                    "image_id": image_id,
                    "image_filename": image_filename,
                    "prediction": objects,
                    }

            #print('Prediction: Objects:', objects)
            
            scene = reconstruct_scene(image, rec)
            #print('Ground scene:', scene)
            neq, acc_scene, acc, total_objects = process_prediction(objects, scene, neq, acc_scene, acc, total_objects)
            total = total +1
            if(total>=100):
                break
    
    
    print('Neq:', neq)
    print('Acc obj wise:', acc, total_objects)
    print('Acc scene wise:', acc_scene, total)

   


def main() -> None:
    parser = argparse.ArgumentParser(description="Run perception accuracy on clevr-val-images.")
    parser.add_argument("--domain", default="clevr", choices=["clevr", "coco"])
    parser.add_argument("--detector", default="grounding-dino", choices=["grounding-dino", "owlv2"])
    parser.add_argument("--attribute-classifier", default="clip-zero-shot")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    run_perception_acc(args.domain, args.detector, args.attribute_classifier, args.device)
    


if __name__ == "__main__":
    main()
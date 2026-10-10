"""
Scene-query primitives over a reconstructed {"objects": {...}, "relations": [...]} scene
(see scene_reconstruct.py). Every C1-C9 grounder in grounders.py composes these instead of
re-implementing scene lookups, mirroring how verbalize.py's phrase helpers (_adj, _obj, _quant,
...) compose into every constraint class.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..common import domain_clevr as domain

Scene = Dict[str, object]  # {"objects": {obj_id: {prop: val}}, "relations": [{"from","to","direction"}]}

#list of obj ids
def object_ids(scene: Scene) -> List[str]:
    return list(scene["objects"].keys())  # type: ignore[index]

#value of a certain prop for a certain oid
def object_value(scene: Scene, obj_id: str, prop: str) -> str:
    return scene["objects"][obj_id][prop]  # type: ignore[index]

#all prop_value pairs in the scene de-duplicated
def present_prop_values(scene: Scene) -> List[Tuple[str, str]]:
    """All (prop, val) pairs actually held by at least one object, deduplicated."""
    seen = set()
    out = []
    for attrs in scene["objects"].values():  # type: ignore[union-attr]
        for prop, val in attrs.items():
            if (prop, val) not in seen:
                seen.add((prop, val))
                out.append((prop, val))
    return out

#return oids with prop=val
def objects_with(scene: Scene, prop: str, val: str) -> List[str]:
    return [oid for oid, attrs in scene["objects"].items() if attrs[prop] == val]  # type: ignore[union-attr]

#return oids where prop!=val
def objects_without(scene: Scene, prop: str, val: str) -> List[str]:
    return [oid for oid, attrs in scene["objects"].items() if attrs[prop] != val]  # type: ignore[union-attr]

#count no of ois with prop=val
def count_with(scene: Scene, prop: str, val: str) -> int:
    return len(objects_with(scene, prop, val))

#count no of ois with prop!=val
def count_without(scene: Scene, prop: str, val: str) -> int:
    return len(objects_without(scene, prop, val))

#check whether all objs have prop=val
def all_share(scene: Scene, prop: str, val: str) -> bool:
    ids = object_ids(scene)
    return bool(ids) and all(object_value(scene, oid, prop) == val for oid in ids)

#prop,val that is shared by all objects
def uniform_property(scene: Scene) -> Optional[Tuple[str, str]]:
    """A (prop, val) pair every object in the scene shares, if one exists."""
    for prop, val in present_prop_values(scene):
        if all_share(scene, prop, val):
            return prop, val
    return None

#what are the values held by prop in the scene other than val
def other_value(prop: str, val: str, exclude: Optional[str] = None) -> Optional[str]:
    """Any domain value for `prop` other than `val` (and `exclude`, if given)."""
    for candidate in domain.PROPERTIES[prop]:
        if candidate != val and candidate != exclude:
            return candidate
    return None


def object_matching_none_of(scene: Scene, prop: str, values: List[str]) -> Optional[str]:
    """An object whose `prop` value is none of `values` (witness for a negated-property constraint)."""
    for oid in object_ids(scene):
        if object_value(scene, oid, prop) not in values:
            return oid
    return None

#obj-id stands left of oid - list of oids to which obj_id stands left of or list o oids that are right of obj_id
#list of oids such that left(obj_id, oid) is true
def neighbors(scene: Scene, obj_id: str, direction: str) -> List[str]:
    """Objects that `obj_id` stands `direction` of, i.e. edges obj_id --direction--> target."""
    return [
        r["to"] for r in scene["relations"]  # type: ignore[union-attr]
        if r["from"] == obj_id and r["direction"] == direction
    ]


def related_pairs(scene: Scene, direction: str) -> List[Tuple[str, str]]:
    return [
        (r["from"], r["to"]) for r in scene["relations"]  # type: ignore[union-attr]
        if r["direction"] == direction
    ]


def find_relation_pairs(
    scene: Scene, prop1: str, val1: str, direction: str, prop2: str, val2: str
) -> List[Tuple[str, str]]:
    """Pairs (o1, o2), distinct, where o1 is (prop1=val1), o2 is (prop2=val2), o1 --direction--> o2."""
    out = []
    for o1, o2 in related_pairs(scene, direction):
        if o1 == o2:
            continue
        if object_value(scene, o1, prop1) == val1 and object_value(scene, o2, prop2) == val2:
            out.append((o1, o2))
    return out


def find_2hop_chains(
    scene: Scene,
    prop1: str, val1: str, direction1: str,
    prop2: str, val2: str, direction2: str,
    prop3: str, val3: str,
) -> List[Tuple[str, str, str]]:
    """Distinct triples (o1, o2, o3): o1 --dir1--> o2 --dir2--> o3, matching prop1/prop2/prop3."""
    out = []
    for o1, o2 in find_relation_pairs(scene, prop1, val1, direction1, prop2, val2):
        for o3 in neighbors(scene, o2, direction2):
            if o3 in (o1, o2):
                continue
            if object_value(scene, o3, prop3) == val3:
                out.append((o1, o2, o3))
    return out

#there exist subject_id for all x 
def universal_relation_holds(scene: Scene, subject_id: str, direction: str, prop2: str, val2: str) -> bool:
    """True if `subject_id` stands `direction` of *every* object matching prop2=val2 (excluding itself)."""
    targets = [oid for oid in objects_with(scene, prop2, val2) if oid != subject_id]
    if not targets:
        return False
    stands_to = set(neighbors(scene, subject_id, direction))
    return all(t in stands_to for t in targets)


def subset_matching(scene: Scene, conditions: List[Tuple[str, str]]) -> List[str]:
    """Objects matching every (prop, val) condition simultaneously."""
    ids = set(object_ids(scene))
    for prop, val in conditions:
        ids &= set(objects_with(scene, prop, val))
    return list(ids)


def uniform_value_in_subset(scene: Scene, ids: List[str], prop: str) -> Optional[str]:
    """The single prop value shared by every object in `ids`, if any (None if empty or mixed)."""
    if not ids:
        return None
    vals = {object_value(scene, oid, prop) for oid in ids}
    return next(iter(vals)) if len(vals) == 1 else None


def absent_value_in_subset(scene: Scene, ids: List[str], prop: str, exclude: Optional[str] = None) -> Optional[str]:
    """A domain value for `prop` that no object in `ids` holds (witness for a negated consequent)."""
    present = {object_value(scene, oid, prop) for oid in ids}
    for candidate in domain.PROPERTIES[prop]:
        if candidate not in present and candidate != exclude:
            return candidate
    return None

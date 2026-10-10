"""
CLEVR domain: property/value vocabulary shared by eval_dataset_gen and finetune_dataset_gen,
reflecting what's actually in the CLEVR-CCIG scenes: 8 colors, 3 shapes, 2 sizes, 2 materials,
4 quadrant regions (labeled "r0".."r3", matching the region atoms already baked into the
rendered scene data).

eval_dataset_gen deliberately excludes "material" from its ASP constraint search space (see
eval_dataset_gen/domain.py) -- that's a one-sided override on top of this domain, not a second
independent copy of it.

A future domain (e.g. COCO) would live as a sibling module, e.g. src/common/domain_coco.py,
exposing the same PROPERTIES/DIRECTIONS shape; run.py's --domain flag picks which one is active.
"""

from __future__ import annotations

PROPERTIES: dict[str, list[str]] = {
    "color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
    "shape": ["cube", "cylinder", "sphere"],
    "size": ["small", "large"],
    "material": ["rubber", "metal"],
    "region": ["r0", "r1", "r2", "r3"],
}

DIRECTIONS: list[str] = ["left", "right", "front", "behind"]
DIRECTIONS_INV: dict[str,str] = {"left": "right", "right":"left", "front":"behind", "behind":"front"}


# 2x2 grid layout, matching the region atoms baked into the rendered scene data
# and the layout comment in eval_dataset_gen/asp_background/background.lp.
REGION_LAYOUT: dict[str, str] = {
    "r0": "top-left",
    "r1": "top-right",
    "r2": "bottom-left",
    "r3": "bottom-right",
}

def system_prompt_text(with_background=False) -> str:
    colors = ", ".join(PROPERTIES["color"])
    shapes = ", ".join(PROPERTIES["shape"])
    sizes = ", ".join(PROPERTIES["size"])
    materials = ", ".join(PROPERTIES["material"])
    regions = "; ".join(f"{r} ({pos})" for r, pos in REGION_LAYOUT.items())
    if with_background:
        background = "with a white background, "
    else:
        background = ""
    return (
        f"Generate a CLEVR style scene {background}that is divided into 4 regions arranged in a 2x2 grid: {regions}. "
        f"Each object has a color (one of: {colors}), a shape (one of: {shapes}), a material (one of: {materials}), "
        f"and a size (one of: {sizes}), and is placed in one of the 4 regions. "
        f"The scene must additionally satisfy the following constraint: "
    )


def scene_setup_text(n_objects: int = 4, with_background: bool = False) -> str:
    return (f"{system_prompt_text(with_background=with_background)} The scene is with {n_objects} objects. ")

def scene_unsat_text(with_unsat: bool = False) -> str:
    if with_unsat:
        return (f" If the scene desciption is invalid or contains a contradiction, generate an empty scene with white background.")
    else:
        return ""
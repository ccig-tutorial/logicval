"""
COCO domain: symbolic vocabulary shared by eval_dataset_gen and
finetune_dataset_gen.

Unlike CLEVR, COCO does not define geometric shapes or materials.
Objects are represented by COCO object categories together with
attributes such as color and coarse image region.

The module exposes the same interface as other domains
(PROPERTIES, DIRECTIONS, REGION_LAYOUT, system_prompt_text),
allowing run.py's --domain flag to switch between domains.
"""

from __future__ import annotations

PROPERTIES: dict[str, list[str]] = {
    "color": ["red", "blue", "green", "brown", "purple", "cyan", "yellow"],
    "shape": ["bicycle", "suitcase", "chair"],
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
    regions = "; ".join(f"{r} ({pos})" for r, pos in REGION_LAYOUT.items())

    if with_background:
        background = "with a white background, "
    else:
        background = ""

    return (
        f"Generate a CLEVR style scene {background}that is divided into 4 regions arranged in a 2x2 grid: {regions}. "
        f"Each object belongs to one of the following categories ({shapes}), has a color (one of: {colors}), "
        f"and is placed in one of the 4 regions. "
        f"The scene must additionally satisfy the following constraint: "
    )


def scene_setup_text(n_objects: int = 4, with_background: bool = False) -> str:
    return (f"{system_prompt_text(with_background=with_background)} The scene is with {n_objects} objects. ")

def scene_unsat_text(with_unsat: bool = False) -> str:
    if with_unsat:
        return (f" If the scene desciption is invalid or contains a contradiction, generate an empty scene with white background.")
    else:
        return ""
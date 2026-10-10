from __future__ import annotations

from dataclasses import dataclass

from .detectors.base import BBox


@dataclass
class DetectedObject:
    obj_id: int  # 0..N-1, assignment order only -- ASP rules use free variables, not literal ids
    bbox: BBox
    properties: dict[str, str]  # e.g. {"color": "red", "shape": "cube", "material": "rubber"}
    region: str  # "r0".."r3"

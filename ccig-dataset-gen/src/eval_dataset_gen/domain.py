"""Eval-side property domain and ASP background loader for CCIG scene generation.

PROPERTIES derives from common/domain_clevr.py (single source of truth for what's actually in
the CCIG scenes) with one deliberate override: "material" is excluded from the ASP constraint
search space to keep eval-set generation tractable. This is the only place that exclusion is
encoded -- finetune_dataset_gen uses the full common/domain_clevr.py PROPERTIES, materials
included.
"""
import importlib
from pathlib import Path
from typing import Dict, List

#from ..common.domain_clevr import PROPERTIES as _ALL_PROPERTIES, DIRECTIONS as RELATIONS

def load_domain(domain: str):
    """Load domain-specific vocabulary."""
    module = importlib.import_module(f"..common.domain_{domain}", package=__package__)
    return module


COUNTS: List[str] = ["1", "2", "3"]

_BACKGROUND_LP = Path(__file__).parent / "asp_background" / "background.lp"


def background_asp(domain:str, n_objects: int = 4) -> str:
    """Return the full background ASP program for a scene with n_objects.

    Property facts and choice rules are generated from PROPERTIES (single source
    of truth). Structural rules (layout, relationships, axioms) come from background.lp.
    """
    
    domain_module = load_domain(domain)
    # Domain-specific vocabulary
    _ALL_PROPERTIES = domain_module.PROPERTIES
    
    PROPERTIES: Dict[str, List[str]] = {k: v for k, v in _ALL_PROPERTIES.items() if k != "size"}

    
    lines = [f"object({i})." for i in range(n_objects)]
    for prop, vals in PROPERTIES.items():
        for val in vals:
            lines.append(f"property({prop}, {val}).")
        lines.append("")

    for prop in PROPERTIES:
        lines.append(f"1 {{ hasProperty(X, {prop}, V) : property({prop}, V) }} 1 :- object(X).")
    lines.append("")

    with _BACKGROUND_LP.open("r", encoding="utf-8") as f:
        for bl in f:
            bl = bl.strip()
            # Skip comment lines and comments
            if not (bl.startswith("%")):
                lines.append(bl)

    return "\n".join(lines)

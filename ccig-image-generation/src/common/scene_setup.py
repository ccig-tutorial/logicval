"""Loads scene_setup_text from ccig-dataset-gen's domain_<domain>.py modules (the
single source of truth for each domain's vocabulary), without needing that
sibling project installed as a package -- both projects have their own
top-level `src` package, so a normal import would collide.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMMON_DIR = _REPO_ROOT / "ccig-dataset-gen" / "src" / "common"


@lru_cache(maxsize=None)
def _load_domain_module(domain: str):
    path = _COMMON_DIR / f"domain_{domain}.py"
    spec = importlib.util.spec_from_file_location(f"ccig_dataset_gen_domain_{domain}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scene_setup_text(n_objects: int, domain: str, with_background: bool = False) -> str:
    return _load_domain_module(domain).scene_setup_text(n_objects, with_background)


def scene_unsat_text(domain: str, with_unsat: bool = False) -> str:
    return _load_domain_module(domain).scene_unsat_text(with_unsat)
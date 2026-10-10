from __future__ import annotations

import sys
from pathlib import Path

# ccig-evaluation and ccig-dataset-gen are sibling pipelines in the same repo, run
# independently (`python -m src.run` from each one's own directory). This shim makes
# ccig-dataset-gen/src importable so the domain vocabulary (colors/shapes/regions/...)
# and the clingo solver are used live from their one source of truth, instead of being
# duplicated and risking drift.
_DATASET_GEN_SRC = Path(__file__).resolve().parents[3] / "ccig-dataset-gen" / "src"
if not _DATASET_GEN_SRC.is_dir():
    raise RuntimeError(
        f"Expected sibling pipeline at {_DATASET_GEN_SRC}, but it does not exist. "
        "ccig-evaluation depends on ccig-dataset-gen/src being checked out alongside it."
    )
if str(_DATASET_GEN_SRC) not in sys.path:
    sys.path.insert(0, str(_DATASET_GEN_SRC))

from common import domain_clevr, domain_coco  # noqa: E402
from eval_dataset_gen.solve import format_scene, solve  # noqa: E402

DOMAINS = {"clevr": domain_clevr, "coco": domain_coco}


def load_domain(domain: str):
    """domain: 'clevr' or 'coco' -> the matching domain_clevr/domain_coco module."""
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'. Available: {sorted(DOMAINS)}")
    return DOMAINS[domain]


__all__ = ["domain_clevr", "domain_coco", "DOMAINS", "load_domain", "solve", "format_scene"]

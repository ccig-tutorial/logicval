from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class PromptRecord:
    """One record from ccig_eval_dataset_{SAT,UNSAT}.jsonl, as produced by ccig-dataset-gen."""

    id: str
    domain: str
    complexity_class: str
    constraint_family: str
    prompts: dict[str, str]  # {"short": ..., "medium": ..., "long": ...}
    instantiated_rule: str
    asp_template_file: str
    status: str  # "SAT" | "UNSAT", the ground-truth label for the *original* symbolic scene
    number_of_objects: int


@dataclass
class MatchedItem:
    """A generated image joined to the prompt record that produced it."""

    id: str
    prompt_field: str  # "short" | "medium" | "long"
    image_path: Path
    record: PromptRecord

    @property
    def prompt_text(self) -> str:
        return self.record.prompts[self.prompt_field]


@dataclass
class ClipScoreResult:
    id: str
    prompt_field: str
    prompt: str
    image_path: str
    clipscore: float | None
    success: bool
    error: str | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeResult:
    id: str
    prompt_field: str
    prompt: str
    image_path: str
    score: float | None  # normalized to 0..1
    raw_score: float | int | None  # backend's native scale, for audit
    rationale: str | None
    success: bool
    error: str | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PerceptionResult:
    # Fields through number_of_objects deliberately use the same names as the ground-truth
    # PromptRecord above (id, prompt_field's text as "prompt", instantiated_rule, status,
    # number_of_objects) -- and are mirrored key-for-key in ccig-human-evaluation's
    # human-perception entries, so the two files are directly diffable field by field
    # without either one referring to the other's file path.
    #
    # No predicted_status/agrees_with_dataset/clingo_program here -- unlike
    # ccig-evaluation's own perception pipeline, this one deliberately never runs the
    # ASP/clingo constraint check; it only produces the detected scene_graph.
    id: str
    prompt_field: str
    image_path: str
    prompt: str
    instantiated_rule: str
    status: str  # ground-truth "SAT" | "UNSAT", same meaning/name as PromptRecord.status
    number_of_objects: int | None  # count of objects in scene_graph (bounding boxes found); None if success=False
    scene_graph: dict[str, Any] | None
    success: bool
    error: str | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

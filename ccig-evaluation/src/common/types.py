from __future__ import annotations



from dataclasses import asdict, dataclass, field
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
    # question -> expected answer, from verbalize()'s subqa (see ccig-dataset-gen/src/common/verbalize.py).
    # Defaults to {} for datasets generated before this field existed.
    subqa: dict[str, str] = field(default_factory=dict)


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
    clipmodel: str
    id: str
    prompt_field: str
    prompt: str
    image_path: str
    clipscore: float | None
    cosine: float | None
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
    id: str
    prompt_field: str
    image_path: str
    instantiated_rule: str
    dataset_status: str
    score:int
    objects: list|None
    scene_graph: dict[str, Any] | None
    clingo_program: str | None
    success: bool
    error: str | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SubQAScore:
    """One sub-question's soft-TIFA score. See soft_tifa/README.md for what
    'excluded_from_score' means (anchor questions that bind a symbol like 'n' or 'c'
    for later questions to reference, but have no checkable expected answer themselves)."""

    question: str
    expected_answer: str
    answer_type: str  # "yes_no" | "true_false" | "count_comparison" | "open_value"
    candidates: dict[str, float]  # candidate answer -> VQA model probability
    score: float | None  # soft score in [0, 1]; None if excluded_from_score
    excluded_from_score: bool

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SoftTifaResult:
    id: str
    prompt_field: str
    image_path: str
    score_am: float | None  # arithmetic mean of sub-question scores
    score_gm: float | None  # geometric mean of sub-question scores
    subquestions: list[SubQAScore]
    success: bool
    error: str | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class HumanResult:
    id: str
    prompt_field: str
    image_path: str
    instantiated_rule: str
    predicted_status:str
    dataset_status: str
    score:int
    objects: list|None
    pred_number_of_objects:int
    actual_number_of_objects:int
    scene_graph: dict[str, Any] | None
    clingo_program: str | None
    success: bool
    error: str | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)




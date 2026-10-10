from __future__ import annotations

from .base import VLMJudge
from .closed.gpt4o import GPT4oJudge
from .open.qwen2_vl import Qwen2VLJudge

JUDGE_REGISTRY: dict[str, type[VLMJudge]] = {
    "gpt-4o": GPT4oJudge,
    "qwen2-vl": Qwen2VLJudge,
}


def build_judge(name: str, device: str | None = None) -> VLMJudge:
    if name not in JUDGE_REGISTRY:
        raise ValueError(f"Unknown judge backend '{name}'. Available: {sorted(JUDGE_REGISTRY)}")
    return JUDGE_REGISTRY[name](device=device)

from __future__ import annotations

from .base import VQABackend
from .closed.gpt4o import GPT4oBackend
from .open.qwen2_vl import Qwen2VLBackend

VQA_REGISTRY: dict[str, type[VQABackend]] = {
    "gpt-4o": GPT4oBackend,
    "qwen2-vl": Qwen2VLBackend,
}


def build_vqa_backend(name: str, device: str | None = None) -> VQABackend:
    if name not in VQA_REGISTRY:
        raise ValueError(f"Unknown VQA backend '{name}'. Available: {sorted(VQA_REGISTRY)}")
    return VQA_REGISTRY[name](device=device)

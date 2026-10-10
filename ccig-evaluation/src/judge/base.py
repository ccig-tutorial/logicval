from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

# Every backend is asked the same rubric and expected to answer in this shape, so
# parsing lives once here rather than being reimplemented per backend.
JUDGE_PROMPT_TEMPLATE = (
    "You are judging whether a generated image matches a text description.\n"
    'Description: "{prompt}"\n'
    "Rate how well the image satisfies all the logical constraints in the description on a scale of 1 (not at all) to "
    "5 (perfect match). Respond in exactly this format:\n"
    "SCORE: <1-5>\n"
    "RATIONALE: <one sentence>"
)

_SCORE_RE = re.compile(r"SCORE:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_RATIONALE_RE = re.compile(r"RATIONALE:\s*(.+)", re.IGNORECASE | re.DOTALL)


@dataclass
class JudgeVerdict:
    score: float  # normalized to 0..1
    raw_score: float  # backend's native 1-5 scale, for audit
    rationale: str | None


def parse_judge_response(text: str) -> JudgeVerdict:
    """Parse the SCORE/RATIONALE response format all judge backends are prompted for."""
    score_match = _SCORE_RE.search(text)
    if not score_match:
        raise ValueError(f"Could not parse SCORE from judge response: {text!r}")
    raw_score = float(score_match.group(1))
    rationale_match = _RATIONALE_RE.search(text)
    rationale = rationale_match.group(1).strip() if rationale_match else None
    return JudgeVerdict(score=(raw_score - 1) / 4, raw_score=raw_score, rationale=rationale)


class VLMJudge(ABC):
    """A vision-language model that scores prompt-image alignment."""

    name: str

    @abstractmethod
    def judge(self, image: Image.Image, prompt: str) -> JudgeVerdict:
        raise NotImplementedError

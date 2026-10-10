from __future__ import annotations

import base64
import math
import os
from io import BytesIO

from PIL import Image

from ..base import VQABackend

_PROMPT_TEMPLATE = (
    "{question}\n"
    "Answer with exactly one word from this list, nothing else: {candidate_list}."
)


class GPT4oBackend(VQABackend):
    """Closed-API VQA backend for soft-TIFA scoring -- same OpenAI client pattern as
    judge/closed/gpt4o.py, but reads `logprobs`/`top_logprobs` on a single forced
    output token instead of parsing free text.

    The Chat Completions API doesn't expose full-vocabulary logits like a local model
    does -- `top_logprobs` (max 20) is the closest it gets, returning the N most
    likely tokens at that position with their log-probabilities. Any candidate that
    doesn't happen to be among those top 20 is scored 0.0 rather than left undefined;
    for the small forced-choice sets used here (2-3 for yes/no or True/False, up to
    ~10 digits, a handful of property values) a genuinely plausible candidate is
    essentially always inside the top 20, so this is a fair approximation of the
    local backend's exact softmax mass -- not a fundamentally different method.
    """

    name = "gpt-4o"

    def __init__(self, model: str = "gpt-4o", device: str | None = None) -> None:
        # device is accepted (and ignored) so callers can build any backend through
        # one uniform build_vqa_backend(name, device=...) signature.
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def answer_distribution(
        self, image: Image.Image, question: str, candidates: list[str]
    ) -> dict[str, float]:
        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt_text = _PROMPT_TEMPLATE.format(question=question, candidate_list=", ".join(candidates))
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
        )

        top = response.choices[0].logprobs.content[0].top_logprobs
        token_probs: dict[str, float] = {}
        for entry in top:
            token_probs[entry.token] = token_probs.get(entry.token, 0.0) + math.exp(entry.logprob)

        result: dict[str, float] = {}
        for candidate in candidates:
            variants = {candidate, " " + candidate, candidate.lower(), " " + candidate.lower()}
            result[candidate] = sum(token_probs.get(v, 0.0) for v in variants)
        return result

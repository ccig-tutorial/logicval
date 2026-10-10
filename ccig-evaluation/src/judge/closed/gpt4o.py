from __future__ import annotations

import base64
import os
from io import BytesIO

from PIL import Image

from ..base import VLMJudge, JudgeVerdict, JUDGE_PROMPT_TEMPLATE, parse_judge_response
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

class GPT4oJudge(VLMJudge):
    name = "gpt-4o"

    def __init__(self, model: str = "gpt-4o", device: str | None = None) -> None:
        # device is accepted (and ignored) so callers can build any judge through one
        # uniform build_judge(name, device=...) signature regardless of backend.
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def judge(self, image: Image.Image, prompt: str) -> JudgeVerdict:
        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": JUDGE_PROMPT_TEMPLATE.format(prompt=prompt)},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
        )
        return parse_judge_response(response.choices[0].message.content)

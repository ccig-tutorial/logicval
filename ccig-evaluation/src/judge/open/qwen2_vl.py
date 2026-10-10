from __future__ import annotations

from PIL import Image

from ..base import JUDGE_PROMPT_TEMPLATE, JudgeVerdict, VLMJudge, parse_judge_response


class Qwen2VLJudge(VLMJudge):
    """Local, open-weight judge backend -- no API cost, needs a GPU to be fast
    (runs on CPU too, just slow; see device resolution below)."""

    name = "qwen2-vl"
    hf_repo = "Qwen/Qwen2-VL-7B-Instruct"

    def __init__(self, device: str | None = None) -> None:
        # The one place GPU-vs-CPU is decided for this backend -- everything after
        # this line just uses self.device, no separate code paths.
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.hf_repo, torch_dtype=torch.bfloat16
        ).to(self.device)
        self._processor = AutoProcessor.from_pretrained(self.hf_repo)

    def judge(self, image: Image.Image, prompt: str) -> JudgeVerdict:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": JUDGE_PROMPT_TEMPLATE.format(prompt=prompt)},
                ],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=128)
        generated = generated[:, inputs["input_ids"].shape[1] :]
        response = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return parse_judge_response(response)

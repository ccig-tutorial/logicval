from __future__ import annotations

from PIL import Image

from ..base import VQABackend

_PROMPT_TEMPLATE = (
    "{question}\n"
    "Answer with exactly one word from this list, nothing else: {candidate_list}."
)


class Qwen2VLBackend(VQABackend):
    """Local, open-weight VQA backend for soft-TIFA scoring -- same model and loading
    pattern as judge/open/qwen2_vl.py (the VLM-judge backend), but here the model is
    forced to a single next token so its softmax gives a genuine per-candidate
    probability, instead of being read as free-text generation.
    """

    name = "qwen2-vl"
    hf_repo = "Qwen/Qwen2-VL-7B-Instruct"

    def __init__(self, device: str | None = None) -> None:
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self._model = (
            Qwen2VLForConditionalGeneration.from_pretrained(self.hf_repo, torch_dtype=torch.bfloat16)
            .to(self.device)
            .eval()
        )
        self._processor = AutoProcessor.from_pretrained(self.hf_repo)

    def _token_ids_for(self, candidate: str) -> set[int]:
        # A word can tokenize differently depending on whether it's preceded by a
        # space, and differently again by case -- collect every variant's *first*
        # token id (max_new_tokens=1 means only the first token is ever generated)
        # and sum their probability mass together as "this candidate's" probability.
        variants = {candidate, " " + candidate, candidate.lower(), " " + candidate.lower()}
        ids: set[int] = set()
        for v in variants:
            encoded = self._processor.tokenizer.encode(v, add_special_tokens=False)
            if encoded:
                ids.add(encoded[0])
        return ids

    def answer_distribution(
        self, image: Image.Image, question: str, candidates: list[str]
    ) -> dict[str, float]:
        import torch

        prompt_text = _PROMPT_TEMPLATE.format(question=question, candidate_list=", ".join(candidates))
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        logits = out.scores[0][0]  # the one generated token's logits, batch index 0
        probs = torch.softmax(logits.float(), dim=-1)

        return {
            candidate: sum(probs[t].item() for t in self._token_ids_for(candidate))
            for candidate in candidates
        }

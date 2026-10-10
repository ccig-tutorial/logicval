from __future__ import annotations

from PIL import Image

from .base import AttributeClassifier

# One noun-phrase template per property. "{label}" is the candidate value being
# scored; "{context}" is an already-known attribute of the same object (see
# base.AttributeClassifier.classify docstring) -- for properties with no natural
# context dependency the template simply ignores it.
_PROMPT_TEMPLATES: dict[str, str] = {
    "color": "a photo of a {label} {context}",
    "shape": "a photo of a {label}",
    "material": "a photo of a {label} object",
    "size": "a photo of a {label} object",
}
_DEFAULT_TEMPLATE = "a photo of a {label} {context}"
_DEFAULT_CONTEXT_NOUN = "object"  # used when context is None but the template needs one


class ClipZeroShotAttribute(AttributeClassifier):
    """Zero-shot attribute classification via CLIP: score the crop against
    "a photo of a [LABEL] ..." for every candidate label, pick the best match."""

    name = "clip-zero-shot"
    default_checkpoint = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        property_name: str,
        labels: list[str],
        device: str | None = None,
        checkpoint: str | None = None,
    ) -> None:
        super().__init__(property_name, labels, device)
        self._template = _PROMPT_TEMPLATES.get(property_name, _DEFAULT_TEMPLATE)

        from transformers import CLIPModel, CLIPProcessor

        checkpoint = checkpoint or self.default_checkpoint
        self._model = CLIPModel.from_pretrained(checkpoint).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(checkpoint)

    def classify(self, crop: Image.Image, context: str | None = None) -> tuple[str, float]:
        import torch

        prompts = [self._template.format(label=label, context=context or _DEFAULT_CONTEXT_NOUN) for label in self.labels]
        inputs = self._processor(text=prompts, images=crop, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1)[0]
        best_idx = int(probs.argmax())
        return self.labels[best_idx], probs[best_idx].item()

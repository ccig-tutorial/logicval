from __future__ import annotations

from PIL import Image

from .base import AttributeClassifier

# One noun-phrase template per property. "{label}" is the candidate value being
# scored; "{context}" is an already-known attribute of the same object (see
# base.AttributeClassifier.classify docstring) -- for properties with no natural
# context dependency the template simply ignores it.
_PROMPT_TEMPLATES: dict[str, list[str]] = {
    "color": [
        "a {label} {context}",
        "a photo of a {label} {context}",
        "a 3D rendered {label} {context}",
        "a {label} geometric {context}",
        "an object that is {label} {context}",
    ],

    "shape": [
        "a {label}",
        "a {label} object",
        "a 3D {label}",
        "a geometric {label}",
        "a 3D object with {label} shape",
    ],

    "material": [
        "a {label} object",
        "a photo of a {label} object",
        "a 3D rendered object made of {label}",
        "an object made of {label}",
        "a geometric object made of {label}",
    ],

    "size": [
        "a {label} object",
        "a photo of a {label} object",
        "a 3D rendered {label} object",
    ],
}
'''
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
'''

class ClipZeroShotAttribute(AttributeClassifier):
    """
    Zero-shot attribute classification with CLIP.

    Classification uses:
      1. multiple prompt templates,
      2. optional multiple image views,
      3. averaging CLIP logits across templates/views,
      4. softmax only after the averaging.

    Example for color with context="cube":

        a red cube
        a blue cube
        a green cube
        ...

    The label with the highest CLIP score is selected.
    """

    name = "clip-zero-shot"
    default_checkpoint = "openai/clip-vit-large-patch14" # "openai/clip-vit-base-patch32"

    def __init__(
        self,
        property_name: str,
        labels: list[str],
        device: str | None = None,
        checkpoint: str | None = None,
    ) -> None:
        super().__init__(property_name, labels, device)

        self.property_name = property_name
        self.labels = labels

        self._templates = _PROMPT_TEMPLATES.get(
            property_name,
            ["a {label} object"],
        )

        from transformers import CLIPModel, CLIPProcessor

        checkpoint = checkpoint or self.default_checkpoint

        self._model = (
            CLIPModel
            .from_pretrained(checkpoint)
            .to(self.device)
            .eval()
        )

        self._processor = CLIPProcessor.from_pretrained(checkpoint)

    def _build_prompts(
        self,
        context: str | None = None,
    ) -> list[str]:
        """
        Build prompts in this order:

            template 1: label 1, label 2, ...
            template 2: label 1, label 2, ...
            ...

        Example:

            [
                "a red cube",
                "a blue cube",
                ...
                "a photo of a red cube",
                "a photo of a blue cube",
                ...
            ]
        """

        context = context or "object"

        prompts: list[str] = []

        for template in self._templates:
            for label in self.labels:
                prompts.append(
                    template.format(
                        label=label,
                        context=context,
                    )
                )

        return prompts

    def _score_image(
        self,
        image: Image.Image,
        context: str | None = None,
    ):
        """
        Return CLIP logits with shape:

            [num_templates, num_labels]
        """

        import torch

        prompts = self._build_prompts(context)

        inputs = self._processor(
            text=prompts,
            images=image.convert("RGB"),
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Shape before reshape:
        #
        # [1, num_templates * num_labels]
        #
        # Reshape to:
        #
        # [num_templates, num_labels]
        logits = outputs.logits_per_image[0]

        logits = logits.reshape(
            len(self._templates),
            len(self.labels),
        )

        return logits

    def classify(
        self,
        crop: Image.Image,
        context: str | None = None,
    ) -> tuple[str, float]:
        """
        Classify one image using all prompt templates.
        """

        import torch

        logits = self._score_image(
            crop,
            context=context,
        )

        # Average the different prompt formulations.
        #
        # [num_templates, num_labels]
        #             ↓ mean
        # [num_labels]
        mean_logits = logits.mean(dim=0)

        # Only normalize AFTER averaging.
        probs = torch.softmax(mean_logits, dim=-1)

        best_idx = int(probs.argmax())

        return (
            self.labels[best_idx],
            float(probs[best_idx]),
        )

    def classify_ensemble(
        self,
        images: Sequence[Image.Image],
        context: str | None = None,
    ) -> tuple[str, float]:
        """
        Classify multiple views/crops of the same object.

        For each image:
            average prompt-template logits

        Then:
            average image-view logits

        Finally:
            softmax + argmax

        This is preferable to majority-voting the individual
        predicted labels.
        """

        import torch

        if not images:
            raise ValueError("classify_ensemble() received no images")

        image_logits = []

        for image in images:
            logits = self._score_image(
                image,
                context=context,
            )

            # Average prompt templates for this particular view.
            mean_logits = logits.mean(dim=0)

            image_logits.append(mean_logits)

        # [num_views, num_labels]
        image_logits = torch.stack(image_logits)

        # Average evidence from all views.
        #
        # [num_views, num_labels]
        #          ↓
        # [num_labels]
        mean_logits = image_logits.mean(dim=0)

        # Convert final scores into probabilities.
        probs = torch.softmax(mean_logits, dim=-1)

        best_idx = int(probs.argmax())

        return (
            self.labels[best_idx],
            float(probs[best_idx]),
        )
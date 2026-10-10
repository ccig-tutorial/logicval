from __future__ import annotations

from PIL import Image

from .base import BBox, ObjectDetector


class Owlv2Detector(ObjectDetector):
    """Open-vocabulary detector, text-prompted per call. Lighter than
    GroundingDinoDetector -- the CPU-friendlier default when no GPU is available."""

    name = "owlv2"
    hf_repo = "google/owlv2-base-patch16-ensemble"

    def __init__(self, device: str | None = None) -> None:
        super().__init__(device)
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self._processor = Owlv2Processor.from_pretrained(self.hf_repo)
        self._model = Owlv2ForObjectDetection.from_pretrained(self.hf_repo).to(self.device)

    def detect(self, image: Image.Image, class_prompts: list[str]) -> list[BBox]:
        import torch

        # Owlv2 responds better to full sentence queries than bare nouns.
        text_labels = [[f"a photo of a {p}" for p in class_prompts]]
        inputs = self._processor(text=text_labels, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([(image.height, image.width)])
        results = self._processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=text_labels
        )[0]

        return [
            BBox(
                x0=box[0].item(), y0=box[1].item(), x1=box[2].item(), y1=box[3].item(),
                # strip the "a photo of a " wrapper back off so labels match domain vocab
                label=label.removeprefix("a photo of a "),
                score=score.item(),
            )
            for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"])
        ]

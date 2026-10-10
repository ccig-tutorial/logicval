from __future__ import annotations

from PIL import Image

from .base import BBox, ObjectDetector


class GroundingDinoDetector(ObjectDetector):
    """Open-vocabulary detector, text-prompted per call. Heavier than Owlv2Detector --
    prefer this one when a GPU is available; both implement the identical detect()
    signature so callers don't need to know which is loaded."""

    name = "grounding-dino"
    hf_repo = "IDEA-Research/grounding-dino-tiny"

    def __init__(self, device: str | None = None) -> None:
        super().__init__(device)
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.hf_repo)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.hf_repo).to(self.device)

    def detect(self, image: Image.Image, class_prompts: list[str]) -> list[BBox]:
        import torch

        text_labels = [class_prompts]  # one query list per image, per this model's batch convention
        inputs = self._processor(images=image, text=text_labels, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            threshold=0.3,
            text_threshold=0.25,
            target_sizes=[(image.height, image.width)],
            text_labels=text_labels,
        )[0]

        return [
            BBox(x0=box[0].item(), y0=box[1].item(), x1=box[2].item(), y1=box[3].item(), label=label, score=score.item())
            for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"])
        ]

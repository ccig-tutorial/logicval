from __future__ import annotations

from PIL import Image

from .base import BBox, ObjectDetector




class GroundingDinoDetector(ObjectDetector):
    """Open-vocabulary detector, text-prompted per call. Heavier than Owlv2Detector --
    prefer this one when a GPU is available; both implement the identical detect()
    signature so callers don't need to know which is loaded."""

    name = "grounding-dino"
    hf_repo = "IDEA-Research/grounding-dino-base" #"IDEA-Research/grounding-dino-tiny" # #

    def __init__(self, device: str | None = None) -> None:
        super().__init__(device)
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.hf_repo)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.hf_repo).to(self.device)

    
    @staticmethod
    def bbox_iou(box1: BBox, box2: BBox) -> float:
    
        """Compute IoU between two bounding boxes."""

        x0 = max(box1.x0, box2.x0)
        y0 = max(box1.y0, box2.y0)
        x1 = min(box1.x1, box2.x1)
        y1 = min(box1.y1, box2.y1)

        intersection_w = max(0.0, x1 - x0)
        intersection_h = max(0.0, y1 - y0)
        intersection = intersection_w * intersection_h

        area1 = max(0.0, box1.x1 - box1.x0) * max(0.0, box1.y1 - box1.y0)
        area2 = max(0.0, box2.x1 - box2.x0) * max(0.0, box2.y1 - box2.y0)

        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    @staticmethod
    def bbox_area(box: BBox) -> float:
        return max(0.0, box.x1 - box.x0) * max(
            0.0, box.y1 - box.y0
        )

    @classmethod
    def containment_ratio(
        cls,
        inner: BBox,
        outer: BBox,
    ) -> float:
        """
        Fraction of `inner` that is contained inside `outer`.

        1.0 means the inner box is completely inside the outer box.
        """

        x0 = max(inner.x0, outer.x0)
        y0 = max(inner.y0, outer.y0)
        x1 = min(inner.x1, outer.x1)
        y1 = min(inner.y1, outer.y1)

        intersection_w = max(0.0, x1 - x0)
        intersection_h = max(0.0, y1 - y0)

        intersection = intersection_w * intersection_h
        inner_area = cls.bbox_area(inner)

        if inner_area <= 0:
            return 0.0

        return intersection / inner_area
    
    
    @classmethod
    def nms(
        cls,
        detections: list[BBox],
        iou_threshold: float = 0.5,
        containment_threshold: float = 0.8,
    ) -> list[BBox]:
        """
        Class-agnostic NMS.

        Grounding DINO can return multiple textual labels for the same
        physical object, e.g.:

            "cube cylinder"
            "cylinder"
            "cube"

        Therefore we deliberately do NOT compare labels here.

        A lower-confidence detection is removed when:
          1. its IoU with a higher-confidence detection is high, OR
          2. it is mostly contained inside a higher-confidence detection.
        """

        if not detections:
            return []

        # Highest confidence first.
        detections = sorted(
            detections,
            key=lambda d: d.score,
            reverse=True,
        )

        kept: list[BBox] = []

        for detection in detections:
            suppress = False

            for existing in kept:
                iou = cls.bbox_iou(detection, existing)

                # Standard NMS.
                if iou >= iou_threshold:
                    suppress = True
                    break

                # Also catch boxes that are almost completely
                # contained within an existing detection.
                containment = cls.containment_ratio(
                    detection,
                    existing,
                )

                if containment >= containment_threshold:
                    suppress = True
                    break

            if not suppress:
                kept.append(detection)

        return kept

    
    def detect(self, image: Image.Image, class_prompts: list[str]) -> list[BBox]:
        import torch
        text_labels = [class_prompts]  # one query list per image, per this model's batch convention
        inputs = self._processor(images=image, text=text_labels, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            threshold=0.15,
            text_threshold=0.15,
            target_sizes=[(image.height, image.width)],
            text_labels=text_labels,
        )[0]

        # Grounding DINO -> BBox objects
        detections = [
            BBox(
            x0=box[0].item(),
            y0=box[1].item(),
            x1=box[2].item(),
            y1=box[3].item(),
            label=label,
            score=score.item(),
            )
            for box, score, label in zip(
            results["boxes"],
            results["scores"],
            results["text_labels"],
            )
        ]

        # Class-aware NMS
        # Remove duplicate detections.
        detections = self.nms(
            detections,
            iou_threshold=0.5,
            containment_threshold=0.8,
        )
        return detections
        
        #return [
        #    BBox(x0=box[0].item(), y0=box[1].item(), x1=box[2].item(), y1=box[3].item(), label=label, score=score.item())
        #    for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"])
        #]

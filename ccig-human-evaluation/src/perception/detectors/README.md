# detectors

Open-vocabulary, zero-shot object detection: given an image and a list of class-name
text prompts (e.g. `["cube", "sphere", "cylinder"]` for CLEVR, `["bicycle", "suitcase",
"chair"]` for COCO — pulled from `domain_clevr.PROPERTIES["shape"]` /
`domain_coco.PROPERTIES["shape"]`, not re-typed), return bounding boxes.

- **`base.py`** — `ObjectDetector` interface (`detect(image, class_prompts) ->
  list[BBox]`), `BBox` dataclass. `device` is resolved once in `__init__`
  (`cuda` if available, else `cpu`); implementations never branch on it themselves.
- **`grounding_dino.py`** — `GroundingDinoDetector`
  (`IDEA-Research/grounding-dino-tiny`). Higher accuracy, heavier — prefer on GPU.
- **`owlv2.py`** — `Owlv2Detector` (`google/owlv2-base-patch16-ensemble`). Lighter,
  the more CPU-friendly default when no GPU is available.
- **`registry.py`** — `DETECTOR_REGISTRY` + `build_detector(name, device=None)`,
  selected via `--detector` on the CLI.

No CLEVR/COCO-specific training was used for either model (chosen per the project
decision to start with zero-shot and revisit accuracy later) — if detection quality
on synthetic CLEVR renders proves insufficient, the next step is a domain-trained
detector implementing the same `ObjectDetector` interface, registered alongside these.

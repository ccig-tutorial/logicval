# attributes

Per-property classification of a detected, cropped object (color, shape, material,
size — whichever properties the active domain defines, see `common/dataset_gen.py`).

- **`base.py`** — `AttributeClassifier` interface: `classify(crop, context=None) ->
  (label, confidence)`. `context` carries an already-known attribute of the same
  object (used for color, see below); classifiers that don't need it ignore it.
- **`clip_zero_shot.py`** — `ClipZeroShotAttribute`, the one v1 implementation.
  Zero-shot CLIP classification: score the crop against `"a photo of a [LABEL] ..."`
  for every candidate label in the property's vocabulary, pick the best match.
  One instance handles one property (constructed with `property_name` + `labels`).
- **`registry.py`** — `ATTRIBUTE_REGISTRY` + `build_attribute_classifier(...)`,
  selected via `--attribute-classifier` on the CLI. Deliberately a single entry today;
  add a non-CLIP classifier (e.g. a small trained head) by subclassing
  `AttributeClassifier` and registering it — nothing else changes.

## Classification order (perception/run.py)

For CLEVR: **shape first** (`classify(crop)`, no context), then **color** with
`context=predicted_shape` (`"a photo of a red cube"` is a stronger zero-shot query
than `"a photo of a red object"`), then **material** and **size** independently.
For COCO: category comes directly from the detector's matched label — only
**color** is classified (no material/size/shape classifiers; COCO's domain module
doesn't define those properties).

Labels for every classifier come straight from `domain_clevr.PROPERTIES` /
`domain_coco.PROPERTIES` (imported live via `common/dataset_gen.py`), never re-typed
— this is what keeps CLEVR's 8 colors (incl. gray) vs. COCO's 7 (no gray) correct
automatically, and keeps the label set in sync if the domain module changes.

# perception

Turns a generated image into a scene graph: detect objects, classify their
properties, and derive their region/relations from geometry. Deliberately does
**not** run the ASP/clingo constraint check ccig-evaluation's own perception
pipeline does (see `../common/dataset_gen.py`) -- this package only produces the
detected `scene_graph`, nothing about whether it satisfies the original constraint.

Entry point: `run_perception()` in `run.py`, called from `app.py`'s "Run perception
on all images" button (not a standalone CLI -- there is no `src/run.py` here). It
loops over matched `(image, PromptRecord)` pairs and calls `_perceive_scene()` once
per image.

## Pipeline, in order

```
run_perception()                                        run.py
  │
  ├─ load_domain(domain)                                 ../common/dataset_gen.py
  │     -> domain_clevr / domain_coco module (PROPERTIES vocab: shapes/colors/...)
  │
  ├─ _load_prior_results(out_path, ...)                   run.py
  │     -> reuse any already-successful result for an (id, prompt_field) instead
  │        of redoing detection/classification (resume support)
  │
  ├─ build_detector(detector_name)  -- lazy, only if something actually needs
  │  build_attribute_classifier(...) per property  (re)processing (see run.py)
  │     -> GroundingDinoDetector | Owlv2Detector           detectors/registry.py
  │        one classifier per domain_module.PROPERTIES key  attributes/registry.py
  │        (built once per run, not per object/image)
  │
  └─ for each MatchedItem not already done (image + its PromptRecord):
        │
        ├─ 1. DETECT                                       run.py :: _perceive_scene()
        │      detector.detect(image, class_prompts)        detectors/*.py
        │      class_prompts = domain_module.PROPERTIES["shape"]
        │      -> list[BBox]                                detectors/base.py
        │
        ├─ 2. CROP + NEUTRALIZE (per box)                   crop.py
        │      crop_and_neutralize(image, bbox)
        │      -> a PIL crop isolating just that object
        │
        ├─ 3. CLASSIFY ATTRIBUTES (per crop)                run.py :: _classify_object_properties()
        │      CLEVR: shape -> color (shape-conditioned)     attributes/clip_zero_shot.py
        │             -> material ("size" deliberately
        │             excluded -- never shown, classified,
        │             or saved)
        │      COCO:  color only (shape/category comes       (see attributes/README.md)
        │             straight from the detector's label)
        │
        ├─ 4. REGION + RELATIONS (per box, then pairwise)   regions.py
        │      region_of(cx, cy, W, H)      -> "r0".."r3" quadrant
        │      pairwise_relations(regions)  -> left/right/front/behind
        │      (static 2x2 adjacency lookup, transcribed from
        │       ccig-dataset-gen/.../asp_background/background.lp -- plain
        │       Python; this pipeline never calls clingo)
        │
        │      => one DetectedObject per box                types.py
        │         {obj_id, bbox, properties, region}
        │
        └─ 5. RECORD                                        run.py
               to_graph_dict(objects)                        scene_graph.py (JSON scene graph)
               -> PerceptionResult                            ../common/types.py
                  (one per image; appended to results list)

  write_json(out_path, {method, domain, detector, attribute_classifier, results})
                                                               ../common/io.py
  (written after every item, not just at the end, so a killed/resumed run
  never loses already-computed results)
```

## File-by-file summary

| File | Role |
|---|---|
| `run.py` | Orchestrates the whole pipeline per image and writes `PerceptionResult`s. |
| `detectors/` | Step 1 — open-vocabulary bounding boxes for the domain's class prompts. See `detectors/README.md`. |
| `crop.py` | Step 2 — crop + neutralize a bbox's region out of the full image before classification. |
| `attributes/` | Step 3 — one CLIP zero-shot classifier per property, built once per run. See `attributes/README.md`. |
| `regions.py` | Step 4 — bbox center -> quadrant region, and the static adjacency table -> pairwise relations. |
| `types.py` | `DetectedObject` — the per-object record threaded from step 4 onward. |
| `scene_graph.py` | Step 5 — `to_graph_dict()`, the JSON-friendly scene graph for the output file. |
| `../common/dataset_gen.py` | Live import shim into `ccig-dataset-gen/src` for `load_domain` (vocabulary only — never imports the ASP solver). |
| `../common/types.py` | `MatchedItem` (image + `PromptRecord` in), `PerceptionResult` (out). |
| `../common/io.py` | Matches images to prompt records by `id`; writes the results JSON atomically. |

## Object numbering

`DetectedObject.obj_id` is just the detector's output order (0, 1, 2, ...) — it has
no relationship to any id in the original CLEVR/COCO scene the prompt was generated
from. That never mattered for anything downstream here, since this pipeline doesn't
evaluate `instantiated_rule` against the detected objects at all.

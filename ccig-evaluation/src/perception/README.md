# perception

The core new capability: turn a generated image into a scene graph, then check it
against the original logical constraint with clingo.

Entry point: `run_perception()` in `run.py`, called once per `--method perception` run
from `src/run.py`. It loops over matched `(image, PromptRecord)` pairs and calls
`_perceive_scene()` once per image, then does the ASP solve + comparison itself.

## Running it

Two separate inputs are required, run from `ccig-evaluation/` (paths below are
relative to the repo root, `CCIG_Eval/`):

```
python -m src.run \
  --images-dir     data/generated_images/<batch>/          \  # generated PNGs
  --prompts-file   data/ccig_eval_dataset/<domain>_<combo>_scenes_{SAT,UNSAT}.json \  # ground truth
  --method perception \
  --domain clevr \
  --detector owlv2 \          # or grounding-dino (GPU)
  --device cpu                # or cuda; omit to auto-detect
```

- **`--images-dir`** — a folder of `{id}-{short|medium|long}.png` files, e.g.
  `data/generated_images/flux.1-dev-batch1-step001921/`. Also has a `manifest.jsonl`
  next to the images (written by `ccig-image-generation`), but perception does not
  read it — images are matched to records by parsing the filename, not the manifest.
- **`--prompts-file`** — the ground-truth records for that same domain/batch, e.g.
  `data/ccig_eval_dataset/clevr_1_scenes_SAT.json` (despite the `.json` extension,
  it's one JSON record per line). This is what supplies `instantiated_rule` and
  `status` — it comes from `ccig-dataset-gen`, not from image generation, and there
  is no default path for it; it must always be passed explicitly and match the
  batch `--images-dir` was generated from.

No third "single manifest" input exists today — an images folder alone is not
enough to run perception on.

## Pipeline, in order

```
run_perception()                                        run.py
  │
  ├─ load_domain(domain)                                 ../common/dataset_gen.py
  │     -> domain_clevr / domain_coco module (PROPERTIES vocab: shapes/colors/...)
  │
  ├─ build_detector(detector_name)                        detectors/registry.py
  │     -> GroundingDinoDetector | Owlv2Detector           detectors/grounding_dino.py
  │                                                        detectors/owlv2.py
  │
  ├─ build_attribute_classifier(...) per property          attributes/registry.py
  │     -> one classifier per domain_module.PROPERTIES key  attributes/clip_zero_shot.py
  │        (built once per run, not per object/image)
  │
  └─ for each MatchedItem (image + its PromptRecord):
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
        │             -> material -> size
        │      COCO:  color only (shape/category comes       (see attributes/README.md)
        │             straight from the detector's label)
        │
        ├─ 4. REGION + RELATIONS (per box, then pairwise)   regions.py
        │      region_of(cx, cy, W, H)      -> "r0".."r3" quadrant
        │      pairwise_relations(regions)  -> left/right/front/behind
        │      (static 2x2 adjacency lookup, transcribed from
        │       ccig-dataset-gen/.../asp_background/background.lp -- plain
        │       Python, not clingo; clingo is reserved for step 6)
        │
        │      => one DetectedObject per box                types.py
        │         {obj_id, bbox, properties, region}
        │
        ├─ 5. BUILD ASP FACTS                                scene_graph.py
        │      build_scene_facts(objects)
        │      -> object/1, hasProperty/3, hasRelationship/3 text,
        │         in the exact atom shapes ccig-dataset-gen's
        │         format_scene() parses
        │
        ├─ 6. SOLVE                                          run.py (calls out to)
        │      program = facts + "\n" + item.record.instantiated_rule
        │      solve(program, ...)                           ../common/dataset_gen.py
        │                                                     -> ccig-dataset-gen/src/eval_dataset_gen/solve.py
        │      -> predicted_status ("SAT" | "UNSAT")          (subprocess call to the clingo binary)
        │
        └─ 7. COMPARE + RECORD                                run.py
               agrees_with_dataset = predicted_status == item.record.status
               to_graph_dict(objects)                         scene_graph.py (JSON-friendly scene graph)
               -> PerceptionResult                             ../common/types.py
                  (one per image; appended to results list)

  write_json(out_path, {method, domain, detector, attribute_classifier, results})
                                                               ../common/io.py
```

## File-by-file summary

| File | Role |
|---|---|
| `run.py` | Orchestrates the whole pipeline per image; the only place that calls `solve()` and writes `PerceptionResult`s. |
| `detectors/` | Step 1 — open-vocabulary bounding boxes for the domain's class prompts. See `detectors/README.md`. |
| `crop.py` | Step 2 — crop + neutralize a bbox's region out of the full image before classification. |
| `attributes/` | Step 3 — one CLIP zero-shot classifier per property, built once per run. See `attributes/README.md`. |
| `regions.py` | Step 4 — bbox center -> quadrant region, and the static adjacency table -> pairwise relations. |
| `types.py` | `DetectedObject` — the per-object record threaded from step 4 onward. |
| `scene_graph.py` | Step 5/7 — `build_scene_facts()` (ASP text for clingo) and `to_graph_dict()` (JSON for the output file). |
| `../common/dataset_gen.py` | Live import shim into `ccig-dataset-gen/src` for `load_domain`, `solve`, `format_scene`. |
| `../common/types.py` | `MatchedItem` (image + `PromptRecord` in), `PerceptionResult` (out). |
| `../common/io.py` | Matches images to prompt records by `id`; writes the final results JSON. |

## Object numbering

`DetectedObject.obj_id` is just the detector's output order (0, 1, 2, ...) — it has
no relationship to any id in the original CLEVR/COCO scene the prompt was generated
from. That's fine because `instantiated_rule` constraints (step 6) use free ASP
variables (`X`, `Y`, ...) bound over `object(X)`, not literal object ids, so they
apply unchanged no matter how perception numbered the detected objects.

## Why perception detects more than color/shape/material

CLEVR's `domain_clevr.PROPERTIES` also defines `size` (small/large). Despite
`ccig-dataset-gen/src/eval_dataset_gen/domain.py`'s docstring claiming `material` is
excluded from its ASP constraint search space, the code there actually excludes
`size` — and real dataset records do contain `size`-based `instantiated_rule`s. So
attribute classification here (step 3) is driven generically by
`domain_module.PROPERTIES`, not a hardcoded property list, to cover whatever the
dataset actually constrains on.

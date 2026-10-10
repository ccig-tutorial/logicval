# common

Shared, provider-agnostic pieces used by every evaluation method.

- **`dataset_gen.py`** — imports `ccig-dataset-gen/src`'s domain constants
  (`domain_clevr`, `domain_coco` — colors/shapes/materials/regions vocabulary) *live*,
  by adding that sibling pipeline's `src/` to `sys.path` at import time. Nothing here
  is copied: if `ccig-dataset-gen` changes its color list, `ccig-human-evaluation`
  picks it up automatically. This requires `ccig-dataset-gen/` to be checked out as a
  sibling directory — the shim raises immediately at import time if it isn't found.
  Deliberately does **not** import `ccig-dataset-gen`'s clingo wrapper
  (`eval_dataset_gen.solve`) — this pipeline never runs the ASP constraint check.
- **`io.py`** — matches generated images to the prompt records that produced them.
  Images are expected as `{id}-{prompt_field}.png` (the convention `ccig-image-generation`
  writes, `prompt_field ∈ {short, medium, long}`); `{id}` must match the `id` field of a
  record in the `ccig_eval_dataset_{SAT,UNSAT}.jsonl` passed via `--prompts-file`.
  `match_images_to_prompts()` is the main entry point other modules call.
- **`types.py`** — dataclasses shared across methods: `PromptRecord`, `MatchedItem`, and
  one result dataclass per evaluation method (`ClipScoreResult`, `JudgeResult`,
  `PerceptionResult`), each with `to_json()` for writing output files.

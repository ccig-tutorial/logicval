# CCIG Image Generation

Generates images from CCIG eval prompts (`data/ccig_eval_dataset/*.json`, one JSONL file per
domain/complexity/status combination, e.g. `clevr_3_scenes_SAT.json`) using text-to-image models.
Generation only — scoring/constraint checking against the prompts is a separate, future module.

## Structure

```
src/
├── common/          # shared types + dataset I/O, provider-agnostic
├── closed/          # closed-source, API-based models (OpenAI, Google, ...)
├── open/            # open-weight, locally-run models (diffusers, ...)
└── run.py           # CLI entry point
```

## Closed-source models

Registered in `src/closed/registry.py`:

| model                | provider | notes                                                              |
|-----------------------|----------|---------------------------------------------------------------------|
| `gpt-image-1`         | OpenAI   | legacy -- kept for comparison, superseded by `gpt-image-2`          |
| `gpt-image-2`         | OpenAI   | current flagship (released Apr 2026); `quality` set on the class (`low`/`medium`/`high`) -- see pricing note below |
| `gemini-2.0-flash`    | Google   | legacy -- kept for comparison; despite the registry name, actually calls `gemini-2.5-flash-image` ("Nano Banana") |
| `gemini-3.1-flash-image` | Google | cheaper/faster sibling of `gemini-3-pro-image` -- closest Google equivalent to `gpt-image-2`'s "low" tier (Nano Banana Pro has no cheap tier: 1K and 2K cost the same) |
| `gemini-3-pro-image`  | Google   | current flagship, aka "Nano Banana Pro"; `image_size` set on the class (`1K`/`2K`/`4K`, 1K/2K priced the same) |

`gpt-image-2`, `gemini-3.1-flash-image`, and `gemini-3-pro-image` all price per image by
quality/resolution tier rather than a flat rate -- check `quality`/`image_size` on the model class
in `src/closed/gpt_image.py` / `src/closed/gemini.py` before a large run, since it directly drives
cost. Rough per-image figures at the smallest useful size (verify against your own usage
dashboard -- these drift and vary by source):
- `gpt-image-2` @ 1024x1024: ~$0.006 low, ~$0.05-0.08 medium, ~$0.21 high
- `gemini-3.1-flash-image` @ 1K: ~$0.067 (~$0.034 batch)
- `gemini-3-pro-image` @ 1K/2K: ~$0.134 (~$0.067 batch); 4K: ~$0.24

## Open-source models

Registered in `src/open/registry.py`, implemented on top of `diffusers`:

| model            | status  | gated? | notes                                   |
|-------------------|---------|--------|-------------------------------------------|
| `sd3.5-large`     | working | yes    | `StableDiffusion3Pipeline` -- accept the license on the model page, then `huggingface-cli login` |
| `flux.1-dev`      | working | yes    | `FluxPipeline`, guidance-distilled -- accept the license on the model page, then `huggingface-cli login` |
| `flux.1-schnell`  | working | yes    | `FluxPipeline`, few-step, not guidance-distilled -- same login requirement as `flux.1-dev` |
| `qwen-image`      | working | no     | `QwenImagePipeline`, guidance-distilled; needs a recent `diffusers` version |

"gated" is per the Hugging Face Hub API (checked directly, not assumed) -- gated repos need you
to accept the model's license on its Hub page while logged in, then `huggingface-cli login`
locally before `from_pretrained` will succeed.

All working models share one implementation (`src/open/_diffusers_common.py`) that loads
base weights from Hugging Face Hub and, when `--checkpoint` is passed, loads a LoRA
finetuned checkpoint on top via `peft`'s `PeftModel.from_pretrained` -- the same format
`ccig-finetuning` writes checkpoints in (`transformer.save_pretrained(...)`).

## Setup

```bash
cd ccig-image-generation

# closed, API-based models only
pip install -r requirements.txt

# adds the open-source / diffusers stack (torch, diffusers, transformers, peft, ...)
pip install -r requirements-open.txt
huggingface-cli login   # required for gated repos: sd3.5-large, flux.1-dev, flux.1-schnell

export OPENAI_API_KEY=...    # for gpt-image-1, gpt-image-2
export GEMINI_API_KEY=...    # for gemini-2.0-flash, gemini-3-pro-image (or GOOGLE_API_KEY)
```

`.env.example` in this directory lists both keys; the code does not auto-load a `.env` file
(no `python-dotenv` dependency), so `set -a && source .env && set +a` (or export manually)
before running.

## Eval dataset files

`data/ccig_eval_dataset/` has 6 JSONL files (484 samples total), one per
domain/complexity-class/status combination:

| file                          | samples |
|---------------------------------|---------|
| `clevr_1_scenes_SAT.json`     | 355     |
| `clevr_1_scenes_UNSAT.json`   | 22      |
| `clevr_3_scenes_SAT.json`     | 25      |
| `clevr_3_scenes_UNSAT.json`   | 17      |
| `clevr_6_scenes_SAT.json`     | 25      |
| `clevr_6_scenes_UNSAT.json`   | 40      |

## Run

```bash
# single file, a few samples (smoke test / quality pilot)
python -m src.run --model gpt-image-2 --dataset ../data/ccig_eval_dataset/clevr_3_scenes_SAT.json --prompt-field short --limit 5
python -m src.run --model gemini-3-pro-image --dataset ../data/ccig_eval_dataset/clevr_3_scenes_SAT.json --prompt-field short --limit 5

# single file, full run, one prompt field
python -m src.run --model gpt-image-2 --dataset ../data/ccig_eval_dataset/clevr_6_scenes_UNSAT.json --prompt-field long

# all 6 dataset files, one prompt field, for a given model
for f in ../data/ccig_eval_dataset/*.json; do
  python -m src.run --model gpt-image-2 --dataset "$f" --prompt-field short
done

# open-source model, base weights
python -m src.run --model flux.1-schnell --limit 5

# open-source model, LoRA finetuned checkpoint (from ccig-finetuning)
python -m src.run --model sd3.5-large --checkpoint ../outputs/checkpoints-finetuning/sd3.5-large/run1
```

Options:
- `--prompt-field {short,medium,long}` — which NL description to use as the prompt (default `medium`)
- `--out` — output root (default `../data/generated_images`, i.e. the repo-root `data/` folder)
- `--limit` — cap the number of prompts processed
- `--dataset` — path to one of the eval-dataset files under `data/ccig_eval_dataset/`
  (the CLI default is stale -- always pass this explicitly)
- `--checkpoint` — path to a LoRA finetuned checkpoint directory (open-source models only)

## Output

Images and a manifest are written to `data/generated_images/<model>[-<variant>]/<dataset>/`,
where `<variant>` is the checkpoint name (open models) or quality/resolution tier (closed models
that expose one, e.g. `gpt-image-2-low`) and `<dataset>` is the stem of `--dataset`
(e.g. `clevr_3_scenes_SAT`):
- `data/generated_images/<model>[-<variant>]/<dataset>/<prompt_id>-<prompt_field>.png`
- `data/generated_images/<model>[-<variant>]/<dataset>/manifest.jsonl` — one record per prompt:
  `id`, `model`, `prompt`, `prompt_field`, `scene_generation_setup`, `image_path`, `success`,
  `error`, `variant`

## Repairing a manifest (retry missing/errored rows)

A `run.py` pass can leave gaps in the manifest -- rows with `error` set (a transient API
failure, rate limit, depleted credits, ...) or missing entirely (the process died partway
through). `src/repair_manifest.py` fills those in without regenerating everything:

```bash
python -m src.repair_manifest --model gemini-3-pro-image \
  --dataset ../data/ccig_eval_dataset/clevr_1_scenes_SAT.json \
  --prompt-field long
```

For every `(id, prompt_field)` pair expected for that dataset (every dataset id crossed with
`--prompt-field`, or -- if `--prompt-field` is omitted -- whatever fields already appear in the
existing manifest), it:
- leaves a good row (`success: true`, `error: null`) untouched, printing `[exists]`;
- if the manifest row is missing/errored but the image is already on disk (e.g. an earlier
  manifest write got clobbered, not generation never happening), recovers it from the file
  without calling the API again, printing `[recovered]`;
- otherwise retries generation exactly once, printing `[fixed]` or `[still-failing]` depending
  on the outcome;
- rewrites `manifest.jsonl` after **every** row it touches, not once at the end -- a crash,
  Ctrl-C, or API outage partway through an interrupted run never loses already-completed
  retries, and the file on disk always reflects true current state. This also collapses any
  stray duplicate rows left over from earlier partial/rerun manifests down to one row per
  id+field, and carries through rows for any `prompt_field` not passed to this call untouched
  (it will never be the reason another field's rows disappear).

Same `--out`/`--checkpoint` flags as `run.py`; the output folder is resolved with the exact same
`<model>[-<variant>]/<dataset>/` convention, so make sure `--model`/`--checkpoint` match the
original run or this will look at (or create) the wrong folder. Also callable directly as a
function -- `from src.repair_manifest import repair_manifest` -- if you want to script repairs
across several dataset files.

## Adding a model

- **Closed (API-based)**: add a class in `src/closed/<provider>.py` implementing
  `ClosedImageModel.generate(prompt) -> Image`, register it in `src/closed/registry.py`.
- **Open (diffusers-based)**: add a class in `src/open/<model>.py` subclassing
  `DiffusersImageModel` (set `pipeline_cls`, `hf_repo`, `_call_kwargs`), register it in
  `src/open/registry.py`.
- **Open (non-diffusers)**: subclass `OpenImageModel` directly and implement `generate()` with
  the model's own inference code.

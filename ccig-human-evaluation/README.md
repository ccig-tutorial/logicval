# ccig-human-evaluation

A browser-based tool for hand-annotating scene graphs on generated CLEVR/COCO
images. It lets a human's labels be compared against `ccig-evaluation`'s
automated perception pipeline, so you can measure how accurate that pipeline
actually is, image by image.

For each image you draw a bounding box per object and pick its
shape/color/material from a fixed vocabulary; region and left/right/
front/behind relations are computed automatically from the boxes. If
perception has already run on the batch, its (imperfect) detections
pre-populate the boxes so you're correcting, not starting from scratch.

## Setup

```bash
cd ccig-human-evaluation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# only needed to run the "Run perception on all images" button:
pip install torch "transformers>=4.46" accelerate scipy
```

## Running it

```bash
python -m src.app --port 5001
```

`http://localhost:5001` lands on the [browse page](#browse-page) — for the
original images_dir/prompts_file-driven flow (which also runs perception),
open `http://localhost:5001/setup` and fill in the form:

| Field | Example |
|---|---|
| Images dir | `data/generated_images/gpt-image-2-low/clevr_1_scenes_SAT` |
| Prompts file | `data/ccig_eval_dataset/clevr_1_scenes_SAT.jsonl` |

Both must be from the same generation batch. Output location is derived
automatically, not entered by hand (see below).

## Input / output

**Input:** an images dir of `{id}-{short|medium|long}.png` files, and the
matching `ccig_eval_dataset_*.json` prompts/ground-truth file (one JSON
record per line) that batch was generated from.

**Output:** two JSON files, written to a location derived from the images
dir (`<model>` = the folder right after `generated_images`) and the prompts
file (`<dataset>` = its filename stem):

```
data/evaluation/<model>/<dataset>_auto-perception.json
data/evaluation/<model>/<dataset>_human-perception.json
```

Both share the same entry shape (one object per image, `results`/
`annotations` list) and the same field names as the ground-truth file where
the concept is shared: `id`, `prompt_field`, `image_path`, `prompt`,
`instantiated_rule`, `status`, `number_of_objects`, `scene_graph`. Neither
file references the other's path, and neither carries anything ASP/clingo-
related — this pipeline only detects objects and their properties, it never
runs the constraint-satisfaction check ccig-evaluation's own perception
pipeline does. `auto-perception` additionally carries `success`/`error` (did
detection succeed on this image). `human-perception` additionally carries
`reasonable_scene` (`true`/`false`/`null`) — a human-only judgment of whether
the generated image is a sensible scene at all, set per image in the
annotation view (Yes/No toggle); its starting value (before that image has
ever been saved) is whatever you picked on the setup form.

If a human entry is missing but an automated one exists for that image, it's
auto-copied over as a starting point the next time perception runs or the
batch is loaded — so every autosaved edit always starts from perception's
guess unless you clear it (or click "Mark scene as empty").

## Browse page

`http://localhost:5001/browse` (also the landing page at `/` — see below) is a
second, independent page for opening any `*_human-perception.json` file
directly, without going through the setup form: three dropdowns in one row --
**Model**, **Dataset**, **Image**.
- **Model** lists every folder directly under `data/evaluation/`.
- **Dataset** (once a model's picked) lists every `*_human-perception.json`
  file directly inside that model's folder.
- **Image** (once a dataset's picked) lists that file's images; picking one
  opens the same box/property editor as the main page.

No prompt, no constraint, no automated-perception comparison, no file paths
shown anywhere — just the model, the dataset, the image, and the annotation.
The image itself is always located as `data/generated_images/<model>/
<dataset>/<id>-<field>.png` from the model/dataset you picked — never from
the `image_path` stored inside the JSON entry, which is whatever path the
machine that originally ran perception happened to have (e.g. a local
absolute path that doesn't exist inside a Cloud Run deployment's mounted
`/srv/data`). It edits the same `human-perception.json` files described
above (nothing separate to keep in sync), and doesn't require or touch a
configured batch on the main page.

The original setup-form page still exists, at `/setup`.

## Password protection

There's no authentication by default -- fine for local use, not safe to expose
as-is (`/api/setup` reads arbitrary local file paths; the save endpoints let
anyone with the URL overwrite your annotations). Set a password to turn it on:

```bash
CCIG_HUMAN_EVAL_PASSWORD=some-long-random-string python -m src.app --port 5001
```

Every route (main page, `/browse`, all `/api/...` endpoints) then requires
HTTP Basic Auth with that password (any username) -- your browser will prompt
for it once and remember it for the session. Leaving the variable unset keeps
the server exactly as before, no prompt, nothing changed.

## Deploying somewhere other than your own machine

See **[README-GCP.md](README-GCP.md)** for deploying this to Google Cloud Run
(`deploy.sh`, `Dockerfile`) so remote annotators can use it over the internet.

## Not yet implemented

- Batch/aggregate agreement metrics (perception vs. human, judge vs. human,
  CLIP score vs. human) — this tool only produces the raw annotation files;
  scoring them is a separate step.

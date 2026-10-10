#!/usr/bin/env python3
"""Flask backend for the human scene-graph annotation tool.

Single-session, local-only tool (no auth, in-memory state) -- point it at an
images dir + a prompts file, optionally run the perception pipeline once to
get a starting-point scene graph per image, then annotate by hand in the
browser. Output locations are derived, not chosen by hand, from the nearest
"data" ancestor directory of images_dir, plus the model name (the path
component right after "generated_images") and dataset name (prompts_file's
stem):

    <data>/evaluation/<model>/<dataset>_auto-perception.json
    <data>/evaluation/<model>/<dataset>_human-perception.json

Both are single combined files (one entry per image, keyed by id/prompt_field),
not one file per image. Every edit autosaves into <dataset>_human-perception.json
-- <dataset>_auto-perception.json (the perception pipeline's own output) is
never touched by that autosave.

Run from `ccig-human-evaluation/`:
    python -m src.app --port 5001
Then open http://localhost:5001 and fill in the setup form.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, redirect, request, send_file, send_from_directory

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.common.dataset_gen import load_domain  # noqa: E402
from src.common.io import match_images_to_prompts, write_json  # noqa: E402
from src.common.types import MatchedItem  # noqa: E402
from src.perception.detectors.base import BBox  # noqa: E402
from src.perception.regions import bbox_center, region_of  # noqa: E402
from src.perception.run import run_perception  # noqa: E402
from src.perception.scene_graph import to_graph_dict  # noqa: E402
from src.perception.types import DetectedObject  # noqa: E402

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"), static_url_path="")

# ---------------------------------------------------------------------------
# Optional password gate. Off by default (plain local usage is unaffected) --
# only turns on when CCIG_HUMAN_EVAL_PASSWORD is set in the environment, e.g.
# right before exposing this server through a public tunnel (ngrok, cloudflared,
# ...). The tunnel gives you HTTPS transport; it does nothing about access
# control on its own, and this app has none otherwise -- every route (including
# /api/setup, which reads arbitrary local file paths) would be reachable by
# anyone with the URL.
_AUTH_PASSWORD = os.environ.get("CCIG_HUMAN_EVAL_PASSWORD")


@app.before_request
def _require_auth():
    if not _AUTH_PASSWORD:
        return None
    auth = request.authorization
    if auth is None or not secrets.compare_digest(auth.password, _AUTH_PASSWORD):
        return Response(
            "Authentication required.",
            401,
            {"WWW-Authenticate": 'Basic realm="ccig-human-evaluation"'},
        )
    return None


# ---------------------------------------------------------------------------
# In-memory session state. One annotator, one browser tab, one images-dir at
# a time -- global state is fine here, this isn't a multi-user server.
# ---------------------------------------------------------------------------
STATE: dict = {
    "configured": False,
    "images_dir": None,
    "prompts_file": None,
    "domain": "clevr",
    "detector": "owlv2",
    "attribute_classifier": "clip-zero-shot",
    "device": None,
    "model_name": None,  # path component right after "generated_images", e.g. "gpt-image-2-low"
    "dataset_name": None,  # prompts_file's stem, e.g. "clevr_1_scenes_SAT"
    "perception_dir": None,  # <data>/evaluation/<model>/ -- holds both json files below
    "items_by_key": {},  # (id, field) -> MatchedItem
    "automated_by_key": {},  # (id, field) -> PerceptionResult dict (from <dataset>_auto-perception.json)
    "image_size_cache": {},  # str(path) -> (w, h)
}
PERCEPTION_JOB: dict = {"status": "idle", "done": 0, "total": 0, "current": None, "error": None}


def _key(image_id: str, field: str) -> str:
    return f"{image_id}::{field}"


def _perception_out_path() -> Path:
    return Path(STATE["perception_dir"]) / f"{STATE['dataset_name']}_auto-perception.json"


def _human_out_path() -> Path:
    return Path(STATE["perception_dir"]) / f"{STATE['dataset_name']}_human-perception.json"


def _load_human_annotations() -> dict:
    """Reads <dataset>_human-perception.json fresh off disk every call (not cached in
    STATE) -- it's a single small file, and re-reading avoids ever serving a stale
    in-memory copy if the file's been edited outside this server."""
    path = _human_out_path()
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {_key(a["id"], a["prompt_field"]): a for a in payload.get("annotations", [])}


def _save_human_annotation(image_id: str, field: str, entry: dict) -> None:
    """Read-modify-write the combined <dataset>_human-perception.json: replace this
    image's entry, leave every other image's entry untouched."""
    annotations = _load_human_annotations()
    annotations[_key(image_id, field)] = entry
    write_json(
        _human_out_path(),
        {
            "method": "human_perception",
            "domain": STATE["domain"],
            "annotations": list(annotations.values()),
        },
    )


def _seed_human_from_automated() -> None:
    """For every image with an automated-perception entry but no human-perception
    entry yet, write a copy of the automated entry as the starting human-perception
    entry (same field names -- see PerceptionResult) so opening an unannotated image
    always starts from perception's guess, not blank, and so the two are "matched"
    (see _scene_graphs_match) until you actually change something. Images with no
    automated entry are left alone -- nothing to copy. Called after every (re)load of
    the automated-perception file, so it stays true whether that file existed before
    this server started or was just produced by "Run perception"."""
    human_by_key = _load_human_annotations()
    to_add = {
        k: {
            "id": result["id"],
            "prompt_field": result["prompt_field"],
            "image_path": result["image_path"],
            "prompt": result.get("prompt"),
            "instantiated_rule": result["instantiated_rule"],
            "status": result["status"],
            "number_of_objects": result["number_of_objects"],
            "scene_graph": result["scene_graph"],
            "annotated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        for k, result in STATE["automated_by_key"].items()
        if k not in human_by_key and result.get("scene_graph") is not None
    }
    if not to_add:
        return
    human_by_key.update(to_add)
    write_json(
        _human_out_path(),
        {
            "method": "human_perception",
            "domain": STATE["domain"],
            "annotations": list(human_by_key.values()),
        },
    )


def _scene_graphs_match(human_sg: dict, automated_sg: dict) -> bool:
    """True if the two scene graphs describe the same objects (shape/color/material/
    region/bbox -- "size" is deliberately excluded everywhere, see _domain_vocab),
    ignoring detector confidence (det_score, which a hand-drawn box never
    meaningfully has) and object-id/ordering -- i.e. answers "has the human
    annotation actually diverged from what perception detected", not "are the two
    JSON blobs byte-identical"."""
    keys = ("shape", "color", "material", "region", "bbox")

    def normalize(sg: dict) -> list:
        items = []
        for obj in sg.get("objects", {}).values():
            items.append(tuple(tuple(obj.get(key, [])) if key == "bbox" else obj.get(key) for key in keys))
        return sorted(items, key=repr)

    return normalize(human_sg) == normalize(automated_sg)


def _find_data_root(path: Path) -> Path | None:
    """Walk up from `path` looking for a directory literally named "data" --
    the sibling of generated_images/ and ccig_eval_dataset/ at the repo root.
    Independent of exactly how deep images_dir is under it."""
    for candidate in (path, *path.parents):
        if candidate.name == "data":
            return candidate
    return None


def _model_name_from_images_dir(images_dir: Path) -> str:
    """The model/batch identifier is the path component right after
    "generated_images", not just images_dir's own folder name -- some batches
    (e.g. gpt-image-2-low) nest a per-dataset subfolder under the model folder
    (generated_images/gpt-image-2-low/clevr_3_scenes_SAT/), where images_dir.name
    would wrongly be the dataset name instead. Others (e.g. the flux batches)
    put images directly in generated_images/<model>/, where the two coincide."""
    parts = images_dir.parts
    if "generated_images" in parts:
        idx = parts.index("generated_images")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return images_dir.name


def _load_perception_file() -> None:
    """(Re)load <dataset>_auto-perception.json into STATE, if it exists -- e.g. after a
    server restart, or after run_perception finishes."""
    path = _perception_out_path()
    STATE["automated_by_key"] = {}
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    for result in payload.get("results", []):
        if result.get("success") and result.get("scene_graph") is not None:
            STATE["automated_by_key"][_key(result["id"], result["prompt_field"])] = result


def _image_size(path: Path) -> tuple[int, int]:
    cache = STATE["image_size_cache"]
    key = str(path)
    if key not in cache:
        from PIL import Image

        with Image.open(path) as im:
            cache[key] = im.size
    return cache[key]


def _domain_vocab() -> dict:
    # "size" is deliberately excluded from the vocab the UI ever sees -- neither shown
    # nor classified nor saved, for either automated- or human-perception, per user
    # request. domain_module.PROPERTIES (from the shared ccig-dataset-gen sibling) is
    # not itself modified -- other pipelines may still need "size" -- this just never
    # forwards it past this one point.
    domain_module = load_domain(STATE["domain"])
    return {k: v for k, v in domain_module.PROPERTIES.items() if k != "size"}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
@app.route("/api/setup", methods=["POST"])
def api_setup():
    body = request.get_json(force=True)
    images_dir = Path(body["images_dir"]).expanduser().resolve()
    prompts_file = Path(body["prompts_file"]).expanduser().resolve()
    domain = body.get("domain", "clevr")

    if not images_dir.is_dir():
        return jsonify({"error": f"images_dir does not exist: {images_dir}"}), 400
    if not prompts_file.is_file():
        return jsonify({"error": f"prompts_file does not exist: {prompts_file}"}), 400

    data_root = _find_data_root(images_dir)
    if data_root is None:
        return jsonify(
            {"error": f"could not find a 'data' ancestor directory of images_dir ({images_dir}) -- "
                      "expected it under .../data/generated_images/<model>/"}
        ), 400

    try:
        items: list[MatchedItem] = match_images_to_prompts(images_dir, prompts_file)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"failed to match images to prompts: {e!r}"}), 400
    if not items:
        return jsonify({"error": "no images matched the prompts file -- check the two paths belong to the same batch"}), 400

    model_name = _model_name_from_images_dir(images_dir)
    dataset_name = prompts_file.stem
    perception_dir = data_root / "evaluation" / model_name
    perception_dir.mkdir(parents=True, exist_ok=True)

    STATE.update(
        {
            "configured": True,
            "images_dir": str(images_dir),
            "prompts_file": str(prompts_file),
            "domain": domain,
            "detector": body.get("detector", "owlv2"),
            "attribute_classifier": body.get("attribute_classifier", "clip-zero-shot"),
            "device": body.get("device") or None,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "perception_dir": str(perception_dir),
            "items_by_key": {_key(it.id, it.prompt_field): it for it in items},
            "image_size_cache": {},
        }
    )
    _load_perception_file()
    _seed_human_from_automated()
    return jsonify(_state_summary())


def _number_of_objects(entry: dict | None) -> int:
    """0 if entry is None (no json / no entry for this image) or its own
    number_of_objects is None (a failed perception run) -- reads the entry's own
    stored number_of_objects field rather than recomputing from scene_graph, since
    every entry (auto- or human-perception) now stores that field itself."""
    if entry is None:
        return 0
    return entry.get("number_of_objects") or 0


def _state_summary() -> dict:
    human_by_key = _load_human_annotations()
    images = []
    for it in STATE["items_by_key"].values():
        k = _key(it.id, it.prompt_field)
        automated_entry = STATE["automated_by_key"].get(k)
        human_entry = human_by_key.get(k)
        matched = (
            _scene_graphs_match(human_entry["scene_graph"], automated_entry["scene_graph"])
            if human_entry is not None and automated_entry is not None
            else None
        )
        images.append(
            {
                "id": it.id,
                "field": it.prompt_field,
                "filename": it.image_path.name,
                "has_automated": automated_entry is not None,
                "has_human": human_entry is not None,
                "automated_number_of_objects": _number_of_objects(automated_entry),
                "human_number_of_objects": _number_of_objects(human_entry),
                "matched": matched,
                "reasonable_scene": human_entry.get("reasonable_scene") if human_entry is not None else None,
            }
        )
    images.sort(key=lambda r: (int(r["id"]) if r["id"].isdigit() else r["id"], r["field"]))
    return {
        "configured": STATE["configured"],
        "images_dir": STATE["images_dir"],
        "prompts_file": STATE["prompts_file"],
        "domain": STATE["domain"],
        "model_name": STATE["model_name"],
        "dataset_name": STATE["dataset_name"],
        "perception_dir": STATE["perception_dir"],
        "automated_perception_path": str(_perception_out_path()) if STATE["configured"] else None,
        "human_perception_path": str(_human_out_path()) if STATE["configured"] else None,
        "vocab": _domain_vocab() if STATE["configured"] else {},
        "images": images,
        "perception_job": PERCEPTION_JOB,
    }


@app.route("/api/state", methods=["GET"])
def api_state():
    if not STATE["configured"]:
        return jsonify({"configured": False})
    return jsonify(_state_summary())


# ---------------------------------------------------------------------------
# Run the (real, ML-backed) perception pipeline once over every matched image
# ---------------------------------------------------------------------------
def _run_perception_job() -> None:
    PERCEPTION_JOB.update(status="running", done=0, total=len(STATE["items_by_key"]), current=None, error=None)

    def _on_item_done(done: int, total: int, item_id: str) -> None:
        # Fires after every image (not just at the end) -- see run.py::run_perception's
        # on_item_done docstring. out_path is also rewritten each time, so a poller could
        # alternatively watch that file's mtime; this is just more direct.
        PERCEPTION_JOB.update(done=done, total=total, current=item_id)

    try:
        items = list(STATE["items_by_key"].values())
        run_perception(
            items,
            STATE["domain"],
            STATE["detector"],
            STATE["attribute_classifier"],
            STATE["device"],
            _perception_out_path(),
            on_item_done=_on_item_done,
        )
        _load_perception_file()
        _seed_human_from_automated()
        PERCEPTION_JOB.update(status="done", done=len(items))
    except Exception as e:  # noqa: BLE001
        PERCEPTION_JOB.update(status="error", error=repr(e))


@app.route("/api/run_perception", methods=["POST"])
def api_run_perception():
    if not STATE["configured"]:
        return jsonify({"error": "not configured yet"}), 400
    if PERCEPTION_JOB["status"] == "running":
        return jsonify({"error": "already running"}), 409
    thread = threading.Thread(target=_run_perception_job, daemon=True)
    thread.start()
    return jsonify(PERCEPTION_JOB)


@app.route("/api/run_perception/status", methods=["GET"])
def api_run_perception_status():
    return jsonify(PERCEPTION_JOB)


# ---------------------------------------------------------------------------
# Per-image load / save
# ---------------------------------------------------------------------------
@app.route("/api/image/<image_id>/<field>", methods=["GET"])
def api_get_image(image_id: str, field: str):
    k = _key(image_id, field)
    item = STATE["items_by_key"].get(k)
    if item is None:
        return jsonify({"error": "unknown id/field"}), 404

    w, h = _image_size(item.image_path)
    human_entry = _load_human_annotations().get(k)
    perception = STATE["automated_by_key"].get(k)

    has_human = human_entry is not None
    has_automated = perception is not None

    # These two paths are reported independently of which one the scene graph below was
    # actually loaded from, so the UI can always show "does an automated-perception
    # result exist for this image" and "does a human annotation exist for this image"
    # side by side -- not just which single source won.
    automated_perception_path = str(_perception_out_path()) if has_automated else ""
    human_perception_path = str(_human_out_path()) if has_human else ""

    # `prefer` only matters when both exist -- with only one available there's nothing to
    # choose between, so it wins regardless of what was asked for. Defaults to "human".
    prefer = request.args.get("prefer", "human")
    if has_human and has_automated:
        use = prefer if prefer in ("human", "automated") else "human"
    elif has_human:
        use = "human"
    elif has_automated:
        use = "automated"
    else:
        use = "empty"

    if use == "human":
        scene_graph = human_entry["scene_graph"]
    elif use == "automated":
        scene_graph = perception["scene_graph"]
    else:
        scene_graph = {"objects": {}, "relations": []}

    matched = _scene_graphs_match(human_entry["scene_graph"], perception["scene_graph"]) if has_human and has_automated else None

    return jsonify(
        {
            "id": image_id,
            "field": field,
            "image_width": w,
            "image_height": h,
            "prompt": item.record.prompts.get(field, ""),
            "instantiated_rule": item.record.instantiated_rule,
            "status": item.record.status,
            "source": use,
            "has_human": has_human,
            "has_automated": has_automated,
            "matched": matched,
            "automated_perception_path": automated_perception_path,
            "human_perception_path": human_perception_path,
            "automated_number_of_objects": _number_of_objects(perception),
            "human_number_of_objects": _number_of_objects(human_entry),
            # Human-only field -- automated-perception has no equivalent, and no
            # server-side default: None means "no human entry has ever recorded this
            # yet", and the UI's own session-configurable default fills the gap until
            # a save actually happens.
            "reasonable_scene": human_entry.get("reasonable_scene") if human_entry is not None else None,
            "scene_graph": scene_graph,
        }
    )


@app.route("/api/image/<image_id>/<field>/file", methods=["GET"])
def api_get_image_file(image_id: str, field: str):
    item = STATE["items_by_key"].get(_key(image_id, field))
    if item is None:
        return jsonify({"error": "unknown id/field"}), 404
    return send_file(item.image_path)


@app.route("/api/image/<image_id>/<field>", methods=["POST"])
def api_save_image(image_id: str, field: str):
    """Recompute region + relations server-side from the drawn boxes (same
    geometry perception.regions uses) and update this image's entry in the
    combined <dataset>_human-perception.json -- every other image's entry is
    left untouched, and <dataset>_auto-perception.json is never rewritten by
    this endpoint."""
    k = _key(image_id, field)
    item = STATE["items_by_key"].get(k)
    if item is None:
        return jsonify({"error": "unknown id/field"}), 404

    body = request.get_json(force=True)
    raw_objects = body.get("objects", [])
    w, h = _image_size(item.image_path)

    detected: list[DetectedObject] = []
    for obj_id, obj in enumerate(raw_objects):
        x0, y0, x1, y1 = obj["bbox"]
        # "size" is dropped even if a client sends it -- see _domain_vocab.
        properties = {k2: v for k2, v in obj.get("properties", {}).items() if v and k2 != "size"}
        bbox = BBox(x0=x0, y0=y0, x1=x1, y1=y1, label=properties.get("shape", ""), score=1.0)
        cx, cy = bbox_center(bbox)
        region = region_of(cx, cy, w, h)
        detected.append(DetectedObject(obj_id=obj_id, bbox=bbox, properties=properties, region=region))

    scene_graph = to_graph_dict(detected)
    # Same field names as PerceptionResult (common/types.py) for everything through
    # number_of_objects -- id, prompt_field, image_path, prompt, instantiated_rule, status,
    # number_of_objects, scene_graph -- so the two files are structurally identical
    # wherever the concept is shared, and neither file's entries reference the other
    # file's path (that used to be here as automated_perception_path/source_perception_file;
    # the API layer still reports each file's own path, in /api/state and per-image GET,
    # but it's no longer duplicated into the saved data itself).
    payload = {
        "id": image_id,
        "prompt_field": field,
        "image_path": str(item.image_path),
        "prompt": item.record.prompts.get(field, ""),
        "instantiated_rule": item.record.instantiated_rule,
        "status": item.record.status,
        "number_of_objects": len(detected),
        "scene_graph": scene_graph,
        # Human-only judgment call ("is this a reasonable/sensible generated scene at
        # all"), no automated-perception equivalent. The client always sends its
        # current toggle value (either this image's own saved value, or its session
        # default if this is the first save for this image) -- body.get(...) here is
        # just a defensive fallback if an older client omits it.
        "reasonable_scene": body.get("reasonable_scene"),
        "annotated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save_human_annotation(image_id, field, payload)
    return jsonify({**payload, "human_perception_path": str(_human_out_path())})


# ---------------------------------------------------------------------------
# Browse page -- a second, independent UI (at /browse, and the landing page once
# a password is set) for opening any *_human-perception.json file directly from
# a chain of dropdowns (rather than via the images_dir/prompts_file setup form
# above) and editing its entries. Deliberately stateless -- no STATE involvement,
# every request is fully identified by its query params -- so it never interacts
# with or is affected by whatever batch is (or isn't) configured on the main page.
# Shows only what the user asked for: no prompt, no constraint, no
# automated-perception path or toggle, no dataset status -- just Model / Dataset /
# Image, and the same box/property editor as the main page.
#
# Layout on disk, model and dataset both literal folder/file names, not derived
# from anything stored inside a JSON entry:
#   <data>/evaluation/<model>/<dataset>_human-perception.json
#   <data>/generated_images/<model>/<dataset>/<id>-<field>.png
# Deliberately does NOT trust each entry's own "image_path" field to find the
# image file -- that path was written by whatever machine originally ran
# perception/saved the annotation, and won't exist as-is in a different
# environment (e.g. a local absolute path baked into a JSON that's since been
# uploaded to a Cloud Run deployment, where the mounted data root is different).
# The image path is always recomputed fresh from the selected model/dataset/id/
# field instead.
# ---------------------------------------------------------------------------
_DATA_ROOT = _ROOT.parent / "data"


def _discover_browse_models() -> list[str]:
    evaluation_dir = _DATA_ROOT / "evaluation"
    if not evaluation_dir.is_dir():
        return []
    return sorted(p.name for p in evaluation_dir.iterdir() if p.is_dir())


def _discover_browse_datasets(model: str) -> list[Path]:
    """Every *_human-perception.json directly inside evaluation/<model>/ -- not
    recursive, model must be one of _discover_browse_models()'s own results (checked
    by the caller before this is trusted for anything)."""
    if model not in _discover_browse_models():
        return []
    return sorted((_DATA_ROOT / "evaluation" / model).glob("*_human-perception.json"))


def _resolve_browse_path(raw: str) -> Path | None:
    """Only ever accept a path that's literally one of the files
    _discover_browse_datasets() would find for its own parent folder's name as the
    model -- the query string is untrusted client input, and this must never become
    an arbitrary-file read/write endpoint."""
    if not raw:
        return None
    try:
        candidate = Path(raw).resolve()
    except OSError:
        return None
    return candidate if candidate in _discover_browse_datasets(candidate.parent.name) else None


def _browse_image_path(path: Path, image_id: str, field: str) -> Path:
    """The model/dataset a human-perception.json belongs to are just its own
    location (parent folder name / filename minus suffix) -- from there the image
    file's path is recomputed fresh, per the module docstring above, not read from
    the entry's own stored (possibly foreign-environment) image_path field."""
    model = path.parent.name
    dataset = path.stem.removesuffix("_human-perception")
    return _DATA_ROOT / "generated_images" / model / dataset / f"{image_id}-{field}.png"


def _load_browse_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_browse_entry(data: dict, image_id: str, field: str) -> dict | None:
    return next(
        (a for a in data.get("annotations", []) if a["id"] == image_id and a["prompt_field"] == field),
        None,
    )


@app.route("/api/browse/models", methods=["GET"])
def api_browse_models():
    return jsonify(_discover_browse_models())


@app.route("/api/browse/datasets", methods=["GET"])
def api_browse_datasets():
    model = request.args.get("model", "")
    out = []
    for path in _discover_browse_datasets(model):
        try:
            count = len(_load_browse_file(path).get("annotations", []))
        except (json.JSONDecodeError, OSError):
            count = None
        out.append({"path": str(path), "dataset": path.stem.removesuffix("_human-perception"), "count": count})
    return jsonify(out)


@app.route("/api/browse/file", methods=["GET"])
def api_browse_file():
    path = _resolve_browse_path(request.args.get("path", ""))
    if path is None:
        return jsonify({"error": "unknown file"}), 404
    data = _load_browse_file(path)
    images = [
        {
            "id": a["id"],
            "field": a["prompt_field"],
            "number_of_objects": len(a.get("scene_graph", {}).get("objects", {})),
            "reasonable_scene": a.get("reasonable_scene"),
        }
        for a in data.get("annotations", [])
    ]
    images.sort(key=lambda r: (int(r["id"]) if r["id"].isdigit() else r["id"], r["field"]))
    domain = data.get("domain", "clevr")
    domain_module = load_domain(domain)
    vocab = {k: v for k, v in domain_module.PROPERTIES.items() if k != "size"}  # see _domain_vocab above
    return jsonify({"domain": domain, "vocab": vocab, "images": images})


@app.route("/api/browse/image", methods=["GET"])
def api_browse_get_image():
    path = _resolve_browse_path(request.args.get("path", ""))
    if path is None:
        return jsonify({"error": "unknown file"}), 404
    image_id, field = request.args.get("id", ""), request.args.get("field", "")
    entry = _find_browse_entry(_load_browse_file(path), image_id, field)
    if entry is None:
        return jsonify({"error": "unknown id/field"}), 404
    w, h = _image_size(_browse_image_path(path, image_id, field))
    return jsonify(
        {
            "image_width": w,
            "image_height": h,
            "reasonable_scene": entry.get("reasonable_scene"),
            "scene_graph": entry["scene_graph"],
        }
    )


@app.route("/api/browse/image/file", methods=["GET"])
def api_browse_get_image_file():
    path = _resolve_browse_path(request.args.get("path", ""))
    if path is None:
        return jsonify({"error": "unknown file"}), 404
    image_id, field = request.args.get("id", ""), request.args.get("field", "")
    if _find_browse_entry(_load_browse_file(path), image_id, field) is None:
        return jsonify({"error": "unknown id/field"}), 404
    return send_file(_browse_image_path(path, image_id, field))


@app.route("/api/browse/image", methods=["POST"])
def api_browse_save_image():
    path = _resolve_browse_path(request.args.get("path", ""))
    if path is None:
        return jsonify({"error": "unknown file"}), 404
    image_id, field = request.args.get("id", ""), request.args.get("field", "")

    data = _load_browse_file(path)
    entry = _find_browse_entry(data, image_id, field)
    if entry is None:
        return jsonify({"error": "unknown id/field"}), 404

    body = request.get_json(force=True)
    raw_objects = body.get("objects", [])
    w, h = _image_size(_browse_image_path(path, image_id, field))

    detected: list[DetectedObject] = []
    for obj_id, obj in enumerate(raw_objects):
        x0, y0, x1, y1 = obj["bbox"]
        properties = {k2: v for k2, v in obj.get("properties", {}).items() if v and k2 != "size"}
        bbox = BBox(x0=x0, y0=y0, x1=x1, y1=y1, label=properties.get("shape", ""), score=1.0)
        cx, cy = bbox_center(bbox)
        region = region_of(cx, cy, w, h)
        detected.append(DetectedObject(obj_id=obj_id, bbox=bbox, properties=properties, region=region))

    entry["scene_graph"] = to_graph_dict(detected)
    entry["number_of_objects"] = len(detected)
    entry["reasonable_scene"] = body.get("reasonable_scene")
    entry["annotated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    write_json(path, data)
    return jsonify({"number_of_objects": entry["number_of_objects"], "reasonable_scene": entry["reasonable_scene"]})


@app.route("/browse")
def browse_page():
    return send_from_directory(app.static_folder, "browse.html")


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
@app.route("/setup")
def setup_page():
    """The original images_dir/prompts_file-driven page -- still here, just no
    longer the landing page (see index() below)."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/")
def index():
    # /browse is the landing page -- lands there straight after the password
    # prompt (when CCIG_HUMAN_EVAL_PASSWORD is set) since that's the page actually
    # used day to day. The original setup-form page is still reachable at /setup.
    return redirect("/browse")


def main() -> None:
    parser = argparse.ArgumentParser(description="Human scene-graph annotation tool.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if _AUTH_PASSWORD:
        print("[auth] password protection is ON (CCIG_HUMAN_EVAL_PASSWORD is set)")
    else:
        print("[auth] password protection is OFF -- set CCIG_HUMAN_EVAL_PASSWORD before exposing this via a public tunnel")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

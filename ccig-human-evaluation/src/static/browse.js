"use strict";

// Independent of app.js -- this page edits a *_human-perception.json file picked
// directly from a dropdown, not a STATE-backed images_dir/prompts_file batch, and
// has no concept of automated perception at all (no toggle, no read-only view, no
// "matched" flag). The canvas/box-editing logic below intentionally mirrors app.js's
// (same interaction model), but isn't shared code with it -- see browse.css's header
// comment for why.

const PROPERTY_ORDER = ["shape", "color", "material"]; // "size" deliberately excluded, same as app.js
const HANDLE_PX = 8; // resize-handle hit radius, in *display* pixels

const el = (id) => document.getElementById(id);

const state = {
  vocab: {},
  images: [], // [{id, field, number_of_objects, reasonable_scene}]
  currentModel: "",
  currentFile: "", // selected human-perception.json path
  currentKey: null, // {id, field}
  natural: { w: 1, h: 1 },
  scale: 1,
  objects: [], // [{properties: {shape,color,material}, bbox: [x0,y0,x1,y1]}]
  selected: -1,
  img: new Image(),
  reasonableScene: true,
  reasonableSceneRecorded: false, // has that value actually been saved, or is it just the in-memory default
};

let drag = null; // {mode: "new"|"move"|"resize", ...}
let saveTimer = null;

// ---------------------------------------------------------------------
// Model -> dataset -> image picker. Model = a literal folder name under
// data/evaluation/; dataset = a literal *_human-perception.json filename inside
// that folder (see app.py's _discover_browse_models/_discover_browse_datasets) --
// two separate round trips, one per dropdown level, rather than one big fetch
// grouped client-side, since a deployment could have many models/datasets.
// ---------------------------------------------------------------------
async function loadModelList() {
  const res = await fetch("/api/browse/models");
  const models = await res.json();
  const sel = el("model-select");
  sel.innerHTML = '<option value="">-- choose a model --</option>';
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    sel.appendChild(opt);
  }
}

el("model-select").addEventListener("change", async (e) => {
  state.currentModel = e.target.value;
  state.currentFile = "";
  state.currentKey = null;
  el("picker-error").textContent = "";
  el("workspace-panel").classList.add("hidden");

  const fileSel = el("file-select");
  const imageSel = el("image-select");
  imageSel.innerHTML = '<option value="">-- choose a dataset first --</option>';
  imageSel.disabled = true;

  if (!state.currentModel) {
    fileSel.innerHTML = '<option value="">-- choose a model first --</option>';
    fileSel.disabled = true;
    return;
  }
  const res = await fetch(`/api/browse/datasets?model=${encodeURIComponent(state.currentModel)}`);
  const datasets = await res.json();
  fileSel.innerHTML = '<option value="">-- choose a dataset --</option>';
  for (const f of datasets) {
    const opt = document.createElement("option");
    opt.value = f.path;
    opt.textContent = f.count === null ? `${f.dataset}  [unreadable]` : `${f.dataset}  (${f.count} images)`;
    fileSel.appendChild(opt);
  }
  fileSel.disabled = false;
});

el("file-select").addEventListener("change", async (e) => {
  const path = e.target.value;
  state.currentFile = path;
  state.currentKey = null;
  el("picker-error").textContent = "";
  el("workspace-panel").classList.add("hidden");
  const imageSel = el("image-select");
  if (!path) {
    imageSel.innerHTML = '<option value="">-- choose a dataset first --</option>';
    imageSel.disabled = true;
    return;
  }
  const res = await fetch(`/api/browse/file?path=${encodeURIComponent(path)}`);
  const data = await res.json();
  if (!res.ok) {
    el("picker-error").textContent = data.error || "failed to load file";
    return;
  }
  state.vocab = data.vocab || {};
  state.images = data.images || [];
  populateImageSelect();
  imageSel.disabled = state.images.length === 0;
  if (state.images.length) {
    const first = state.images[0];
    imageSel.value = `${first.id}::${first.field}`;
    loadImage(first.id, first.field);
  }
});

function populateImageSelect() {
  const sel = el("image-select");
  sel.innerHTML = "";
  for (const im of state.images) {
    const opt = document.createElement("option");
    opt.value = `${im.id}::${im.field}`;
    const mark = im.reasonable_scene === true ? " ✓" : im.reasonable_scene === false ? " ✗" : "";
    opt.textContent = `${im.id}-${im.field}  [${im.number_of_objects} objects${mark}]`;
    sel.appendChild(opt);
  }
}

el("image-select").addEventListener("change", (e) => {
  const [id, field] = e.target.value.split("::");
  loadImage(id, field);
});

// ---------------------------------------------------------------------
// Load / render one image
// ---------------------------------------------------------------------
async function loadImage(id, field) {
  state.currentKey = { id, field };
  el("workspace-panel").classList.remove("hidden");

  const res = await fetch(`/api/browse/image?path=${encodeURIComponent(state.currentFile)}&id=${encodeURIComponent(id)}&field=${encodeURIComponent(field)}`);
  const data = await res.json();
  if (!res.ok) {
    el("picker-error").textContent = data.error || "failed to load image";
    return;
  }
  el("picker-error").textContent = "";

  // reasonableSceneRecorded tracks whether a value has actually been saved, separately
  // from the in-memory default -- without it the toggle would show "Yes" as already
  // selected the moment an image loads (default is true), even though nothing's been
  // saved, which is how true was silently never making it into human-perception.json
  // while false always did (clicking "No" changes the value and saves; leaving the
  // seemingly-already-selected "Yes" alone triggers nothing).
  state.reasonableScene = data.reasonable_scene ?? true;
  state.reasonableSceneRecorded = data.reasonable_scene !== null && data.reasonable_scene !== undefined;
  renderReasonableToggle();

  state.natural = { w: data.image_width, h: data.image_height };
  state.objects = objectsDictToArray(data.scene_graph.objects || {});
  state.selected = -1;

  state.img = new Image();
  state.img.onload = () => {
    layoutCanvas();
    render();
  };
  const params = `path=${encodeURIComponent(state.currentFile)}&id=${encodeURIComponent(id)}&field=${encodeURIComponent(field)}`;
  state.img.src = `/api/browse/image/file?${params}&t=${Date.now()}`;

  renderObjectList();
  renderEditor();
}

function renderReasonableToggle() {
  const yesBtn = el("reasonable-yes");
  const noBtn = el("reasonable-no");
  yesBtn.classList.toggle("active", state.reasonableSceneRecorded && state.reasonableScene === true);
  noBtn.classList.toggle("active", state.reasonableSceneRecorded && state.reasonableScene === false);
}

for (const btn of [el("reasonable-yes"), el("reasonable-no")]) {
  btn.addEventListener("click", () => {
    state.reasonableScene = btn.dataset.value === "true";
    state.reasonableSceneRecorded = true;
    renderReasonableToggle();
    scheduleSave();
  });
}

function objectsDictToArray(dict) {
  return Object.keys(dict)
    .sort((a, b) => Number(a) - Number(b))
    .map((k) => {
      const o = dict[k];
      const properties = {};
      for (const p of PROPERTY_ORDER) if (o[p]) properties[p] = o[p];
      return { properties, bbox: o.bbox.slice(0, 4) };
    });
}

function layoutCanvas() {
  const canvas = el("canvas");
  const containerWidth = canvas.parentElement.clientWidth;
  const dispW = Math.min(containerWidth, 900);
  state.scale = dispW / state.natural.w;
  canvas.width = dispW;
  canvas.height = state.natural.h * state.scale;
}

window.addEventListener("resize", () => {
  if (state.img.src) {
    layoutCanvas();
    render();
  }
});

// ---------------------------------------------------------------------
// Canvas rendering
// ---------------------------------------------------------------------
function render() {
  const canvas = el("canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.img.complete && state.img.naturalWidth) {
    ctx.drawImage(state.img, 0, 0, canvas.width, canvas.height);
  }
  state.objects.forEach((obj, i) => drawBox(ctx, obj, i === state.selected));
  if (drag && drag.mode === "new") drawRect(ctx, drag.rectDisplay, "#2563eb", false);
}

function toDisplay([x0, y0, x1, y1]) {
  const s = state.scale;
  return [x0 * s, y0 * s, x1 * s, y1 * s];
}

// Mirrors perception/regions.py::region_of -- a 2x2 quadrant of the image, purely a
// function of the box's center, never hand-entered.
function regionOf([x0, y0, x1, y1]) {
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  const left = cx < state.natural.w / 2;
  const top = cy < state.natural.h / 2;
  if (top && left) return "r0";
  if (top && !left) return "r1";
  if (!top && left) return "r2";
  return "r3";
}

function objectLabel(obj) {
  const props = PROPERTY_ORDER.filter((p) => obj.properties[p]).map((p) => obj.properties[p]).join(":") || "(unset)";
  return `${props} [${regionOf(obj.bbox)}]`;
}

function drawBox(ctx, obj, selected) {
  const rect = toDisplay(obj.bbox);
  drawRect(ctx, rect, selected ? "#2563eb" : "#f59e0b", selected);
  const label = objectLabel(obj);
  ctx.font = "12px sans-serif";
  const [x0, y0] = rect;
  const textW = ctx.measureText(label).width + 6;
  ctx.fillStyle = selected ? "#2563eb" : "#f59e0b";
  ctx.fillRect(x0, Math.max(0, y0 - 16), textW, 16);
  ctx.fillStyle = "#fff";
  ctx.fillText(label, x0 + 3, Math.max(12, y0 - 4));
}

function drawRect(ctx, [x0, y0, x1, y1], color, withHandles) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  if (withHandles) {
    ctx.fillStyle = color;
    for (const [hx, hy] of cornersOf([x0, y0, x1, y1])) {
      ctx.fillRect(hx - HANDLE_PX / 2, hy - HANDLE_PX / 2, HANDLE_PX, HANDLE_PX);
    }
  }
}

function cornersOf([x0, y0, x1, y1]) {
  return [
    [x0, y0],
    [x1, y0],
    [x0, y1],
    [x1, y1],
  ];
}

// ---------------------------------------------------------------------
// Mouse interaction (all math in *display* pixels; convert to natural
// pixels only when writing into state.objects)
// ---------------------------------------------------------------------
function canvasPos(e) {
  const r = el("canvas").getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}

el("canvas").addEventListener("mousedown", (e) => {
  const [mx, my] = canvasPos(e);

  if (state.selected >= 0) {
    const rect = toDisplay(state.objects[state.selected].bbox);
    const corners = cornersOf(rect);
    for (let ci = 0; ci < corners.length; ci++) {
      const [hx, hy] = corners[ci];
      if (Math.abs(mx - hx) <= HANDLE_PX && Math.abs(my - hy) <= HANDLE_PX) {
        drag = { mode: "resize", corner: ci, start: rect.slice() };
        return;
      }
    }
    if (mx >= rect[0] && mx <= rect[2] && my >= rect[1] && my <= rect[3]) {
      drag = { mode: "move", offset: [mx - rect[0], my - rect[1]], size: [rect[2] - rect[0], rect[3] - rect[1]] };
      return;
    }
  }

  // click on another existing box -> select it, no drag
  for (let i = state.objects.length - 1; i >= 0; i--) {
    const rect = toDisplay(state.objects[i].bbox);
    if (mx >= rect[0] && mx <= rect[2] && my >= rect[1] && my <= rect[3]) {
      selectObject(i);
      return;
    }
  }

  // empty space -> start drawing a new box
  state.selected = -1;
  renderObjectList();
  renderEditor();
  drag = { mode: "new", origin: [mx, my], rectDisplay: [mx, my, mx, my] };
});

el("canvas").addEventListener("mousemove", (e) => {
  if (!drag) return;
  const [mx, my] = canvasPos(e);
  const canvas = el("canvas");
  const cx = Math.max(0, Math.min(canvas.width, mx));
  const cy = Math.max(0, Math.min(canvas.height, my));

  if (drag.mode === "new") {
    drag.rectDisplay = [Math.min(drag.origin[0], cx), Math.min(drag.origin[1], cy), Math.max(drag.origin[0], cx), Math.max(drag.origin[1], cy)];
  } else if (drag.mode === "move") {
    const [w, h] = drag.size;
    let x0 = cx - drag.offset[0];
    let y0 = cy - drag.offset[1];
    x0 = Math.max(0, Math.min(canvas.width - w, x0));
    y0 = Math.max(0, Math.min(canvas.height - h, y0));
    setSelectedDisplayRect([x0, y0, x0 + w, y0 + h]);
  } else if (drag.mode === "resize") {
    let [x0, y0, x1, y1] = drag.start;
    if (drag.corner === 0) [x0, y0] = [cx, cy];
    if (drag.corner === 1) [x1, y0] = [cx, cy];
    if (drag.corner === 2) [x0, y1] = [cx, cy];
    if (drag.corner === 3) [x1, y1] = [cx, cy];
    setSelectedDisplayRect([Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)]);
  }
  render();
});

window.addEventListener("mouseup", () => {
  if (!drag) return;
  if (drag.mode === "new") {
    const [x0, y0, x1, y1] = drag.rectDisplay;
    if (x1 - x0 >= 6 && y1 - y0 >= 6) {
      const bbox = displayRectToNatural([x0, y0, x1, y1]);
      state.objects.push({ properties: {}, bbox });
      selectObject(state.objects.length - 1);
      scheduleSave();
    } else {
      render();
    }
  } else if (drag.mode === "move" || drag.mode === "resize") {
    scheduleSave();
  }
  drag = null;
});

function displayRectToNatural([x0, y0, x1, y1]) {
  const s = state.scale;
  return [x0 / s, y0 / s, x1 / s, y1 / s];
}

function setSelectedDisplayRect(rectDisplay) {
  if (state.selected < 0) return;
  state.objects[state.selected].bbox = displayRectToNatural(rectDisplay);
}

document.addEventListener("keydown", (e) => {
  if ((e.key === "Delete" || e.key === "Backspace") && state.selected >= 0 && document.activeElement.tagName !== "SELECT") {
    deleteSelected();
  }
});

// ---------------------------------------------------------------------
// Selection + right-hand editor
// ---------------------------------------------------------------------
function selectObject(i) {
  state.selected = i;
  render();
  renderObjectList();
  renderEditor();
}

function renderObjectList() {
  const list = el("object-list");
  list.innerHTML = "";
  if (state.objects.length === 0) {
    const empty = document.createElement("p");
    empty.className = "hint no-objects";
    empty.textContent = "No objects.";
    list.appendChild(empty);
  }
  state.objects.forEach((obj, i) => {
    const chip = document.createElement("div");
    chip.className = "object-chip" + (i === state.selected ? " selected" : "");
    chip.innerHTML = `<span>#${i} ${objectLabel(obj)}</span>`;
    chip.addEventListener("click", () => selectObject(i));
    list.appendChild(chip);
  });
}

function renderEditor() {
  const editor = el("object-editor");
  if (state.selected < 0) {
    editor.classList.add("hidden");
    el("no-selection-hint").classList.remove("hidden");
    return;
  }
  editor.classList.remove("hidden");
  el("no-selection-hint").classList.add("hidden");
  const obj = state.objects[state.selected];
  const container = el("property-fields");
  container.innerHTML = "";
  const propsInDomain = PROPERTY_ORDER.filter((p) => state.vocab[p]);
  for (const p of propsInDomain) {
    const field = document.createElement("div");
    field.className = "property-field";
    const label = document.createElement("label");
    label.textContent = p;
    const select = document.createElement("select");
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "– (not set)";
    select.appendChild(blank);
    // Distinct from the blank/unset placeholder above -- this is a real, saved
    // value ("this property was deliberately observed as absent/unclear"), not
    // "nothing chosen yet". select.value is a non-empty string either way, so the
    // change handler below already stores it like any other vocab value.
    const none = document.createElement("option");
    none.value = "Unknown";
    none.textContent = "Unknown";
    select.appendChild(none);
    for (const v of state.vocab[p]) {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      select.appendChild(opt);
    }
    select.value = obj.properties[p] || "";
    select.addEventListener("change", () => {
      if (select.value) obj.properties[p] = select.value;
      else delete obj.properties[p];
      render();
      renderObjectList();
      scheduleSave();
    });
    field.appendChild(label);
    field.appendChild(select);
    container.appendChild(field);
  }

  const regionField = document.createElement("div");
  regionField.className = "property-field";
  regionField.innerHTML = `<label>region (auto)</label><span class="region-readout">${regionOf(obj.bbox)}</span>`;
  container.appendChild(regionField);
}

el("delete-object").addEventListener("click", deleteSelected);

function deleteSelected() {
  if (state.selected < 0) return;
  state.objects.splice(state.selected, 1);
  state.selected = -1;
  render();
  renderObjectList();
  renderEditor();
  scheduleSave();
}

el("mark-empty").addEventListener("click", () => {
  if (state.objects.length > 0 && !confirm(`This image currently has ${state.objects.length} object(s) drawn. Clear them and save this as an empty scene (0 objects)?`)) {
    return;
  }
  state.objects = [];
  state.selected = -1;
  render();
  renderObjectList();
  renderEditor();
  saveNow(); // explicit, deliberate action -- save immediately rather than the usual debounce
});

// ---------------------------------------------------------------------
// Autosave
// ---------------------------------------------------------------------
function scheduleSave() {
  setSaveIndicator("saving");
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveNow, 400);
}

async function saveNow() {
  if (!state.currentFile || !state.currentKey) return;
  const { id, field } = state.currentKey;
  const body = {
    objects: state.objects.map((o) => ({ bbox: o.bbox, properties: o.properties })),
    reasonable_scene: state.reasonableScene,
  };
  const params = `path=${encodeURIComponent(state.currentFile)}&id=${encodeURIComponent(id)}&field=${encodeURIComponent(field)}`;
  const res = await fetch(`/api/browse/image?${params}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (res.ok) {
    const data = await res.json();
    setSaveIndicator("saved");
    // Whatever reasonable_scene held (an explicit choice or still just the in-memory
    // default) is genuinely on disk now -- reflect that in the toggle.
    state.reasonableSceneRecorded = true;
    renderReasonableToggle();
    const im = state.images.find((x) => x.id === id && x.field === field);
    if (im) {
      im.number_of_objects = data.number_of_objects;
      im.reasonable_scene = data.reasonable_scene;
    }
    populateImageSelect();
    // re-select the current image in the (just rebuilt) dropdown
    el("image-select").value = `${id}::${field}`;
  } else {
    setSaveIndicator("error");
  }
}

function setSaveIndicator(status) {
  const s = el("save-indicator");
  s.className = "save-indicator " + status;
  s.textContent = status === "saving" ? "Saving…" : status === "saved" ? "Saved" : status === "error" ? "Save failed" : "";
}

// ---------------------------------------------------------------------
loadModelList();

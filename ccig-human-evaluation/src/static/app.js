"use strict";

const PROPERTY_ORDER = ["shape", "color", "material"]; // "size" deliberately excluded -- never shown, edited, or saved
const HANDLE_PX = 8; // resize-handle hit radius, in *display* pixels

const el = (id) => document.getElementById(id);

const state = {
  vocab: {},
  images: [],
  currentKey: null, // {id, field}
  natural: { w: 1, h: 1 },
  scale: 1,
  objects: [], // [{properties: {shape,color,material,size}, bbox: [x0,y0,x1,y1]}]
  selected: -1,
  img: new Image(),
  viewPreference: "human", // "human" | "automated" -- only decides anything when both exist for the current image
  readOnly: false, // true whenever the *displayed* source (state's "source", not viewPreference) is "automated"
  defaultReasonableScene: true, // set once on the setup form, before any image is loaded
  reasonableScene: true, // this image's current value -- data.reasonable_scene if saved before, else the default above
  reasonableSceneRecorded: false, // has that value actually been saved, or is it just the in-memory default
};

let drag = null; // {mode: "new"|"move"|"resize", ...}
let saveTimer = null;

// ---------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------
el("setup-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = new FormData(e.target);
  const body = Object.fromEntries(form.entries());
  el("setup-error").textContent = "";
  // Client-only setting (the server has no use for it) -- read it here and drop it
  // before sending, rather than have the backend silently ignore an unknown field.
  state.defaultReasonableScene = body.default_reasonable_scene === "true";
  delete body.default_reasonable_scene;
  const res = await fetch("/api/setup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    el("setup-error").textContent = data.error || "setup failed";
    return;
  }
  state.currentKey = null; // force applyState below to load this batch's first image,
  // not silently keep showing whatever image was open in the previous batch
  applyState(data);
  el("setup-panel").classList.add("hidden");
  el("main-panel").classList.remove("hidden");
});

el("change-setup").addEventListener("click", () => {
  // Also reset here (not just on submit) so the *next* setup submission is
  // guaranteed to load fresh even if this batch happened to share an id/field
  // with the new one -- applyState only auto-loads when currentKey is null.
  state.currentKey = null;
  el("main-panel").classList.add("hidden");
  el("setup-panel").classList.remove("hidden");
});

async function refreshState() {
  const res = await fetch("/api/state");
  const data = await res.json();
  if (data.configured) {
    applyState(data);
    el("setup-panel").classList.add("hidden");
    el("main-panel").classList.remove("hidden");
  }
}

function applyState(data) {
  state.vocab = data.vocab || {};
  state.images = data.images || [];
  el("ctx-model").textContent = data.model_name || "";
  el("ctx-dataset").textContent = data.dataset_name || "";
  populateImageSelect();
  updatePerceptionStatus(data.perception_job);
  if (!state.currentKey && state.images.length) {
    const first = state.images[0];
    loadImage(first.id, first.field);
  }
}

function populateImageSelect() {
  const sel = el("image-select");
  const prevValue = sel.value;
  sel.innerHTML = "";
  for (const im of state.images) {
    const opt = document.createElement("option");
    opt.value = `${im.id}::${im.field}`;
    const matchMark = im.matched === true ? " ✓" : im.matched === false ? " ✎" : "";
    const marks = `${im.has_human ? `h:${im.human_number_of_objects}` : ""} ${im.has_automated ? `p:${im.automated_number_of_objects}` : ""}`.trim();
    opt.textContent = `${im.id}-${im.field}${marks ? "  [" + marks + matchMark + "]" : ""}`;
    sel.appendChild(opt);
  }
  if (prevValue) sel.value = prevValue;
}

el("image-select").addEventListener("change", (e) => {
  const [id, field] = e.target.value.split("::");
  loadImage(id, field);
});

// ---------------------------------------------------------------------
// Run perception
// ---------------------------------------------------------------------
el("run-perception").addEventListener("click", async () => {
  const res = await fetch("/api/run_perception", { method: "POST" });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || "failed to start perception job");
    return;
  }
  pollPerceptionStatus();
});

async function pollPerceptionStatus() {
  const res = await fetch("/api/run_perception/status");
  const data = await res.json();
  updatePerceptionStatus(data);
  if (data.status === "running") {
    setTimeout(pollPerceptionStatus, 1500);
  } else if (data.status === "done") {
    await refreshState(); // pick up has_automated markers + prefill for images without human edits yet
    if (state.currentKey) {
      const cur = state.images.find((im) => im.id === state.currentKey.id && im.field === state.currentKey.field);
      if (cur && !cur.has_human) loadImage(cur.id, cur.field);
    }
  }
}

function updatePerceptionStatus(job) {
  if (!job) return;
  const s = el("perception-status");
  if (job.status === "running") s.textContent = `running perception... (${job.done}/${job.total}${job.current ? ", last: " + job.current : ""})`;
  else if (job.status === "done") s.textContent = `perception done (${job.done} images)`;
  else if (job.status === "error") s.textContent = `perception failed: ${job.error}`;
  else s.textContent = "";
}

// ---------------------------------------------------------------------
// Load / render one image
// ---------------------------------------------------------------------
async function loadImage(id, field) {
  state.currentKey = { id, field };
  const data = await fetchImageMeta(id, field);

  state.natural = { w: data.image_width, h: data.image_height };
  state.objects = objectsDictToArray(data.scene_graph.objects || {});
  state.selected = -1;

  state.img = new Image();
  state.img.onload = () => {
    layoutCanvas();
    render();
  };
  state.img.src = `/api/image/${id}/${field}/file?t=${Date.now()}`;

  renderObjectList();
  renderEditor();
}

// Fetches + applies everything about the current image except the canvas/objects
// themselves (prompt, constraint, status, both paths, both object counts, the
// human/automated toggle state, and the matched/different flag). Used by loadImage
// (full load) and by saveNow (so the matched flag and counts stay correct right after
// every autosave, without re-fetching the PNG or resetting your in-progress edit).
async function fetchImageMeta(id, field) {
  const res = await fetch(`/api/image/${id}/${field}?prefer=${state.viewPreference}`);
  const data = await res.json();

  el("ctx-prompt").textContent = data.prompt || "";
  el("ctx-rule").textContent = data.instantiated_rule || "";
  setPathField("ctx-automated-path", data.automated_perception_path);
  setPathField("ctx-human-path", data.human_perception_path);
  el("ctx-automated-count").textContent = `${data.automated_number_of_objects} object${data.automated_number_of_objects === 1 ? "" : "s"}`;
  el("ctx-human-count").textContent = `${data.human_number_of_objects} object${data.human_number_of_objects === 1 ? "" : "s"}`;
  updateViewToggle(data);

  // Automated is a read-only view -- perception's own result was never meant to be
  // hand-edited (that would silently corrupt it as a baseline to compare against);
  // only "human" is ever writable. renderEditor()/canvas handlers below all check this.
  state.readOnly = data.source === "automated";
  el("canvas").classList.toggle("readonly", state.readOnly);
  el("canvas-hint").textContent = state.readOnly
    ? "Viewing automated perception (read-only) -- switch to Human to edit."
    : "Drag on empty space to add an object box. Click a box to select it; drag its corners to resize, drag its body to move. Delete/Backspace removes the selected box.";

  // null/undefined means this image's human entry has never recorded this (either no
  // human entry yet, or it was auto-seeded from automated, which has no such field) --
  // fall back to the session default until an actual save writes a real value.
  // reasonableSceneRecorded tracks whether that's *actually happened* yet, separately
  // from the in-memory default: without it, the toggle would show "Yes" as already
  // selected the moment an image loads (since the default is true), even though
  // nothing's been saved -- easy to mistake for "already recorded" and skip past,
  // which is exactly how true silently never made it into human-perception.json
  // while false always did (clicking "No" changes the value and triggers a save;
  // leaving the seemingly-already-selected "Yes" alone triggers nothing).
  state.reasonableScene = data.reasonable_scene ?? state.defaultReasonableScene;
  state.reasonableSceneRecorded = data.reasonable_scene !== null && data.reasonable_scene !== undefined;
  renderReasonableToggle();

  return data;
}

// A reusable Yes/No toggle for human-only true/false judgment calls (currently just
// reasonable_scene, kept generic in case another one is added later).
function makeBoolToggle(yesId, noId, getValue, setValue, getRecorded, setRecorded) {
  function render() {
    const yesBtn = el(yesId);
    const noBtn = el(noId);
    const value = getValue();
    const recorded = getRecorded();
    yesBtn.classList.toggle("active", recorded && value === true);
    noBtn.classList.toggle("active", recorded && value === false);
    yesBtn.disabled = state.readOnly;
    noBtn.disabled = state.readOnly;
  }
  for (const btn of [el(yesId), el(noId)]) {
    btn.addEventListener("click", () => {
      if (state.readOnly) return;
      setValue(btn.dataset.value === "true");
      setRecorded(true);
      render();
      scheduleSave();
    });
  }
  return render;
}

const renderReasonableToggle = makeBoolToggle(
  "reasonable-yes", "reasonable-no",
  () => state.reasonableScene, (v) => { state.reasonableScene = v; },
  () => state.reasonableSceneRecorded, (v) => { state.reasonableSceneRecorded = v; }
);

// "source" (what actually got displayed) can differ from state.viewPreference when
// only one of human/automated exists for this particular image -- that single one wins
// regardless of the toggle, per the API's fallback rule.
function updateViewToggle(data) {
  const humanBtn = el("view-human");
  const automatedBtn = el("view-automated");
  humanBtn.disabled = !data.has_human;
  automatedBtn.disabled = !data.has_automated;
  // source is "empty" when neither file has this image yet -- that's still "human
  // mode" (a fresh box you draw always saves to the human file), so Human should
  // read as the active side even though there's nothing to toggle to yet.
  humanBtn.classList.toggle("active", data.source === "human" || data.source === "empty");
  automatedBtn.classList.toggle("active", data.source === "automated");
  el("view-warning").classList.toggle("hidden", !(data.source === "automated" && data.has_human));

  // matched is null unless both exist -- data.matched compares the two scene graphs'
  // objects (ignoring det_score), so it answers "have you actually changed anything
  // since this was seeded from perception", not just "do both files exist".
  const flag = el("match-flag");
  if (data.matched === null || data.matched === undefined) {
    flag.classList.add("hidden");
  } else {
    flag.classList.remove("hidden");
    flag.classList.toggle("matched", data.matched);
    flag.classList.toggle("different", !data.matched);
    flag.textContent = data.matched ? "✓ matched" : "✎ different";
  }
}

for (const btn of [el("view-human"), el("view-automated")]) {
  btn.addEventListener("click", () => {
    state.viewPreference = btn.dataset.pref;
    if (state.currentKey) loadImage(state.currentKey.id, state.currentKey.field);
  });
}

function setPathField(id, path) {
  const node = el(id);
  node.textContent = path || "";
  node.classList.toggle("path-empty", !path);
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
// function of the box's center, never hand-entered. Recomputed live from the current
// bbox on every render so it's always accurate even before the box has been saved.
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

  if (state.readOnly) {
    // Selecting (to inspect properties) is still allowed -- only drawing/moving/
    // resizing boxes is blocked. No `drag` is ever set here, so mousemove/mouseup
    // are no-ops for the rest of this interaction.
    for (let i = state.objects.length - 1; i >= 0; i--) {
      const rect = toDisplay(state.objects[i].bbox);
      if (mx >= rect[0] && mx <= rect[2] && my >= rect[1] && my <= rect[3]) {
        selectObject(i);
        return;
      }
    }
    state.selected = -1;
    renderObjectList();
    renderEditor();
    return;
  }

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
  if (state.readOnly) return;
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
  el("mark-empty").disabled = state.readOnly;
}

el("mark-empty").addEventListener("click", () => {
  if (state.readOnly) return;
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
    select.disabled = state.readOnly;
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

  // Region isn't a dropdown -- it's derived from the box position (regionOf()), the
  // same way the server derives it on save. Shown read-only so it's still visible
  // "as part of the properties" without inviting hand-entry that could drift from
  // the geometry the box actually implies.
  const regionField = document.createElement("div");
  regionField.className = "property-field";
  regionField.innerHTML = `<label>region (auto)</label><span class="region-readout">${regionOf(obj.bbox)}</span>`;
  container.appendChild(regionField);

  el("delete-object").classList.toggle("hidden", state.readOnly);
}

el("delete-object").addEventListener("click", deleteSelected);

function deleteSelected() {
  if (state.readOnly || state.selected < 0) return;
  state.objects.splice(state.selected, 1);
  state.selected = -1;
  render();
  renderObjectList();
  renderEditor();
  scheduleSave();
}

// ---------------------------------------------------------------------
// Autosave
// ---------------------------------------------------------------------
function scheduleSave() {
  if (state.readOnly) return; // belt-and-suspenders -- every caller above already checks this
  setSaveIndicator("saving");
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveNow, 400);
}

async function saveNow() {
  if (!state.currentKey || state.readOnly) return;
  const { id, field } = state.currentKey;
  const body = {
    objects: state.objects.map((o) => ({ bbox: o.bbox, properties: o.properties })),
    reasonable_scene: state.reasonableScene,
  };
  const res = await fetch(`/api/image/${id}/${field}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (res.ok) {
    setSaveIndicator("saved");
    // Re-fetch this image's metadata (not the canvas/objects) so the matched/different
    // flag, counts, and toggle state reflect what was actually just written.
    const meta = await fetchImageMeta(id, field);
    const im = state.images.find((x) => x.id === id && x.field === field);
    if (im) {
      im.has_human = meta.has_human;
      im.human_number_of_objects = meta.human_number_of_objects;
      im.matched = meta.matched;
    }
    populateImageSelect();
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
refreshState();

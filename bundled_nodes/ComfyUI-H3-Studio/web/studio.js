import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { installCharacterPresets } from "./character_presets.js";
import { installStudioLoras } from "./studio_loras.js";

const MODES = ["Reference to video", "First + last frame", "Video inpainting"];
const MEDIA_DEFAULTS = { ref_image: "", last_image: "", source_video: "", start_seconds: 0, end_seconds: 0 };
const states = new Set();
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const rounded = value => Math.round(value * 1000) / 1000;
const nodes = () => (app.rootGraph || app.graph)?._nodes || (app.rootGraph || app.graph)?.nodes || [];
const isClass = (node, name) => node?.comfyClass === name || node?.type === name;

function element(tag, className = "", text) {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text !== undefined) item.textContent = text;
    return item;
}

function widget(node, name) {
    const slot = node.inputs?.find(input => input.name === name);
    return (slot && node.getWidgetFromSlot?.(slot)) || node.widgets?.find(item => item.name === name);
}

function setValue(node, name, value) {
    const item = widget(node, name);
    if (!item) return;
    if (isClass(node, "H3StudioMedia") && ["ref_image", "last_image", "source_video"].includes(name)) {
        if (name === "source_video") node._h3StudioTrim?.close?.();
        const revisions = node._h3StudioMediaRevisions ??= {};
        revisions[name] = (revisions[name] || 0) + 1;
    }
    item.value = value;
    item.callback?.(value);
    node.graph?.change?.();
    node.setDirtyCanvas?.(true, true);
}

function hideWidget(item) {
    if (!item || item._h3StudioHidden) return;
    item._h3StudioHidden = true;
    item.hidden = true;
    item.computeSize = () => [0, -4];
    if (item.element) item.element.style.display = "none";
}

function hideNativeControls(node, names) {
    if (node._h3StudioNativeControls) return;
    const managed = new Set(names);
    const replaced = item => {
        if (!item || item.type === "h3_studio_panel") return false;
        if (managed.has(item.name)) return true;
        return node.inputs?.some(input => managed.has(input.name)
            && (input._widget === item || (input.widgetId && input.widgetId === item.widgetId)));
    };
    // Promoted subgraph widgets can be rebuilt after configuration. Filter at the
    // host's visibility/layout boundaries, keeping every widget and binding intact.
    const isWidgetVisible = node.isWidgetVisible;
    node.isWidgetVisible = function (item, ...args) {
        return !replaced(item) && (isWidgetVisible?.call(this, item, ...args) ?? !item?.hidden);
    };
    const getLayoutWidgets = node.getLayoutWidgets;
    node.getLayoutWidgets = function (...args) {
        const items = getLayoutWidgets?.apply(this, args) || this.widgets || [];
        return items.filter(item => !replaced(item));
    };
    node._h3StudioNativeControls = true;
}

function modeFor(node) {
    const engine = node.properties?.h3StudioControls ? node : nodes().find(item => item.properties?.h3StudioControls);
    return widget(engine || {}, "mode")?.value || MODES[0];
}

function viewURL(value, defaults = {}) {
    if (typeof value !== "string" || !value.trim()) return null;
    const type = value.match(/ \[(input|output|temp)\]$/)?.[1] || defaults.type || "input";
    const path = value.replace(/\\/g, "/").replace(/ \[(input|output|temp)\]$/, "");
    const parts = path.split("/");
    if (path.startsWith("/") || /^[a-z]:/i.test(path) || parts.includes("..")) return null;
    const filename = parts.pop();
    if (!filename) return null;
    const subfolder = defaults.subfolder ?? parts.join("/");
    return api.apiURL(`/view?${new URLSearchParams({ filename, subfolder, type })}`);
}

function installStyle() {
    if (document.getElementById("h3-studio-style")) return;
    const style = element("style");
    style.id = "h3-studio-style";
    style.textContent = `
.h3-studio {box-sizing:border-box;width:100%;height:auto!important;max-height:none;overflow:visible;padding:14px;background:transparent;color:#edf6ef;border:0;border-radius:0;font:13px/1.45 system-ui,sans-serif;color-scheme:dark;}
.h3-studio.h3-controls {background:#14271f;border:1px solid #3b5b47;border-radius:12px;}
.h3-studio * {box-sizing:border-box;}
.h3-studio [hidden] {display:none!important;}
.h3-studio h3 {margin:0 0 4px;font-size:19px;letter-spacing:-.35px;}
.h3-studio p {margin:0;}
.h3-studio .h3-kicker {font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#9dcaad;font-weight:700;margin-bottom:5px;}
.h3-studio .h3-muted {color:#a9c1b1;font-size:12px;}
.h3-studio .h3-section {border-top:1px solid #34513e;margin-top:16px;padding-top:14px;}
.h3-studio .h3-section-title {font-size:12px;font-weight:700;letter-spacing:.7px;color:#b8d5c2;margin:0 0 10px;}
.h3-studio .h3-field {display:flex;flex-direction:column;gap:6px;min-width:0;margin-bottom:12px;}
.h3-studio .h3-field>span {color:#d7e8dc;font-weight:600;}
.h3-studio input,.h3-studio select,.h3-studio textarea {width:100%;min-width:0;border:1px solid #496653;border-radius:7px;background:#0e1c15;color:#f1f8f3;padding:9px 10px;font:13px/1.4 system-ui,sans-serif;outline:none;}
.h3-studio textarea {resize:none;min-height:112px;overflow:hidden;}
.h3-studio input:focus,.h3-studio select:focus,.h3-studio textarea:focus {border-color:#a8dfb9;box-shadow:0 0 0 2px #a8dfb919;}
.h3-studio .h3-mode select {font-size:16px;font-weight:650;padding:11px;background:#294d37;border-color:#709b7d;}
.h3-studio .h3-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 12px;}
.h3-studio .h3-check {flex-direction:row;align-items:center;font-size:12px;}
.h3-studio .h3-check input {width:16px;height:16px;accent-color:#88c99a;}
.h3-studio .h3-slider {display:grid;grid-template-columns:minmax(70px,1fr) 72px;align-items:center;gap:12px;}
.h3-studio .h3-slider input[type="number"] {text-align:right;font-variant-numeric:tabular-nums;}
.h3-studio .h3-field:has(input:disabled) {opacity:.55;}
.h3-studio button {border:1px solid #52765e;background:#284c36;color:#edf9f0;border-radius:7px;padding:9px 12px;font:600 12px/1.3 system-ui,sans-serif;cursor:pointer;}
.h3-studio button:hover:not(:disabled) {background:#356447;border-color:#91bd9c;}
.h3-studio button:focus-visible {outline:2px solid #bfebcb;outline-offset:2px;}
.h3-studio button:disabled {opacity:.5;cursor:default;}
.h3-studio .h3-primary {width:100%;padding:13px;font-size:16px;background:#418454;border-color:#9acaab;}
.h3-studio .h3-primary:hover:not(:disabled) {background:#519c65;}
.h3-studio .h3-secondary {width:100%;margin-bottom:8px;background:#203e2c;}
.h3-studio .h3-status {min-height:32px;font-size:12px;color:#c0d7c8;margin-top:9px;}
.h3-studio .h3-status[data-error="true"] {color:#ffbcb0;}
.h3-studio .h3-character-toolbar {display:flex;align-items:center;gap:10px;margin-bottom:10px;}
.h3-studio .h3-character-toolbar .h3-section-title {flex:1;margin:0;}
.h3-studio .h3-character-toolbar select {flex:1;min-width:0;}
.h3-studio .h3-character-toolbar button {flex:none;}
.h3-studio .h3-character-status {min-height:18px;color:#b9cfbf;font-size:12px;overflow-wrap:anywhere;margin-bottom:12px;}
.h3-studio .h3-character-status[data-error="true"] {color:#ffbcb0;}
.h3-studio.h3-character-dialog {width:min(540px,calc(100vw - 48px));}
.h3-studio.h3-character-dialog>p.h3-muted {margin:8px 0 18px;}
.h3-studio .h3-card {border:1px solid #45674f;border-radius:10px;overflow:hidden;background:#102018;margin-top:12px;}
.h3-studio .h3-card.h3-drag {border:2px solid #a3dfb4;background:#23462f;}
.h3-studio .h3-card-header {display:flex;align-items:center;justify-content:space-between;gap:8px;padding:10px 11px;font-weight:650;}
.h3-studio .h3-badge {font-size:9px;letter-spacing:.9px;padding:3px 6px;border-radius:4px;background:#31563d;color:#cae5d2;text-transform:uppercase;}
.h3-studio .h3-card-media {position:relative;background:#09130e;min-height:105px;display:flex;align-items:center;justify-content:center;overflow:hidden;}
.h3-studio .h3-card-media img,.h3-studio .h3-card-media video {display:block;width:100%;max-height:245px;min-height:100px;object-fit:contain;background:#09130e;}
.h3-studio .h3-empty {padding:27px 14px;text-align:center;color:#8eaf9a;font-size:12px;}
.h3-studio .h3-card-footer {padding:10px;display:flex;flex-wrap:wrap;align-items:center;gap:7px;}
.h3-studio .h3-filename {width:100%;font-size:10px;color:#9db9a7;overflow-wrap:anywhere;}
.h3-studio .h3-clear {margin-left:auto;padding:7px 9px;background:transparent;color:#acc7b5;}
.h3-studio .h3-trim {padding:11px;border-top:1px solid #36523f;}
.h3-studio input[type="range"] {padding:0;border:0;accent-color:#8ecda0;background:transparent;}
.h3-studio .h3-trim {display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.h3-studio .h3-trim>button {flex:none;}
.h3-studio .h3-trim>p {flex:1;min-width:140px;}
.h3-studio details>summary {color:#b9d2c1;cursor:pointer;font-size:12px;margin:7px 0 12px;}
.h3-studio .h3-preview-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;}
.h3-studio .h3-preview-grid .h3-card {margin-top:0;min-width:0;}
.h3-studio .h3-preserved {border-color:#4e7794;}
.h3-studio .h3-preserved .h3-card-header {background:#203b4b;color:#c3e7fc;}
.h3-studio .h3-target .h3-card-header {background:#244831;color:#ccecd5;}
.h3-studio .h3-guide {border-color:#6b725b;}
.h3-studio .h3-guide .h3-card-header {background:#373e2d;color:#e0e6ca;}
.h3-studio .h3-preview-grid img,.h3-studio .h3-preview-grid video {width:100%;height:205px;object-fit:contain;background:#09130e;display:block;}
.h3-studio-dialog {border:1px solid #6d997a;border-radius:14px;padding:18px;background:#14271f;color:#eef8f0;width:min(840px,92vw);max-height:94vh;overflow:auto;}
.h3-studio-dialog::backdrop {background:#050b08cf;}
.h3-studio-dialog .h3-rectangle-stage {position:relative;height:min(58vh,490px);background:#08110b;margin:15px 0;overflow:hidden;}
.h3-studio-dialog .h3-rectangle-stage img,.h3-studio-dialog .h3-rectangle-stage video {position:absolute;inset:0;width:100%;height:100%;object-fit:contain;}
.h3-studio-dialog .h3-draw-surface {position:absolute;touch-action:none;cursor:crosshair;}
.h3-studio-dialog .h3-rectangle-box {position:absolute;border:2px solid #a4ebba;background:#79dc9b30;pointer-events:none;}
.h3-studio-dialog .h3-dialog-actions {display:flex;justify-content:flex-end;gap:8px;}

.h3-clip-dialog {box-sizing:border-box;width:min(1160px,96vw);max-height:96vh;overflow:auto;border:1px solid #6d997a;border-radius:16px;padding:24px;background:linear-gradient(145deg,#213b2c,#14271f);color:#eef5ff;font:14px/1.45 system-ui,sans-serif;box-shadow:0 24px 100px #0009;}
.h3-clip-dialog::backdrop {background:rgba(3,7,13,.82);}
.h3-clip-dialog * {box-sizing:border-box;}
.h3-clip-dialog h2 {margin:0;font-size:24px;letter-spacing:-.5px;}
.h3-clip-dialog p {margin:8px 0;color:#bac9d9;}
.h3-clip-header {display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:16px;}
.h3-clip-header span {max-width:48%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#95a8bd;font-size:12px;}
.h3-clip-stage {position:relative;border:1px solid #344251;border-radius:10px;overflow:hidden;background:#060a10;}
.h3-clip-video {display:block;width:100%;height:min(48vh,520px);min-height:160px;object-fit:contain;background:#060a10;}
.h3-clip-stage-label {position:absolute;left:12px;top:12px;padding:4px 9px;background:#101b27c9;border:1px solid #354e67;border-radius:4px;color:#c1d8ee;font-size:10px;font-weight:700;letter-spacing:1px;pointer-events:none;}
.h3-clip-controls,.h3-clip-fields,.h3-clip-actions {display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:12px;}
.h3-clip-controls input {flex:1;min-width:80px;accent-color:#7dd3fc;}
.h3-clip-controls span {font-variant-numeric:tabular-nums;color:#bad0e5;font-size:12px;min-width:78px;text-align:right;}
.h3-clip-dialog button {padding:9px 14px;border:1px solid #4b6178;border-radius:6px;background:#283a4d;color:#e3f0ff;cursor:pointer;font:inherit;transition:background .12s;}
.h3-clip-dialog button:hover:not(:disabled) {background:#385169;}
.h3-clip-dialog button:disabled {opacity:.45;cursor:default;}
.h3-clip-fields label {display:flex;align-items:center;gap:8px;color:#c0d0e0;}
.h3-clip-fields input {width:106px;padding:9px;background:#0e1722;color:white;border:1px solid #435c75;border-radius:6px;font:14px/1.4 ui-monospace,monospace;}
.h3-clip-range {position:relative;height:70px;margin:24px 12px 4px;background:#0b1420;border-radius:5px;touch-action:none;}
.h3-clip-filmstrip {position:absolute;inset:0;display:flex;overflow:hidden;border-radius:5px;opacity:.8;pointer-events:none;}
.h3-clip-filmstrip canvas {display:block;width:10%;height:100%;object-fit:cover;background:linear-gradient(135deg,#192839,#0a1521);border-right:1px solid #44617a55;}
.h3-clip-shade {position:absolute;top:0;height:100%;background:#050a12bf;pointer-events:none;}
.h3-clip-selection {position:absolute;top:0;height:100%;border-top:3px solid #74d6fa;border-bottom:3px solid #74d6fa;background:#45bdec11;pointer-events:none;}
.h3-clip-range .h3-clip-handle {position:absolute;top:-5px;width:24px;height:80px;padding:0;transform:translateX(-50%);border:2px solid #ade9ff;border-radius:5px;background:#256483;cursor:ew-resize;touch-action:none;box-shadow:0 2px 8px #0009;}
.h3-clip-handle::after {content:"";display:block;width:4px;height:22px;margin:auto;border-left:1px solid #dbf7ff;border-right:1px solid #dbf7ff;}
.h3-clip-handle span {position:absolute;top:-20px;left:50%;transform:translateX(-50%);font-size:10px;letter-spacing:.4px;font-weight:700;color:#aeeaff;}
.h3-clip-range .h3-clip-handle:focus {outline:2px solid white;outline-offset:2px;z-index:2;}
.h3-clip-playhead {position:absolute;top:-5px;bottom:-5px;width:2px;background:white;box-shadow:0 0 4px #000;pointer-events:none;}
.h3-clip-playhead::before {content:"";position:absolute;left:-4px;top:-3px;border-left:5px solid transparent;border-right:5px solid transparent;border-top:7px solid white;}
.h3-clip-ruler {display:flex;justify-content:space-between;margin:9px 12px 0;color:#7f97af;font-size:11px;font-variant-numeric:tabular-nums;}
.h3-clip-dialog .h3-clip-duration {color:#93dff9;font-size:14px;font-weight:600;margin:12px 0;}
.h3-clip-dialog .h3-clip-shortcuts {font-size:11px;color:#8198b0;margin:12px 0 0;}
.h3-clip-actions {padding-top:14px;border-top:1px solid #354454;}
.h3-clip-actions .h3-clip-cancel {margin-left:auto;}
.h3-clip-actions .h3-clip-apply {background:#23759a;border-color:#5db3d5;color:#fff;font-weight:600;}
.h3-clip-status {min-height:1.5em;color:#d0dce7;font-size:12px;}
@media(max-width:640px) {.h3-clip-dialog {padding:15px;}.h3-clip-header {display:block;}.h3-clip-header span {display:block;max-width:100%;margin-top:6px;}.h3-clip-video {height:35vh;}.h3-clip-fields {gap:8px;}.h3-clip-fields button {font-size:12px;padding:8px;}}

.h3-clip-dialog {color-scheme:dark;}
.h3-clip-dialog .h3-clip-apply {background:#418454;border-color:#9acaab;}
.h3-clip-dialog button:focus-visible {outline:2px solid #bfebcb;outline-offset:2px;}
`;
    document.head.append(style);
}

function attachPanel(node, name, panel, minWidth = 360) {
    installStyle();
    let height = 100;
    let frame = 0;
    let removed = false;
    let automaticHeight = null;
    const textSizes = new WeakMap();
    const schedule = () => {
        if (!removed && !frame) frame = requestAnimationFrame(fit);
    };
    const item = node.addDOMWidget(name, "h3_studio_panel", panel, {
        serialize: false, hideOnZoom: false, getMinHeight: () => height, getMaxHeight: () => height,
        afterResize: schedule,
    });
    item.serialize = false;
    const computeSize = node.computeSize;
    node.computeSize = function (...args) {
        const size = computeSize.apply(this, args);
        size[0] = Math.max(size[0], minWidth);
        // Include native sockets, widget margins and the node's bottom padding.
        size[1] = Math.max(size[1], (item.y || 0) + height + 8);
        return size;
    };
    function fit() {
        frame = 0;
        // An inactive subgraph or offscreen node may not have mounted its DOM yet.
        if (removed || !panel.isConnected || !panel.offsetWidth) return;
        for (const input of panel.querySelectorAll("textarea")) {
            if (!input.offsetWidth) continue;
            const previous = textSizes.get(input);
            if (previous?.value === input.value && previous.width === input.offsetWidth) continue;
            input.style.height = "0px";
            input.style.height = `${Math.ceil(input.scrollHeight + input.offsetHeight - input.clientHeight)}px`;
            textSizes.set(input, { value: input.value, width: input.offsetWidth });
        }
        // offsetHeight/scrollHeight are unscaled, unlike the zoomed canvas bounds.
        height = Math.ceil(Math.max(panel.offsetHeight, panel.scrollHeight)) + item.margin * 2;
        const minimum = node.computeSize();
        const followsContent = automaticHeight === null || Math.abs(node.size[1] - automaticHeight) < 2;
        const nextHeight = followsContent ? minimum[1] : Math.max(node.size[1], minimum[1]);
        const nextWidth = Math.max(node.size[0], minimum[0]);
        if (Math.abs(node.size[1] - nextHeight) >= 1 || Math.abs(node.size[0] - nextWidth) >= 1) {
            if (followsContent || nextHeight > node.size[1]) automaticHeight = nextHeight;
            node.setSize([nextWidth, nextHeight]);
        } else if (automaticHeight === null) automaticHeight = nextHeight;
        node.setDirtyCanvas?.(true, true);
    }
    const resize = new ResizeObserver(schedule);
    resize.observe(panel);
    const mutations = new MutationObserver(schedule);
    mutations.observe(panel, { subtree: true, childList: true, characterData: true,
        attributes: true, attributeFilter: ["hidden", "open", "class"] });
    for (const event of ["input", "load", "loadedmetadata", "error", "toggle"]) panel.addEventListener(event, schedule, true);
    node._h3StudioPanel = { schedule };
    const onRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
        removed = true;
        cancelAnimationFrame(frame);
        resize.disconnect(); mutations.disconnect();
        return onRemoved?.apply(this, args);
    };
    document.fonts?.ready.then(schedule);
    schedule();
    return item;
}

function track(node, update, cleanup = () => {}) {
    const state = { node, update: () => { update(); node._h3StudioPanel?.schedule(); }, cleanup };
    states.add(state);
    const removed = node.onRemoved;
    node.onRemoved = function (...args) {
        states.delete(state);
        cleanup();
        return removed?.apply(this, args);
    };
    return state;
}

function stopMedia(root) {
    for (const video of root.querySelectorAll("video")) {
        video.pause();
        video.removeAttribute("src");
        video.load();
    }
}

function studioTrimSource(node) {
    const value = String(widget(node, "source_video")?.value || "");
    const url = viewURL(value);
    if (!url) throw new Error("Choose a source video before opening Trim video.");
    const revision = node._h3StudioMediaRevisions?.source_video || 0;
    return { key: JSON.stringify([value, revision]), url, filename: value.replace(/\\/g, "/").split("/").at(-1) };
}

function openStudioTrimEditor(node) {
    const state = (node._h3StudioTrim ??= {});
    if (state.close) { state.dialog?.focus(); return; }
    let source;
    try { source = studioTrimSource(node); }
    catch (error) { window.alert(error.message); return; }
    installStyle();
    const previousFocus = document.activeElement;
    const dialog = element("dialog", "h3-clip-dialog");
    state.dialog = dialog;
    state.sourceKey = source.key;
    dialog.setAttribute("aria-label", "Trim source video");
    const header = element("div", "h3-clip-header");
    header.append(element("h2", "", "Choose your clip"), element("span", "", source.filename));
    const stage = element("div", "h3-clip-stage");
    const video = element("video", "h3-clip-video");
    video.preload = "metadata";
    video.playsInline = true;
    stage.append(video, element("span", "h3-clip-stage-label", "SOURCE PREVIEW"));
    const controls = element("div", "h3-clip-controls");
    const play = element("button", "", "Play selection");
    const scrub = element("input");
    Object.assign(scrub, { type: "range", min: "0", max: "0", step: "0.01", value: "0", disabled: true });
    scrub.setAttribute("aria-label", "Video position");
    const position = element("span", "", "0.00 s");
    controls.append(play, scrub, position);

    const range = element("div", "h3-clip-range");
    const filmstrip = element("div", "h3-clip-filmstrip");
    const leftShade = element("div", "h3-clip-shade");
    const rightShade = element("div", "h3-clip-shade");
    leftShade.style.left = "0";
    rightShade.style.right = "0";
    const selection = element("div", "h3-clip-selection");
    const startHandle = element("button", "h3-clip-handle");
    const endHandle = element("button", "h3-clip-handle");
    startHandle.appendChild(element("span", "", "IN"));
    endHandle.appendChild(element("span", "", "OUT"));
    const playhead = element("div", "h3-clip-playhead");
    playhead.style.left = "0%";
    const fields = element("div", "h3-clip-fields");
    const startInput = element("input");
    const endInput = element("input");
    for (const [input, labelText, handle] of [[startInput, "Start (s)", startHandle], [endInput, "End (s)", endHandle]]) {
        Object.assign(input, { type: "number", min: "0", step: "0.01", value: "0", disabled: true });
        const label = element("label", "", labelText);
        label.appendChild(input);
        fields.appendChild(label);
        handle.setAttribute("role", "slider");
        handle.setAttribute("aria-label", labelText);
        handle.disabled = true;
    }
    range.append(filmstrip, leftShade, rightShade, selection, playhead, startHandle, endHandle);
    const ruler = element("div", "h3-clip-ruler");
    const rulerLabels = Array.from({ length: 5 }, () => element("span", "", "0.0 s"));
    ruler.append(...rulerLabels);
    const setStart = element("button", "", "Set start here");
    const setEnd = element("button", "", "Set end here");
    fields.append(setStart, setEnd);
    const duration = element("p", "h3-clip-duration", "Selected: —");
    const status = element("p", "h3-clip-status", "Loading video preview…");
    status.setAttribute("role", "status");
    const actions = element("div", "h3-clip-actions");
    const full = element("button", "", "Full video");
    const cancel = element("button", "h3-clip-cancel", "Cancel");
    const apply = element("button", "h3-clip-apply", "Apply Trim");
    const readinessControls = [play, startHandle, endHandle, startInput, endInput, setStart, setEnd, full, apply, scrub];
    readinessControls.forEach((item) => { item.disabled = true; });
    actions.append(full, cancel, apply);
    dialog.append(header, stage, controls, range, ruler, duration, fields, element("p", "h3-clip-shortcuts", "I / O: set start / end · Space: play selection · ← / →: seek · Shift + arrow: seek 1 second"), status, actions);

    let closed = false;
    let ready = false;
    let start = 0;
    let end = 0;
    let drag = null;
    let thumbnails = null;
    const gap = () => Math.min(0.01, video.duration);
    const render = () => {
        startInput.value = String(rounded(start));
        endInput.value = String(rounded(end));
        startHandle.style.left = `${start / video.duration * 100}%`;
        endHandle.style.left = `${end / video.duration * 100}%`;
        selection.style.left = startHandle.style.left;
        selection.style.width = `${(end - start) / video.duration * 100}%`;
        leftShade.style.width = startHandle.style.left;
        rightShade.style.width = `${(video.duration - end) / video.duration * 100}%`;
        startHandle.setAttribute("aria-valuemin", "0");
        startHandle.setAttribute("aria-valuemax", String(end - gap()));
        startHandle.setAttribute("aria-valuenow", String(start));
        endHandle.setAttribute("aria-valuemin", String(start + gap()));
        endHandle.setAttribute("aria-valuemax", String(video.duration));
        endHandle.setAttribute("aria-valuenow", String(end));
        duration.textContent = `Selected: ${(end - start).toFixed(2)} s · ${start.toFixed(2)} to ${end.toFixed(2)} s`;
    };
    const choose = (which, value) => {
        if (!ready) return;
        if (!Number.isFinite(value)) { status.textContent = "Enter a valid time in seconds."; render(); return; }
        video.pause();
        if (which === "start") start = clamp(value, 0, end - gap());
        else end = clamp(value, start + gap(), video.duration);
        render();
        video.currentTime = which === "start" ? start : end;
        status.textContent = end - start < 0.25 ? "This selection is very short. Choose at least 0.25 seconds for MiniMax generation." : "Only Apply Trim saves your selection.";
    };
    const close = () => {
        if (closed) return;
        closed = true;
        ready = false;
        drag = null;
        thumbnails?.();
        video.pause();
        video.removeAttribute("src");
        video.load();
        dialog.close();
        dialog.remove();
        state.close = null;
        state.dialog = null;
        state.sourceKey = null;
        previousFocus?.focus?.();
    };
    state.close = close;
    cancel.addEventListener("click", close);
    dialog.addEventListener("cancel", (event) => { event.preventDefault(); close(); });
    video.addEventListener("loadedmetadata", () => {
        if (closed) return;
        ready = Number.isFinite(video.duration) && video.duration > 0;
        readinessControls.forEach((item) => { item.disabled = !ready; });
        if (!ready) { status.textContent = "The video duration could not be read. Try a browser-playable MP4."; return; }
        const savedStart = Number(widget(node, "start_seconds")?.value);
        const savedEnd = Number(widget(node, "end_seconds")?.value);
        end = savedEnd > 0 && Number.isFinite(savedEnd) ? clamp(savedEnd, gap(), video.duration) : video.duration;
        start = Number.isFinite(savedStart) ? clamp(savedStart, 0, end - gap()) : 0;
        scrub.max = String(video.duration);
        startInput.max = endInput.max = String(video.duration);
        video.currentTime = start;
        rulerLabels.forEach((label, index) => { label.textContent = `${(video.duration * index / 4).toFixed(1)} s`; });
        status.textContent = "Drag IN and OUT to choose your clip. Audio is kept in sync. Your original file stays unchanged.";
        render();
    });
    video.addEventListener("error", () => {
        if (closed) return;
        ready = false;
        readinessControls.forEach((item) => { item.disabled = true; });
        status.textContent = "This video cannot be previewed by your browser. Choose a browser-playable MP4 to trim it here. Your saved trim has not changed.";
    });
    video.addEventListener("timeupdate", () => {
        if (closed) return;
        if (ready && !video.paused && video.currentTime >= end) { video.pause(); video.currentTime = end; }
        scrub.value = String(video.currentTime);
        position.textContent = `${video.currentTime.toFixed(2)} s`;
        if (ready) playhead.style.left = `${clamp(video.currentTime / video.duration, 0, 1) * 100}%`;
    });
    video.addEventListener("play", () => { play.textContent = "Pause"; });
    video.addEventListener("pause", () => { play.textContent = "Play selection"; });
    play.addEventListener("click", () => {
        if (!ready) return;
        if (!video.paused) { video.pause(); return; }
        if (video.currentTime < start || video.currentTime >= end - 0.001) video.currentTime = start;
        video.play().catch(() => { status.textContent = "Playback failed. You can still scrub and choose times."; });
    });
    scrub.addEventListener("input", () => {
        if (!ready) return;
        video.pause();
        video.currentTime = clamp(Number(scrub.value), 0, video.duration);
    });
    for (const [which, input, handle, button] of [["start", startInput, startHandle, setStart], ["end", endInput, endHandle, setEnd]]) {
        input.addEventListener("change", () => choose(which, input.value.trim() ? Number(input.value) : NaN));
        button.addEventListener("click", () => choose(which, video.currentTime));
        handle.addEventListener("pointerdown", (event) => {
            if (!ready || event.button !== 0) return;
            drag = { which, pointerId: event.pointerId };
            handle.focus();
            handle.setPointerCapture(event.pointerId);
            event.preventDefault();
        });
        handle.addEventListener("pointermove", (event) => {
            if (!drag || drag.pointerId !== event.pointerId || drag.which !== which) return;
            const bounds = range.getBoundingClientRect();
            choose(which, (event.clientX - bounds.left) / bounds.width * video.duration);
        });
        const finish = (event) => {
            if (!drag || drag.pointerId !== event.pointerId) return;
            drag = null;
            if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
        };
        handle.addEventListener("pointerup", finish);
        handle.addEventListener("pointercancel", finish);
        handle.addEventListener("keydown", (event) => {
            if (!ready) return;
            const current = which === "start" ? start : end;
            const step = event.shiftKey ? 1 : 0.1;
            if (event.key === "ArrowLeft" || event.key === "ArrowDown") choose(which, current - step);
            else if (event.key === "ArrowRight" || event.key === "ArrowUp") choose(which, current + step);
            else if (event.key === "Home") choose(which, 0);
            else if (event.key === "End") choose(which, video.duration);
            else return;
            event.preventDefault();
            event.stopPropagation();
        });
        handle.addEventListener("lostpointercapture", () => { drag = null; });
    }
    range.addEventListener("pointerdown", (event) => {
        if (!ready || event.button !== 0 || event.target === startHandle || event.target === endHandle || event.target.parentElement === startHandle || event.target.parentElement === endHandle) return;
        const bounds = range.getBoundingClientRect();
        video.pause();
        video.currentTime = clamp((event.clientX - bounds.left) / bounds.width * video.duration, 0, video.duration);
    });
    full.addEventListener("click", () => {
        if (!ready) return;
        video.pause(); start = 0; end = video.duration; video.currentTime = 0;
        render();
        status.textContent = "Full video selected. Apply Trim saves this change.";
    });
    dialog.addEventListener("keydown", (event) => {
        event.stopPropagation();
        if (!ready || event.ctrlKey || event.metaKey || event.altKey) return;
        const tag = event.target.tagName?.toLowerCase();
        if (tag === "input" || tag === "textarea" || event.target.isContentEditable) return;
        if (event.key.toLowerCase() === "i") choose("start", video.currentTime);
        else if (event.key.toLowerCase() === "o") choose("end", video.currentTime);
        else if (event.key === " " && tag !== "button") play.click();
        else if ((event.key === "ArrowLeft" || event.key === "ArrowRight") && event.target !== startHandle && event.target !== endHandle) {
            video.pause();
            video.currentTime = clamp(video.currentTime + (event.key === "ArrowLeft" ? -1 : 1) * (event.shiftKey ? 1 : 1 / 30), 0, video.duration);
        } else return;
        event.preventDefault();
    });
    apply.addEventListener("click", () => {
        if (!ready || drag) return;
        try {
            if (studioTrimSource(node).key !== source.key) throw new Error("The source video changed. Cancel and reopen Trim video to use the new video.");
        } catch (error) {
            apply.disabled = true;
            status.textContent = error.message;
            return;
        }
        const savedStart = rounded(start);
        const savedEnd = end >= video.duration - 0.0005 ? 0 : rounded(end);
        if (savedEnd > 0 && savedEnd <= savedStart) {
            status.textContent = "Choose an end time after the start time.";
            return;
        }
        setValue(node, "start_seconds", savedStart);
        setValue(node, "end_seconds", savedEnd);
        state.sync?.();
        close();
    });
    for (const button of dialog.querySelectorAll("button")) button.type = "button";
    document.body.appendChild(dialog);
    dialog.showModal();
    video.src = source.url;
    thumbnails = sampleStudioTrimThumbnails(source.url, filmstrip);
}

function sampleStudioTrimThumbnails(url, strip) {
    const preview = element("video");
    preview.muted = true;
    preview.playsInline = true;
    preview.preload = "auto";
    const canvases = Array.from({ length: 10 }, () => {
        const canvas = element("canvas");
        canvas.width = 144;
        canvas.height = 80;
        strip.appendChild(canvas);
        return canvas;
    });
    let index = 0;
    let closed = false;
    const stop = () => {
        if (closed) return;
        closed = true;
        preview.pause();
        preview.removeAttribute("src");
        preview.load();
    };
    const seek = () => { preview.currentTime = preview.duration * (index + 0.5) / canvases.length; };
    preview.addEventListener("loadedmetadata", () => {
        if (closed) return;
        if (!Number.isFinite(preview.duration) || preview.duration <= 0) { stop(); return; }
        seek();
    });
    preview.addEventListener("seeked", () => {
        if (closed) return;
        if (preview.videoWidth && preview.videoHeight) {
            const canvas = canvases[index];
            const scale = Math.max(canvas.width / preview.videoWidth, canvas.height / preview.videoHeight);
            const width = preview.videoWidth * scale;
            const height = preview.videoHeight * scale;
            try {
                canvas.getContext("2d")?.drawImage(preview, (canvas.width - width) / 2, (canvas.height - height) / 2, width, height);
            } catch { stop(); return; }
        }
        index++;
        if (index === canvases.length) stop();
        else seek();
    });
    preview.addEventListener("error", stop);
    preview.src = url;
    return stop;
}

function addTrim(node, card, video) {
    const panel = element("div", "h3-trim");
    const button = element("button", "", "Trim video");
    button.type = "button";
    button.title = "Open the original video with a filmstrip and draggable IN / OUT handles. Apply Trim saves your selection.";
    const summary = element("p", "h3-muted");
    const state = node._h3StudioTrim ??= {};
    button.addEventListener("click", () => { video.pause(); openStudioTrimEditor(node); });
    panel.append(button, summary);
    card.append(panel);
    function sync() {
        let source;
        try { source = studioTrimSource(node); } catch { /* Empty source card. */ }
        button.disabled = !source;
        if (state.close && source?.key !== state.sourceKey) state.close();
        const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
        const rawStart = Number(widget(node, "start_seconds")?.value || 0);
        const rawEnd = Number(widget(node, "end_seconds")?.value || 0);
        const start = Number.isFinite(rawStart) ? Math.max(0, rawStart) : 0;
        const end = Number.isFinite(rawEnd) && rawEnd > 0 ? rawEnd : duration;
        summary.textContent = !source ? "Choose a video to trim."
            : end > start ? `${start.toFixed(2)} – ${end.toFixed(2)} s · ${(end - start).toFixed(2)} s selected`
                : start > 0 ? `From ${start.toFixed(2)} s to the end` : "Full video selected";
    }
    state.sync = sync;
    video.addEventListener("loadedmetadata", sync);
    video.addEventListener("timeupdate", () => {
        const end = Number(widget(node, "end_seconds")?.value || 0);
        if (end > 0 && !video.paused && video.currentTime >= end) video.pause();
    });
    video.addEventListener("play", () => {
        const start = Number(widget(node, "start_seconds")?.value || 0);
        const end = Number(widget(node, "end_seconds")?.value || 0) || video.duration;
        if (video.currentTime < start || video.currentTime >= end) video.currentTime = start;
    });
    return sync;
}

function mediaCard(node, name, videoCard = false) {
    const card = element("div", "h3-card");
    card.title = videoCard ? "Drop a source video onto this card, or choose a file. Then click Trim video to choose the section to use."
        : "Drop an image onto this card, or choose a file from your computer.";
    const header = element("div", "h3-card-header");
    const title = element("span");
    const badge = element("span", "h3-badge");
    header.append(title, badge);
    const stage = element("div", "h3-card-media");
    const empty = element("div", "h3-empty", videoCard ? "Drop a video here" : "Drop an image here");
    const media = element(videoCard ? "video" : "img");
    media.hidden = true;
    if (videoCard) Object.assign(media, { controls: true, preload: "metadata", playsInline: true });
    else media.alt = "Selected guide image";
    stage.append(empty, media);
    const footer = element("div", "h3-card-footer");
    const choose = element("button", "", videoCard ? "Choose video" : "Choose image");
    choose.type = "button";
    choose.title = videoCard ? "Upload a source video from your computer. Click Trim video to choose the section to use."
        : "Upload an image from your computer to use as a visual guide.";
    const clear = element("button", "h3-clear", "Clear");
    clear.type = "button";
    clear.title = "Remove this file from the card. This does not delete the uploaded file.";
    const filename = element("span", "h3-filename", "No file selected");
    const fileInput = element("input");
    Object.assign(fileInput, { type: "file", accept: videoCard ? "video/*,.mkv,.mov,.webm,.mp4,.avi" : "image/*" });
    fileInput.hidden = true;
    const status = element("p", "h3-status");
    status.style.minHeight = "0";
    status.style.width = "100%";
    status.setAttribute("role", "status");
    footer.append(choose, clear, filename, fileInput, status);
    card.append(header, stage, footer);
    let lastPath;
    let uploading = false;
    const syncTrim = videoCard ? addTrim(node, card, media) : null;
    const sync = () => {
        const value = String(widget(node, name)?.value || "");
        if (value !== lastPath) {
            lastPath = value;
            const url = viewURL(value);
            if (videoCard) media.pause();
            if (url) media.src = url;
            else {
                media.removeAttribute("src");
                if (videoCard) media.load();
            }
            media.hidden = !url;
            empty.hidden = !!url;
            filename.textContent = value || "No file selected";
            filename.title = value;
            clear.disabled = !value || uploading;
        }
        syncTrim?.();
    };
    async function upload(file) {
        if (!file || uploading) return;
        if (!videoCard && file.type && !file.type.startsWith("image/")) {
            status.textContent = "Choose an image for this card.";
            status.dataset.error = "true";
            return;
        }
        const previousValue = String(widget(node, name)?.value || "");
        const revision = node._h3StudioMediaRevisions?.[name] || 0;
        uploading = true;
        choose.disabled = clear.disabled = true;
        status.dataset.error = "false";
        status.textContent = "Uploading…";
        try {
            const body = new FormData();
            body.append("image", file);
            body.append("type", "input");
            body.append("subfolder", "h3-studio");
            body.append("overwrite", "false");
            const response = await api.fetchApi("/upload/image", { method: "POST", body });
            if (!response.ok) throw new Error(`Upload failed (${response.status})`);
            const result = await response.json();
            if (!result.name) throw new Error("The upload returned no filename.");
            if (!node.graph || revision !== (node._h3StudioMediaRevisions?.[name] || 0)
                || previousValue !== String(widget(node, name)?.value || "")) {
                status.textContent = "Upload finished. Your newer media selection was kept.";
                return;
            }
            setValue(node, name, [result.subfolder, result.name].filter(Boolean).join("/"));
            if (videoCard) { setValue(node, "start_seconds", 0); setValue(node, "end_seconds", 0); }
            status.textContent = "";
        } catch (error) {
            status.textContent = error.message || String(error);
            status.dataset.error = "true";
        } finally {
            uploading = false;
            choose.disabled = false;
            clear.disabled = !widget(node, name)?.value;
            fileInput.value = "";
            sync();
        }
    }
    choose.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => upload(fileInput.files?.[0]));
    clear.addEventListener("click", () => {
        setValue(node, name, "");
        if (videoCard) { setValue(node, "start_seconds", 0); setValue(node, "end_seconds", 0); }
        sync();
    });
    card.addEventListener("dragover", event => {
        if (!event.dataTransfer?.types.includes("Files")) return;
        event.preventDefault(); event.stopPropagation(); card.classList.add("h3-drag");
    });
    card.addEventListener("dragleave", event => { if (!card.contains(event.relatedTarget)) card.classList.remove("h3-drag"); });
    card.addEventListener("drop", event => {
        event.preventDefault(); event.stopPropagation(); card.classList.remove("h3-drag");
        upload(event.dataTransfer?.files?.[0]);
    });
    media.addEventListener("error", () => {
        status.textContent = videoCard ? "Browser preview unavailable. The saved file and trim times can still be used." : "This image could not be displayed. Choose another image.";
    });
    return { card, title, badge, sync, media };
}

function installMedia(node) {
    if (node._h3StudioMedia || !widget(node, "ref_image")) return;
    node._h3StudioMedia = true;
    ["ref_image", "last_image", "source_video", "start_seconds", "end_seconds"].forEach(name => hideWidget(widget(node, name)));
    const panel = element("div", "h3-studio");
    panel.append(element("div", "h3-kicker", "01 / Media"), element("h3", "", "Your starting point"));
    const hint = element("p", "h3-muted");
    panel.append(hint);
    const ref = mediaCard(node, "ref_image");
    const last = mediaCard(node, "last_image");
    const source = mediaCard(node, "source_video", true);
    panel.append(source.card, ref.card, last.card);
    let previousMode;
    const update = () => {
        const mode = modeFor(node);
        const inpaint = mode === MODES[2];
        source.card.hidden = !inpaint;
        last.card.hidden = mode !== MODES[1];
        ref.title.textContent = mode === MODES[1] ? "First frame" : "Reference image";
        ref.badge.textContent = inpaint ? "Optional" : "Required";
        last.title.textContent = "Last frame"; last.badge.textContent = "Required";
        source.title.textContent = "Source video"; source.badge.textContent = "Required";
        hint.textContent = inpaint ? "Choose a clip, then select the area to edit." : mode === MODES[1]
            ? "Guide the opening and ending of one continuous shot." : "Use an image for identity, style or scene guidance.";
        if (previousMode !== mode && !inpaint) { source.media.pause(); node._h3StudioTrim?.close?.(); }
        previousMode = mode;
        for (const item of [ref, last, source]) {
            // Hidden cards retain their paths without fetching inactive media.
            if (!item.card.hidden) item.sync();
        }
    };
    attachPanel(node, "h3_studio_media", panel);
    track(node, update, () => { node._h3StudioTrim?.close?.(); stopMedia(panel); });
    update();
}

const FIELDS = {
    mode: ["Generation mode", "Choose what you want to create.", "select"],
    reference_description: ["Character description", "Describe the character’s appearance and identity, or load a saved character. This description accompanies the reference image, which is the first frame in First + last frame mode. Use Describe your video for the scene, action, camera and sound.", "textarea"],
    prompt: ["Describe your video", "Describe the subject, action, camera and sound. Reference inputs can be named <Picture 1>.", "textarea"],
    target_megapixels: ["Resolution · MP", "0.8 MP is the default. Higher values need more memory and time.", "number"],
    duration_seconds: ["Duration · seconds", "Used for reference and first/last generation. H3 rounds to its frame grid at 24 fps.", "number"],
    aspect_ratio: ["Frame shape", "Choose the output shape or use the input image's shape.", "select"],
    fit: ["Image fit", "How guide images fit the output frame.", "select"],
    noise_seed: ["Seed", "Keep a seed to compare changes, or use the seed behavior below for variety.", "number"],
    steps: ["Steps", "8 steps is the default turbo setting.", "number"],
    mask_mode: ["Selection", "Use the whole image, find a subject with SAM 3, or draw a rectangle.", "select"],
    sam3_target: ["Target", "Describe the area to select, such as shirt, face, hair or person.", "text"],
    preserve_text: ["Preserve", "Optional. Items to subtract from the target selection. This does not guarantee unchanged generated pixels.", "text"],
    treatment: ["Guide treatment", "Original keeps the guide intact. Isolate selection hides the surroundings. Invert selected colors changes RGB inside the target, not which area is selected.", "select"],
    rectangle: ["Rectangle coordinates", "Normalized left, top, right and bottom coordinates, between 0 and 1.", "text"],
    strength: ["Mask strength", "Blend the guide treatment from 0 (unchanged) to 1 (full treatment). Original uses the selection for prompt guidance and previews only, so strength is inactive.", "number"],
    feather: ["Feather", "Softens the selection boundary.", "number"],
    preserve_refinement: ["Preserve refinement", "SAM 3 refinement passes for the preserve selection.", "number"],
    preserve_cleanup: ["Clean preserve mask", "Remove tiny specks and fill small enclosed holes.", "checkbox"],
    reuse_sam3_masks: ["Reuse saved SAM 3 masks", "Reuse matching masks when only generation settings change.", "checkbox"],
    include_original_video: ["Include original video reference", "Also use the original clip to guide motion and expressions.", "checkbox"],
};
const REMEMBER = [...Object.keys(FIELDS).filter(name => !["mode", "reference_description"].includes(name)), "loras_json"];

function field(node, name) {
    const item = widget(node, name);
    if (!item) return null;
    const [title, tip, kind] = FIELDS[name];
    const row = element("label", `h3-field${kind === "checkbox" ? " h3-check" : ""}`);
    row.title = tip;
    const label = element("span", "", title);
    const input = element(kind === "textarea" ? "textarea" : kind === "select" ? "select" : "input");
    const range = name === "strength" ? element("input") : null;
    input.setAttribute("aria-label", title);
    input.title = tip;
    if (kind === "select") {
        const values = name === "mode" ? MODES : typeof item.options?.values === "function" ? item.options.values() : item.options?.values;
        for (const value of values || [item.value]) {
            const option = element("option", "", String(value));
            option.value = value; input.append(option);
        }
    } else if (kind !== "textarea") {
        input.type = kind;
        if (kind === "number") {
            input.step = name === "strength" ? "0.01" : ["noise_seed", "steps", "preserve_refinement", "feather"].includes(name) ? "1" : "0.1";
            if (item.options?.min !== undefined) input.min = item.options.min;
            if (item.options?.max !== undefined) input.max = item.options.max;
        }
    }
    if (name === "prompt") {
        input.rows = 5;
        input.placeholder = "A single smooth shot… Describe movement, camera, lighting and sound.";
    }
    if (name === "reference_description") {
        input.rows = 5;
        input.placeholder = "Describe appearance, clothing and other identifying details…";
    }
    if (name === "preserve_text") input.placeholder = "Optional · glasses, necklace…";
    if (range) {
        Object.assign(range, { type: "range", min: "0", max: "1", step: "0.01", title: tip });
        Object.assign(input, { min: "0", max: "1", step: "0.01" });
        range.setAttribute("aria-label", `${title} slider`);
        range.addEventListener("input", () => {
            setValue(node, name, Number(range.value));
            input.value = Number(range.value).toFixed(2);
            node._h3StudioControls?.update();
        });
    }
    function sync() {
        const value = widget(node, name)?.value;
        if (range && document.activeElement !== range) range.value = value ?? 1;
        if (document.activeElement === input) return;
        if (kind === "checkbox") input.checked = !!value;
        else input.value = range && Number.isFinite(Number(value)) ? Number(value).toFixed(2) : value ?? "";
    }
    input.addEventListener(kind === "textarea" || kind === "text" ? "input" : "change", () => {
        let value = kind === "checkbox" ? input.checked : kind === "number" ? Number(input.value) : input.value;
        if (kind === "number" && (!input.value.trim() || !Number.isFinite(value))) return;
        if (name === "strength") { value = clamp(value, 0, 1); input.value = value.toFixed(2); }
        if (name === "mode") changeMode(node, value);
        else setValue(node, name, value);
        node._h3StudioControls?.update();
    });
    if (range) {
        const slider = element("div", "h3-slider");
        slider.append(range, input); row.append(label, slider);
    } else if (kind === "checkbox") row.append(input, label); else row.append(label, input);
    hideWidget(item);
    sync();
    return { row, input, range, sync };
}

function snapshotMode(node) {
    return Object.fromEntries(REMEMBER.flatMap(name => {
        const value = widget(node, name)?.value;
        return value === undefined ? [] : [[name, value]];
    }));
}

function modeMediaStore(node) {
    const store = node.properties?.h3StudioMediaByMode;
    return store && typeof store === "object" && !Array.isArray(store) ? store : null;
}

function connectedMedia(node) {
    const input = node.inputs?.find(item => item.name === "media");
    const link = input?.link != null && (node.graph?.getLink?.(input.link) || node.graph?.links?.[input.link]);
    const media = link && node.graph?.getNodeById?.(link.origin_id);
    return isClass(media, "H3StudioMedia") ? media : null;
}

function snapshotMediaMode(node, previous = {}) {
    const media = connectedMedia(node);
    const values = { ...MEDIA_DEFAULTS, ...previous };
    if (media) for (const name of Object.keys(MEDIA_DEFAULTS)) values[name] = widget(media, name)?.value ?? MEDIA_DEFAULTS[name];
    const preset = node.properties.h3CharacterPreset;
    const path = value => String(value || "").replaceAll("\\", "/").replace(/ \[input\]$/, "");
    return { ...values,
        reference_description: String(widget(node, "reference_description")?.value || ""),
        character_preset: preset?.id && path(preset.image) === path(values.ref_image) ? { ...preset } : null,
    };
}

function restoreMediaMode(node, saved = {}) {
    const media = connectedMedia(node);
    const values = { ...MEDIA_DEFAULTS, ...saved };
    // setValue advances media revisions even when paths match, so an upload
    // started in another mode cannot overwrite this mode's current selection.
    if (media) for (const name of Object.keys(MEDIA_DEFAULTS)) setValue(media, name, values[name]);
    setValue(node, "reference_description", String(values.reference_description || ""));
    if (values.character_preset?.id) node.properties.h3CharacterPreset = { ...values.character_preset };
    else delete node.properties.h3CharacterPreset;
    if (node._h3StudioCharacterPresets) ++node._h3StudioCharacterPresets.epoch;
}

function changeMode(node, next) {
    const state = node._h3StudioControls;
    if (!state || state.mode === next || !MODES.includes(next)) return;
    const saved = node.properties.h3StudioModeSettings ??= {};
    saved[state.mode] = snapshotMode(node);
    const mediaByMode = modeMediaStore(node);
    if (mediaByMode) mediaByMode[state.mode] = snapshotMediaMode(node, mediaByMode[state.mode]);
    const restore = saved[next] || {
        ...state.defaults,
        prompt: next === MODES[2]
            ? "Change the shirt to deep red. Preserve its fabric, folds, lighting, the person, motion and background."
            : next === MODES[1]
                ? "A single smooth, continuous shot transitioning naturally from the first frame <Picture 1> to the last frame <Picture 2>. Keep the subjects’ appearance and scene consistent, with coherent movement and gentle camera motion."
                : state.defaults.prompt,
        mask_mode: next === MODES[2] ? "SAM3" : "No mask",
        treatment: next === MODES[2] ? "Invert selected colors" : "Original",
    };
    setValue(node, "mode", next);
    for (const [name, value] of Object.entries(restore)) setValue(node, name, value);
    if (mediaByMode) restoreMediaMode(node, mediaByMode[next]);
    state.mode = next;
    saved[next] = snapshotMode(node);
    state.update();
    for (const entry of states) if (isClass(entry.node, "H3StudioMedia")) entry.update();
}

function seedBehavior(node, parent) {
    const inner = node.subgraph?._nodes || node.subgraph?.nodes || [];
    const noise = inner.find(child => child.widgets?.some(item => item.name === "noise_seed")
        && child.widgets?.some(item => item.name === "control_after_generate"));
    const control = noise?.widgets?.find(item => item.name === "control_after_generate");
    if (!control) return;
    const row = element("label", "h3-field");
    row.append(element("span", "", "Seed behavior"));
    const select = element("select");
    select.setAttribute("aria-label", "Seed behavior");
    select.title = row.title = "Fixed reuses the seed for comparisons. Randomize chooses a new seed for each run; increment and decrement change it by one.";
    for (const value of ["fixed", "randomize", "increment", "decrement"]) {
        const option = element("option", "", value.charAt(0).toUpperCase() + value.slice(1));
        option.value = value; select.append(option);
    }
    select.value = control.value;
    select.addEventListener("change", () => {
        control.value = select.value;
        control.callback?.(select.value);
        node.graph?.change?.();
    });
    row.append(select); parent.append(row);
}

function previewPrompt(output, previewId) {
    const root = String(previewId);
    if (output[root]?.class_type !== "H3StudioMaskPreview") {
        throw new Error("The mask preview output is missing. Reopen the Studio workflow.");
    }
    const keep = new Set();
    const pending = [root];
    while (pending.length) {
        const id = pending.pop();
        if (keep.has(id)) continue;
        const node = output[id];
        if (node.class_type === "H3StudioGenerate" || node.class_type === "SaveVideo") {
            throw new Error("The mask preview is connected to video generation. Restore its guide connection before previewing.");
        }
        keep.add(id);
        for (const value of Object.values(node.inputs || {})) {
            if (Array.isArray(value) && value.length === 2 && Number.isInteger(value[1])
                && Object.prototype.hasOwnProperty.call(output, String(value[0]))) {
                pending.push(String(value[0]));
            }
        }
    }
    return Object.fromEntries(Object.entries(output).filter(([id]) => keep.has(id)));
}

function installControls(node) {
    if (node._h3StudioControls || !widget(node, "mode") || !widget(node, "treatment")) return;
    hideNativeControls(node, [...Object.keys(FIELDS), "loras_json"]);
    const panel = element("div", "h3-studio h3-controls");
    panel.append(element("div", "h3-kicker", "02 / Create"), element("h3", "", "MiniMax H3 Studio"));
    const fields = {};
    for (const name of Object.keys(FIELDS)) {
        const value = field(node, name);
        if (value) fields[name] = value;
    }
    fields.mode.row.classList.add("h3-mode");
    const intro = element("p", "h3-muted");
    panel.append(fields.mode.row, intro);
    const characters = installCharacterPresets(node, fields.reference_description, {
        findWidget: widget, setValue,
        onApplied: () => {
            node._h3StudioControls?.update();
            for (const entry of states) if (isClass(entry.node, "H3StudioMedia")) entry.update();
        },
    });
    if (characters) panel.append(characters.panel);
    if (fields.prompt) panel.append(fields.prompt.row);
    const output = element("div", "h3-grid");
    for (const name of ["target_megapixels", "duration_seconds", "aspect_ratio", "fit", "steps", "noise_seed"]) if (fields[name]) output.append(fields[name].row);
    seedBehavior(node, output);
    panel.append(output);
    let loras = installStudioLoras(node, { findWidget: widget, setValue });
    if (loras) panel.append(loras.panel);
    const maskSection = element("div", "h3-section");
    maskSection.append(element("div", "h3-section-title", "SELECTION & GUIDANCE"));
    for (const name of ["mask_mode", "sam3_target", "preserve_text", "treatment", "strength"]) if (fields[name]) maskSection.append(fields[name].row);
    const draw = element("button", "h3-secondary", "Draw rectangle on source");
    draw.type = "button";
    draw.title = "Open the reference image, first frame or source video to draw an edit area. The rectangle applies at the same relative position in every guide frame.";
    const scope = element("p", "h3-muted");
    maskSection.append(draw, scope);
    panel.append(maskSection);
    const advanced = element("div", "h3-section");
    advanced.append(element("div", "h3-section-title", "FINE ADJUSTMENTS"));
    for (const name of ["rectangle", "feather", "preserve_refinement", "preserve_cleanup", "reuse_sam3_masks", "include_original_video"]) if (fields[name]) advanced.append(fields[name].row);
    panel.append(advanced);
    const actions = element("div", "h3-section");
    const preview = element("button", "h3-secondary", "Preview masks only");
    const run = element("button", "h3-primary", "Run selected mode");
    preview.type = run.type = "button";
    preview.title = "Prepare the source and masks, then show the target, preserved areas and treated guide. MiniMax video generation is skipped.";
    const status = element("p", "h3-status", "Ready when you are.");
    status.setAttribute("role", "status");
    actions.append(preview, run, status); panel.append(actions);
    const state = {
        mode: widget(node, "mode").value, defaults: snapshotMode(node), busy: false,
        update() {
            const mode = widget(node, "mode")?.value || MODES[0];
            if (mode !== state.mode) { changeMode(node, mode); return; }
            for (const value of Object.values(fields)) value.sync();
            characters?.update();
            // Promoted controls can appear after the main settings panel mounts.
            if (!loras && widget(node, "loras_json")) {
                loras = installStudioLoras(node, { findWidget: widget, setValue });
                if (loras) panel.insertBefore(loras.panel, maskSection);
            }
            loras?.update();
            if (modeMediaStore(node) && characters) {
                const select = characters.panel.querySelector('[name="character_preset"]');
                const preset = node.properties.h3CharacterPreset;
                const id = preset?.id || "";
                if (select && select.value !== id) {
                    if (id && !Array.from(select.options).some(option => option.value === id)) {
                        const option = element("option", "", preset.name || "Saved character");
                        option.value = id; select.append(option);
                    }
                    select.value = id;
                }
            }
            const inpaint = mode === MODES[2];
            const selected = widget(node, "mask_mode")?.value;
            const hasPreserve = !!String(widget(node, "preserve_text")?.value || "").trim();
            if (fields.sam3_target) fields.sam3_target.row.hidden = selected !== "SAM3";
            if (fields.preserve_text) fields.preserve_text.row.hidden = false;
            for (const name of ["preserve_refinement", "preserve_cleanup"]) if (fields[name]) fields[name].row.hidden = !hasPreserve;
            if (fields.reuse_sam3_masks) fields.reuse_sam3_masks.row.hidden = selected !== "SAM3" && !hasPreserve;
            if (fields.duration_seconds) fields.duration_seconds.row.hidden = inpaint;
            if (fields.include_original_video) fields.include_original_video.row.hidden = !inpaint;
            if (fields.rectangle) fields.rectangle.row.hidden = selected !== "Rectangle";
            if (fields.strength) {
                const original = widget(node, "treatment")?.value === "Original";
                fields.strength.input.disabled = original;
                fields.strength.range.disabled = original;
                fields.strength.row.title = original
                    ? "Original keeps the input image unchanged. The selection is used only for prompt guidance and previews, so mask strength has no effect."
                    : FIELDS.strength[1];
            }
            draw.hidden = selected !== "Rectangle";
            intro.textContent = inpaint ? "Rework a selected area while using the source clip for motion." : mode === MODES[1]
                ? "Create the motion between your first and last frames." : "Create a new shot from your reference image.";
            scope.textContent = inpaint ? "Green marks the target; blue marks items excluded by Preserve. Inversion changes the target’s colors in the guide."
                : "Masks prepare your image guides. Preserve excludes items from that treatment; it does not freeze them throughout the video.";
            run.title = `Generate using ${mode.toLowerCase()} and update the mask previews.`;
        },
    };
    node._h3StudioControls = state;
    draw.addEventListener("click", () => {
        try { openRectangle(node); }
        catch (error) { status.textContent = error.message; status.dataset.error = "true"; }
    });
    async function queue(previewOnly) {
        if (state.busy) return;
        const outputs = node.properties.h3StudioOutputs;
        const required = previewOnly ? outputs?.preview : outputs?.save;
        if (required === undefined || required === null) {
            status.textContent = "The output connections are missing. Reopen the Studio workflow.";
            status.dataset.error = "true"; return;
        }
        state.busy = true; run.disabled = preview.disabled = true;
        status.dataset.error = "false";
        status.textContent = previewOnly ? "Queueing the mask preview…" : "Queueing your video…";
        try {
            let accepted;
            if (previewOnly) {
                const { output, workflow } = await app.graphToPrompt();
                const result = await api.queuePrompt(0, { output: previewPrompt(output, outputs.preview), workflow });
                if (!result?.prompt_id) {
                    throw new Error(result?.error?.message || "ComfyUI did not accept the mask preview. Check the highlighted settings.");
                }
                accepted = true;
            } else {
                accepted = await app.queuePrompt(0, 1);
            }
            status.dataset.error = String(accepted === false);
            status.textContent = accepted === false ? "ComfyUI could not queue this run. Check the highlighted settings."
                : previewOnly ? "Mask preview queued. Video generation is skipped." : "Video queued. Follow progress in ComfyUI.";
        } catch (error) {
            status.dataset.error = "true";
            status.textContent = `Could not queue: ${error.message || error}`;
        } finally { state.busy = false; run.disabled = preview.disabled = false; }
    }
    preview.addEventListener("click", () => queue(true));
    run.addEventListener("click", () => queue(false));
    attachPanel(node, "h3_studio_controls", panel, 500);
    track(node, state.update, () => loras?.cleanup());
    state.update();
}

function openRectangle(engine) {
    const source = nodes().find(node => isClass(node, "H3StudioMedia"));
    if (!source) throw new Error("Connect the Studio Media node first.");
    const inpaint = modeFor(engine) === MODES[2];
    const path = widget(source, inpaint ? "source_video" : "ref_image")?.value;
    const url = viewURL(path);
    if (!url) throw new Error(inpaint ? "Choose a source video first." : "Choose the reference or first image first.");
    const dialog = element("dialog", "h3-studio h3-studio-dialog");
    dialog.setAttribute("aria-label", "Draw edit rectangle");
    dialog.append(element("h3", "", "Draw your edit area"), element("p", "h3-muted", "Drag over the image. The same normalized rectangle applies to both endpoint images and every source frame."));
    const stage = element("div", "h3-rectangle-stage");
    const media = element(inpaint ? "video" : "img");
    if (inpaint) Object.assign(media, { muted: true, preload: "metadata", playsInline: true });
    else media.alt = "Draw an edit rectangle on this source";
    const surface = element("div", "h3-draw-surface");
    const box = element("div", "h3-rectangle-box");
    surface.append(box); stage.append(media, surface);
    let selection = [0.25, 0.2, 0.75, 0.8];
    try {
        const saved = JSON.parse(widget(engine, "rectangle")?.value || "null");
        if (Array.isArray(saved) && saved.length === 4 && saved.every(Number.isFinite)) selection = saved;
    } catch { /* Keep the default when manually entered coordinates are incomplete. */ }
    let origin = null;
    const render = () => Object.assign(box.style, { left: `${selection[0] * 100}%`, top: `${selection[1] * 100}%`, width: `${(selection[2] - selection[0]) * 100}%`, height: `${(selection[3] - selection[1]) * 100}%` });
    const fit = () => {
        const width = media.videoWidth || media.naturalWidth;
        const height = media.videoHeight || media.naturalHeight;
        if (!width || !height) return;
        const scale = Math.min(stage.clientWidth / width, stage.clientHeight / height);
        Object.assign(surface.style, { width: `${width * scale}px`, height: `${height * scale}px`, left: `${(stage.clientWidth - width * scale) / 2}px`, top: `${(stage.clientHeight - height * scale) / 2}px` });
        render();
    };
    media.addEventListener(inpaint ? "loadedmetadata" : "load", () => {
        if (inpaint) media.currentTime = Number(widget(source, "start_seconds")?.value || 0);
        fit();
    });
    const coordinates = event => {
        const bounds = surface.getBoundingClientRect();
        return [clamp((event.clientX - bounds.left) / bounds.width, 0, 1), clamp((event.clientY - bounds.top) / bounds.height, 0, 1)];
    };
    surface.addEventListener("pointerdown", event => { origin = coordinates(event); surface.setPointerCapture(event.pointerId); event.preventDefault(); });
    surface.addEventListener("pointermove", event => {
        if (!origin) return;
        const point = coordinates(event);
        selection = [Math.min(origin[0], point[0]), Math.min(origin[1], point[1]), Math.max(origin[0], point[0]), Math.max(origin[1], point[1])];
        render();
    });
    surface.addEventListener("pointerup", () => { origin = null; });
    surface.addEventListener("pointercancel", () => { origin = null; });
    const actions = element("div", "h3-dialog-actions");
    const cancel = element("button", "", "Cancel");
    const apply = element("button", "", "Use rectangle");
    cancel.title = "Close the rectangle editor without changing the saved selection.";
    apply.title = "Save this edit rectangle for the selected mode.";
    const observer = new ResizeObserver(fit);
    const close = () => { observer.disconnect(); stopMedia(dialog); dialog.close(); dialog.remove(); };
    cancel.addEventListener("click", close);
    apply.addEventListener("click", () => {
        if (selection[2] - selection[0] < 0.002 || selection[3] - selection[1] < 0.002) return;
        setValue(engine, "rectangle", JSON.stringify(selection.map(rounded)));
        engine._h3StudioControls?.update(); close();
    });
    dialog.addEventListener("cancel", event => { event.preventDefault(); close(); });
    actions.append(cancel, apply); dialog.append(stage, actions);
    document.body.append(dialog); dialog.showModal(); observer.observe(stage); media.src = url;
}

function installPreview(node) {
    if (node._h3StudioPreview) return;
    const panel = element("div", "h3-studio");
    panel.append(element("div", "h3-kicker", "03 / Mask preview"), element("h3", "", "Check the selection"));
    const description = element("p", "h3-muted", "Run Preview masks only to inspect the target and preserve selections.");
    const grid = element("div", "h3-preview-grid h3-section");
    for (const [label, color] of [["Target", "h3-target"], ["Preserved", "h3-preserved"]]) {
        const card = element("div", `h3-card ${color}`);
        card.append(element("div", "h3-card-header", label), element("div", "h3-empty", "Your mask preview appears here"));
        grid.append(card);
    }
    panel.append(description, grid);
    let lastKey = "";
    const display = output => {
        const values = output?.h3_studio_previews;
        if (!Array.isArray(values) || !values.length) return;
        const key = JSON.stringify(values);
        if (key === lastKey) return;
        lastKey = key;
        stopMedia(grid); grid.replaceChildren();
        for (const item of values) {
            const label = String(item.label || "Target");
            const preserved = /preserv/i.test(label);
            const guide = /model input/i.test(label);
            const card = element("div", `h3-card ${guide ? "h3-guide" : preserved ? "h3-preserved" : "h3-target"}`);
            const url = viewURL(item.filename || item.file, { type: item.type || "temp", subfolder: item.subfolder || "" });
            if (!url) continue;
            const video = String(item.mime || "").startsWith("video/") || /\.(mp4|webm|mov)$/i.test(item.filename || item.file);
            const media = element(video ? "video" : "img");
            if (video) Object.assign(media, { controls: true, muted: true, playsInline: true, preload: "metadata" });
            else media.alt = label;
            media.src = url;
            card.append(element("div", "h3-card-header", label), media); grid.append(card);
        }
        description.textContent = "Green: target. Blue: preserved selection. Model input shows the guide after its treatment.";
    };
    node._h3StudioPreview = { display };
    attachPanel(node, "h3_studio_previews", panel, 480);
    const executed = node.onExecuted;
    node.onExecuted = function (output) { const result = executed?.apply(this, arguments); display(output); return result; };
    track(node, () => display(app.nodeOutputs?.[String(node.id)]), () => stopMedia(panel));
}

function install(node, attempt = 0) {
    setTimeout(() => {
        if (isClass(node, "H3StudioMedia")) installMedia(node);
        if (isClass(node, "H3StudioMaskPreview")) installPreview(node);
        if (node.properties?.h3StudioControls) {
            installControls(node);
            if (!node._h3StudioControls && attempt < 8) install(node, attempt + 1);
        }
    }, attempt ? 100 : 0);
}

app.registerExtension({
    name: "MiniMax.H3.Studio",
    setup() {
        installStyle();
        setInterval(() => { for (const state of states) if (state.node.graph) state.update(); }, 500);
        api.addEventListener("executed", event => {
            const detail = event.detail || {};
            const id = String(detail.display_node ?? detail.node);
            nodes().find(node => String(node.id) === id)?._h3StudioPreview?.display(detail.output);
        });
    },
    nodeCreated(node) { install(node); },
    afterConfigureGraph() { for (const node of nodes()) install(node); },
});

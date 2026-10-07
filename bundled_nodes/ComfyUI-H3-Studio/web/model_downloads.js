import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const ENDPOINT = "/h3-studio/models";
const ACTIVE = new Set(["starting", "downloading", "verifying", "cancelling"]);
const bytes = value => `${(Number(value || 0) / 1024 ** 3).toFixed(2)} GB`;

function element(tag, className, text) {
    const value = document.createElement(tag);
    if (className) value.className = className;
    if (text !== undefined) value.textContent = text;
    return value;
}

function button(label, title) {
    const value = element("button", "", label);
    value.type = "button";
    value.title = title;
    return value;
}

function installStyle() {
    if (document.getElementById("h3-download-style")) return;
    const style = element("style");
    style.id = "h3-download-style";
    style.textContent = `
.h3-downloads{height:auto!important;max-height:none!important;overflow:visible;box-sizing:border-box;width:100%;padding:14px;background:#202934;color:#eef4ff;font:13px/1.45 system-ui;cursor:default;border-radius:8px}
.h3-downloads *{box-sizing:border-box}.h3-downloads p{margin:6px 0 12px}.h3-downloads .h3-download-muted{color:#b5c4d9}.h3-downloads .h3-download-toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:10px 0}
.h3-downloads button,.h3-downloads select{color:#f4f8ff;background:#33465f;border:1px solid #577290;border-radius:5px;padding:7px 10px;font:inherit;cursor:pointer}.h3-downloads button:disabled{opacity:.55;cursor:default}.h3-downloads select{flex:1;min-width:150px}
.h3-download-card{padding:12px 0;border-top:1px solid #3a4b60}.h3-download-card[hidden]{display:none}.h3-download-title{font-weight:650}.h3-download-path{font-size:11px;color:#b5c4d9;overflow-wrap:anywhere;margin:4px 0}.h3-download-state{color:#bed5ed;font-size:12px}
.h3-download-card a{color:#9cc9ff;text-decoration:underline}.h3-download-note{font-size:12px;color:#b5c4d9}.h3-download-job{padding:10px;margin:10px 0;background:#15202d;border-radius:5px;overflow-wrap:anywhere}.h3-download-job[data-error=true]{color:#ffc6bd}.h3-download-job progress{display:block;width:100%;height:14px;margin:8px 0}.h3-download-job[hidden]{display:none}
`;
    document.head.append(style);
}

function attach(node) {
    if (node._h3Downloads) return;
    installStyle();
    const panel = element("div", "h3-downloads");
    panel.setAttribute("aria-label", "MiniMax model downloads");
    panel.append(element("div", "h3-download-title", "MODELS & DOWNLOADS"), element("p", "h3-download-muted",
        "Choose a mode to see its models. Downloads begin only when you click a Download button; Run never downloads anything."));
    const toolbar = element("div", "h3-download-toolbar");
    const filter = element("select");
    filter.setAttribute("aria-label", "Models for mode");
    filter.title = "Show the exact models used by one mode, or all three. SAM3 is optional when you do not use SAM3 masks.";
    const refresh = button("Check files", "Check your local model folders. This does not contact model hosts or start a download.");
    toolbar.append(filter, refresh);
    const jobPanel = element("div", "h3-download-job");
    jobPanel.setAttribute("role", "status");
    jobPanel.hidden = true;
    const jobText = element("div");
    const progress = element("progress");
    progress.max = 1;
    const progressText = element("div", "h3-download-path");
    const cancel = button("Cancel download", "Stop this download. A matching partial file can be resumed on your next attempt.");
    jobPanel.append(jobText, progress, progressText, cancel);
    const status = element("p", "h3-download-muted", "Checking local models…");
    status.setAttribute("role", "status");
    const list = element("div");
    panel.append(toolbar, jobPanel, status, list);

    let removed = false, timer, frame, height = 240, latest, pending = false, requesting = false;
    let requestController;
    const cards = new Map();
    const dom = node.addDOMWidget("h3_model_downloads", "h3_model_downloads", panel, {
        serialize: false, hideOnZoom: false, getMinHeight: () => height,
        getMaxHeight: () => height, afterResize: scheduleFit,
    });
    dom.serialize = false;
    const originalCompute = node.computeSize;
    node.computeSize = function (...args) {
        const size = originalCompute.apply(this, args);
        return [Math.max(500, size[0]), Math.max(size[1], (dom.y || 0) + height + 8)];
    };
    function scheduleFit() {
        if (removed || frame) return;
        frame = requestAnimationFrame(() => {
            frame = null;
            if (!panel.isConnected || !panel.offsetWidth) return;
            height = Math.ceil(Math.max(panel.offsetHeight, panel.scrollHeight)) + (dom.margin || 0) * 2;
            const minimum = node.computeSize();
            if (Math.abs(node.size[1] - minimum[1]) > 1 || node.size[0] < 500)
                node.setSize([Math.max(node.size[0], 500), minimum[1]]);
            node.setDirtyCanvas?.(true, true);
        });
    }
    const resize = new ResizeObserver(scheduleFit);
    resize.observe(panel);

    function applyFilter() {
        const mode = filter.value;
        node.properties ??= {};
        node.properties.h3ModelDownloadFilter = mode;
        for (const [id, card] of cards) {
            const model = latest.models.find(row => row.id === id);
            card.root.hidden = Boolean(mode && !model.modes.includes(mode));
        }
        scheduleFit();
    }
    function createCard(model) {
        const root = element("div", "h3-download-card");
        root.dataset.modelId = model.id;
        const title = element("div", "h3-download-title", model.label);
        const path = element("div", "h3-download-path", model.destination);
        const state = element("div", "h3-download-state");
        const actions = element("div", "h3-download-toolbar");
        const link = element("a", "", "Model page ↗");
        link.href = model.page_url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.title = "Open the publisher’s model page for details, license, or manual download.";
        const download = button("Download", `Download only ${model.filename} into its fixed ComfyUI model folder. SHA-256 is checked before installation.`);
        download.dataset.downloadId = model.id;
        download.addEventListener("click", () => action({ action: "download", id: model.id }));
        actions.append(download, link);
        root.append(title, path, state, actions);
        if (model.note) root.append(element("div", "h3-download-note", model.note));
        list.append(root);
        return { root, state, download };
    }
    function render(data) {
        latest = data;
        const modes = [...new Set(data.models.flatMap(model => model.modes))];
        if (!filter.options.length) {
            const all = element("option", "", "All modes"); all.value = ""; filter.append(all);
            for (const mode of modes) { const option = element("option", "", mode); option.value = mode; filter.append(option); }
            filter.value = node.properties?.h3ModelDownloadFilter || "";
        }
        const active = ACTIVE.has(data.job?.state);
        let missing = 0;
        for (const model of data.models) {
            const card = cards.get(model.id) || createCard(model);
            cards.set(model.id, card);
            const present = model.local_status === "present";
            if (!present) ++missing;
            const description = {
                present: "Present locally (file size checked)",
                missing: "Not installed",
                size_mismatch: "Existing file has an unexpected size — check manually",
                blocked: "Existing path needs a manual check",
            }[model.local_status] || model.local_status;
            card.state.textContent = `${description} · ${bytes(model.size_bytes)}`;
            card.download.disabled = pending || active || model.manual || model.local_status !== "missing";
            card.download.textContent = present ? "Installed" : model.manual ? "Manual download" : "Download";
        }
        status.textContent = missing ? `${missing} model${missing === 1 ? "" : "s"} not ready across all modes. Download only the ones you need.` : "All listed model files are present.";
        jobPanel.hidden = !data.job;
        if (data.job) {
            const model = data.models.find(row => row.id === data.job.id);
            jobPanel.dataset.error = String(data.job.state === "error");
            jobText.textContent = `${model?.label || data.job.id}: ${data.job.error || data.job.message}`;
            const total = Number(data.job.total_bytes || 0);
            const loaded = Number(data.job.downloaded_bytes || 0);
            progress.value = total ? Math.min(1, loaded / total) : 0;
            progressText.textContent = total ? `${bytes(loaded)} / ${bytes(total)} (${Math.floor(100 * loaded / total)}%)` : "";
            cancel.hidden = !active;
            cancel.disabled = pending || data.job.state === "cancelling";
        }
        applyFilter();
    }
    async function jsonResponse(response) {
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
        return data;
    }
    async function load() {
        if (removed || requesting) return;
        clearTimeout(timer);
        requesting = true;
        requestController = new AbortController();
        try {
            const response = await api.fetchApi(ENDPOINT, { signal: requestController.signal });
            const data = await jsonResponse(response);
            if (!removed) render(data);
        } catch (error) {
            if (!removed && error.name !== "AbortError") {
                status.textContent = `Could not check models: ${error.message}`;
                scheduleFit();
            }
        } finally {
            requesting = false;
            if (!removed) timer = setTimeout(load, ACTIVE.has(latest?.job?.state) ? 1200 : 15000);
        }
    }
    async function action(payload) {
        if (pending || removed) return;
        pending = true;
        if (latest) render(latest);
        try {
            const data = await jsonResponse(await api.fetchApi(ENDPOINT, {
                method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
            }));
            if (!removed && latest && data.job) render({ ...latest, job: data.job });
            await load();
        } catch (error) {
            if (!removed) { status.textContent = error.message; scheduleFit(); }
        } finally {
            pending = false;
            if (!removed && latest) {
                // Re-enable the actions without discarding an actionable server error.
                for (const model of latest.models) cards.get(model.id).download.disabled =
                    ACTIVE.has(latest.job?.state) || model.manual || model.local_status !== "missing";
                cancel.disabled = latest.job?.state === "cancelling";
            }
        }
    }
    filter.addEventListener("change", applyFilter);
    refresh.addEventListener("click", load);
    cancel.addEventListener("click", () => action({ action: "cancel" }));
    panel.addEventListener("pointerdown", event => event.stopPropagation());
    panel.addEventListener("keydown", event => event.stopPropagation());
    const previousRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
        removed = true;
        clearTimeout(timer); cancelAnimationFrame(frame);
        requestController?.abort(); resize.disconnect();
        return previousRemoved?.apply(this, args);
    };
    node._h3Downloads = { panel, load };
    load(); scheduleFit();
}

app.registerExtension({
    name: "MiniMax.H3.Studio.ModelDownloads",
    nodeCreated(node) {
        if (node.comfyClass === "H3StudioModelDownloads" || node.type === "H3StudioModelDownloads")
            setTimeout(() => attach(node), 0);
    },
    afterConfigureGraph() {
        for (const node of app.graph?._nodes || [])
            if (node.comfyClass === "H3StudioModelDownloads" || node.type === "H3StudioModelDownloads") attach(node);
    },
});

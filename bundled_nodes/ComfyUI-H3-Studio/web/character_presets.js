import { api } from "../../../scripts/api.js";

const ENDPOINT = "/minimax-safe/character-presets";
const IMAGE_EXTENSIONS = /\.(png|jpe?g|webp|gif|bmp|tiff?|ico|avif)$/i;
const IMAGE_TYPES = "PNG, JPEG, WebP, GIF, BMP, TIFF, ICO or AVIF";
const imagePath = value => String(value || "").replaceAll("\\", "/").replace(/ \[input\]$/, "");

function element(tag, className = "", text) {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text !== undefined) item.textContent = text;
    return item;
}

function button(text, action, title) {
    const item = element("button", "", text);
    item.type = "button";
    item.dataset.action = action;
    item.title = title;
    return item;
}

async function responseJson(response, fallback) {
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `${fallback} (${response.status}).`);
    return data;
}

function validPreset(preset) {
    return preset && typeof preset.id === "string" && preset.id
        && typeof preset.name === "string" && preset.name
        && typeof preset.prompt === "string" && typeof preset.image === "string" && preset.image;
}

function referenceNode(node) {
    const input = node.inputs?.find(slot => slot.name === "media");
    const link = input?.link != null && (node.graph?.getLink?.(input.link) || node.graph?.links?.[input.link]);
    const media = link && node.graph?.getNodeById?.(link.origin_id);
    return media?.type === "H3StudioMedia" || media?.comfyClass === "H3StudioMedia" ? media : null;
}

export function installCharacterPresets(node, description, { findWidget, setValue, onApplied }) {
    if (!description) return null;
    const panel = element("div", "h3-section h3-characters");
    panel.dataset.testid = "h3-studio-character-presets";
    const heading = element("div", "h3-character-toolbar");
    const save = button("Save character", "save-character", "Save the Character description and reference image together in your existing character library.");
    heading.append(element("div", "h3-section-title", "CHARACTER"), save);
    const field = element("div", "h3-field");
    const label = element("span", "", "Saved character");
    const row = element("div", "h3-character-toolbar");
    const select = element("select");
    select.name = "character_preset";
    select.setAttribute("aria-label", "Saved character");
    select.title = "Load a saved image and Character description while keeping your scene prompt. In First + last frame mode, this image becomes the first frame. None keeps your current image and description.";
    const refresh = button("Refresh", "refresh-characters", "Refresh the shared character library while keeping your current image and descriptions.");
    const status = element("p", "h3-character-status");
    status.setAttribute("role", "status");
    status.dataset.testid = "character-preset-status";
    row.append(select, refresh); field.append(label, row);
    panel.append(heading, field, description.row, status);

    const state = { presets: [], epoch: 0, request: 0, removed: false, applying: false, dialog: null };
    node._h3StudioCharacterPresets = state;
    function message(text, error = false) { status.textContent = text; status.dataset.error = String(error); }
    function changed() { node.graph?.change?.(); node.setDirtyCanvas?.(true, true); }
    function currentReference() {
        const media = referenceNode(node);
        return { media, image: imagePath(findWidget(media || {}, "ref_image")?.value) };
    }
    function currentDescription() { return String(findWidget(node, "reference_description")?.value || ""); }
    function fillOptions() {
        select.replaceChildren();
        const none = element("option", "", "None — keep current reference"); none.value = ""; select.append(none);
        for (const preset of state.presets) {
            const option = element("option", "", preset.name); option.value = preset.id; select.append(option);
        }
        const active = node.properties.h3CharacterPreset;
        if (active?.id && !state.presets.some(preset => preset.id === active.id)) {
            const option = element("option", "", `${active.name || "Saved character"} (saved in workflow)`);
            option.value = active.id; select.append(option);
        }
        select.value = active?.id || "";
    }
    function clearPreset(text) {
        ++state.epoch;
        delete node.properties.h3CharacterPreset;
        fillOptions(); changed();
        if (text) message(text);
    }
    function update() {
        if (state.removed || state.applying) return;
        const active = node.properties.h3CharacterPreset;
        if (active?.id && currentReference().image !== imagePath(active.image)) {
            clearPreset("Reference changed. Your Character description is still editable.");
        }
    }
    function applyPreset(preset) {
        const { media } = currentReference();
        if (!media || !findWidget(media, "ref_image")) throw new Error("Connect Studio Media to the Media input first.");
        if (!validPreset(preset)) throw new Error("This saved character is missing its image or description.");
        state.applying = true;
        try {
            setValue(media, "ref_image", preset.image);
            setValue(node, "reference_description", preset.prompt);
            node.properties.h3CharacterPreset = { id: preset.id, name: preset.name, image: preset.image };
            fillOptions(); description.sync(); onApplied?.(); changed();
        } finally { state.applying = false; }
        message(`Using ${preset.name}. You can edit its Character description below.`);
    }
    async function refreshList() {
        const request = ++state.request;
        state.refreshAbort?.abort();
        const controller = new AbortController(); state.refreshAbort = controller;
        refresh.disabled = true; message("Loading saved characters…");
        try {
            const response = await api.fetchApi(ENDPOINT, { signal: controller.signal });
            const data = await responseJson(response, "Could not load saved characters");
            if (state.removed || request !== state.request) return;
            state.presets = (Array.isArray(data.presets) ? data.presets : []).filter(validPreset);
            fillOptions();
            message(state.presets.length ? `${state.presets.length} saved character${state.presets.length === 1 ? "" : "s"} available.`
                : "Save a character to reuse its image and description.");
        } catch (error) {
            if (!state.removed && request === state.request && !controller.signal.aborted) message(error.message || String(error), true);
        } finally {
            if (!state.removed && request === state.request) refresh.disabled = false;
        }
    }
    select.addEventListener("change", () => {
        ++state.epoch;
        if (!select.value) { clearPreset("No saved character selected. Your current image and description are unchanged."); return; }
        const preset = state.presets.find(item => item.id === select.value);
        if (!preset) { fillOptions(); message("Refresh the list to load this saved character.", true); return; }
        try { applyPreset(preset); }
        catch (error) { fillOptions(); message(error.message || String(error), true); }
    });
    refresh.addEventListener("click", refreshList);

    function openSaveDialog() {
        if (state.removed) return;
        if (state.dialog) { state.dialog.focus(); return; }
        const dialog = element("dialog", "h3-studio h3-studio-dialog h3-character-dialog");
        dialog.setAttribute("aria-label", "Save character");
        state.dialog = dialog;
        const active = node.properties.h3CharacterPreset?.id ? { ...node.properties.h3CharacterPreset } : null;
        const nameField = element("label", "h3-field");
        const name = element("input");
        Object.assign(name, { type: "text", name: "character_name", maxLength: 100, autocomplete: "off", placeholder: "Give this character a name", value: active?.name || "" });
        name.setAttribute("aria-label", "Character name");
        name.title = "Choose a name for a new character, or explicitly select Update to replace the selected character.";
        nameField.append(element("span", "", "Character name"), name);
        const updateLabel = element("label", "h3-field h3-check");
        const updateExisting = element("input"); updateExisting.type = "checkbox"; updateExisting.name = "update_character";
        updateExisting.title = "Replace only this selected character. This is off by default, so saving normally creates a new character.";
        updateLabel.append(updateExisting, element("span", "", active ? `Update “${active.name}”` : "Update selected character"));
        updateLabel.hidden = !active;
        const choices = element("div");
        const radios = {};
        for (const [value, title] of [["current", "Use current reference"], ["upload", "Upload another image"]]) {
            const choice = element("label", "h3-field h3-check");
            const radio = element("input");
            Object.assign(radio, { type: "radio", name: "character_image_source", value, checked: value === "current" });
            radio.title = value === "current" ? "Use Studio Media’s reference image, which is the first frame in First + last frame mode."
                : "Save a different image with the current Character description.";
            choice.append(radio, element("span", "", title)); choices.append(choice); radios[value] = radio;
        }
        const file = element("input");
        Object.assign(file, { type: "file", name: "character_image_upload", accept: ".png,.jpg,.jpeg,.webp,.gif,.bmp,.tif,.tiff,.ico,.avif", hidden: true });
        file.setAttribute("aria-label", "Character image upload"); file.title = `Choose a ${IMAGE_TYPES} image, up to 100 MiB.`;
        const dialogStatus = element("p", "h3-character-status");
        dialogStatus.dataset.testid = "character-save-status"; dialogStatus.setAttribute("role", "status");
        const actions = element("div", "h3-dialog-actions");
        const cancel = button("Cancel", "cancel-character", "Close this dialog.");
        const submit = button("Save new character", "submit-character", "Save the current Character description and chosen image. The scene prompt is kept separately.");
        actions.append(cancel, submit);
        dialog.append(element("h3", "", "Save character"), element("p", "h3-muted", "Save your Character description and reference image together in the shared character library."), nameField, updateLabel, choices, file, dialogStatus, actions);
        document.body.append(dialog);
        let closed = false;
        let saving = false;
        let saveController;
        const close = () => {
            if (closed) return;
            closed = true; ++state.epoch; saveController?.abort();
            dialog.close(); dialog.remove();
            if (state.dialog === dialog) { state.dialog = null; state.closeDialog = null; }
        };
        state.closeDialog = close;
        cancel.addEventListener("click", close);
        dialog.addEventListener("cancel", event => { event.preventDefault(); close(); });
        for (const radio of Object.values(radios)) radio.addEventListener("change", () => { file.hidden = !radios.upload.checked; });
        updateExisting.addEventListener("change", () => { submit.textContent = updateExisting.checked ? "Update saved character" : "Save new character"; });
        submit.addEventListener("click", async () => {
            if (saving || closed || state.removed) return;
            const prompt = currentDescription();
            const selectedName = name.value.trim();
            const reference = currentReference();
            const uploadFile = file.files?.[0];
            const useUpload = radios.upload.checked;
            const updateId = active && updateExisting.checked ? active.id : null;
            const fail = text => { dialogStatus.textContent = text; dialogStatus.dataset.error = "true"; };
            if (!selectedName) { fail("Enter a character name."); name.focus(); return; }
            if ([...selectedName].length > 100 || /[\x00-\x1f\x7f]/.test(selectedName)) { fail("Use a character name of up to 100 characters on one line."); name.focus(); return; }
            if (!prompt.trim()) { fail("Write a Character description before saving."); return; }
            if ([...prompt].length > 20000 || prompt.includes("\0")) { fail("Keep the Character description to 20,000 characters without null characters before saving."); return; }
            if (!useUpload && !reference.image) { fail("Choose a reference image first, or select Upload another image."); return; }
            if (useUpload && (!uploadFile || !IMAGE_EXTENSIONS.test(uploadFile.name) || uploadFile.size > 100 * 1024 ** 2)) {
                fail(`Choose a ${IMAGE_TYPES} image up to 100 MiB.`); return;
            }
            const epoch = state.epoch;
            const controller = new AbortController(); saveController = controller;
            saving = true;
            for (const control of [submit, name, updateExisting, file, ...Object.values(radios)]) control.disabled = true;
            cancel.textContent = "Close";
            dialogStatus.dataset.error = "false"; dialogStatus.textContent = "Saving character…";
            try {
                let image = reference.image;
                if (useUpload) {
                    dialogStatus.textContent = "Uploading reference image…";
                    const body = new FormData();
                    body.append("image", uploadFile); body.append("type", "input");
                    body.append("subfolder", "Character_Preset_Uploads"); body.append("overwrite", "false");
                    const response = await api.fetchApi("/upload/image", { method: "POST", body, signal: controller.signal });
                    const result = await responseJson(response, "Image upload failed");
                    if (!result.name || (result.type && result.type !== "input")) throw new Error("The upload did not return a usable reference image.");
                    image = [result.subfolder, result.name].filter(Boolean).join("/");
                    if (closed || state.removed || controller.signal.aborted) return;
                }
                const payload = { name: selectedName, prompt, image };
                if (updateId) payload.id = updateId;
                const response = await api.fetchApi(ENDPOINT, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal: controller.signal });
                if (response.status === 409 && !payload.id) throw new Error("That name is already saved. Choose a new name, or explicitly check Update for the selected character.");
                const result = await responseJson(response, "Could not save character");
                if (!validPreset(result.preset)) throw new Error("The saved character response is incomplete.");
                if (state.removed || closed || controller.signal.aborted) return;
                ++state.request; state.refreshAbort?.abort(); refresh.disabled = false;
                state.presets = [...state.presets.filter(item => item.id !== result.preset.id), result.preset].sort((a, b) => a.name.localeCompare(b.name));
                const current = currentReference();
                if (epoch === state.epoch && reference.media === current.media && reference.image === current.image && prompt === currentDescription() && current.media) {
                    applyPreset(result.preset);
                    message(`Saved ${result.preset.name}.`);
                } else {
                    fillOptions(); message(`Saved ${result.preset.name}. Your current workflow values were kept.`);
                }
                close();
            } catch (error) {
                if (!closed && !state.removed && !controller.signal.aborted) fail(error.message || String(error));
            } finally {
                saving = false;
                if (!closed && !state.removed) {
                    for (const control of [submit, name, updateExisting, file, ...Object.values(radios)]) control.disabled = false;
                    cancel.textContent = "Cancel";
                }
            }
        });
        dialog.showModal(); name.focus();
    }
    save.addEventListener("click", openSaveDialog);
    const onRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
        state.removed = true; ++state.request; ++state.epoch;
        state.refreshAbort?.abort(); state.closeDialog?.();
        return onRemoved?.apply(this, args);
    };
    fillOptions(); refreshList();
    return { panel, update };
}

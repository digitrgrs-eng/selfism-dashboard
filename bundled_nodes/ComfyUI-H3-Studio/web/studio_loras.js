import { api } from '../../../scripts/api.js';

const MAX_ROWS = 16;
const MAX_UPLOAD_BYTES = 4 * 1024 ** 3;
let installedNames = [];
let installedRequest = null;
let panelNumber = 0;

function findWidget(node, name) {
    const slot = node.inputs?.find(input => input.name === name);
    return (slot && node.getWidgetFromSlot?.(slot)) || node.widgets?.find(widget => widget.name === name);
}

function cleanName(value) {
    if (typeof value !== 'string' || value.length > 1024) throw new Error('Choose a valid LoRA filename.');
    const name = value.trim().replaceAll('\\', '/');
    if (name && (name.startsWith('/') || name.includes(':') || name.split('/').some(part => !part || part === '.' || part === '..') || /[\x00-\x1f\x7f]/.test(name))) {
        throw new Error('Choose an installed LoRA or upload a .safetensors file.');
    }
    return name;
}

export function parseStudioLoraRows(value) {
    if (typeof value !== 'string' || new TextEncoder().encode(value).length > 128 * 1024) throw new Error('LoRA settings are too large or are not text.');
    let rows;
    try { rows = JSON.parse(value); } catch { throw new Error('The saved LoRA list is not valid JSON.'); }
    if (!Array.isArray(rows) || rows.length > MAX_ROWS) throw new Error(`Use up to ${MAX_ROWS} LoRA rows.`);
    return rows.map((row, index) => {
        if (!row || typeof row !== 'object' || Array.isArray(row)) throw new Error(`LoRA ${index + 1} is not a valid row.`);
        const enabled = row.enabled ?? true;
        const strength = row.strength ?? 1;
        const triggers = row.triggers ?? '';
        if (typeof enabled !== 'boolean' || typeof strength !== 'number' || !Number.isFinite(strength) || strength < -10 || strength > 10 || typeof triggers !== 'string' || triggers.length > 4096) {
            throw new Error(`LoRA ${index + 1} has invalid settings.`);
        }
        return { enabled, name: cleanName(row.name ?? ''), strength, triggers };
    });
}

async function getInstalledNames(refresh = false) {
    if (installedRequest && !refresh) return installedRequest;
    installedRequest = (async () => {
        const response = await api.fetchApi('/object_info/LoraLoaderModelOnly');
        if (!response.ok) throw new Error(`Could not load installed LoRAs (${response.status}).`);
        const info = await response.json();
        const entry = info.LoraLoaderModelOnly?.input?.required?.lora_name;
        const names = Array.isArray(entry?.[0]) ? entry[0] : entry?.[1]?.options;
        if (!Array.isArray(names)) throw new Error('ComfyUI did not return its installed LoRA list.');
        installedNames = names.filter(name => typeof name === 'string').map(name => name.replaceAll('\\', '/'));
        return installedNames;
    })();
    try { return await installedRequest; } catch (error) { installedRequest = null; throw error; }
}

function element(tag, className, text) {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text !== undefined) item.textContent = text;
    return item;
}

function button(text, action, title) {
    const item = element('button', 'h3-lora-button', text);
    item.type = 'button';
    item.dataset.action = action;
    item.title = title;
    item.setAttribute('aria-label', title);
    Object.assign(item.style, { minHeight: '32px', padding: '5px 10px', color: '#e9efeb', background: '#294838', border: '1px solid #50755d', borderRadius: '6px', cursor: 'pointer', font: '600 13px system-ui' });
    return item;
}

function inputStyle(input) {
    Object.assign(input.style, { boxSizing: 'border-box', minWidth: '0', borderRadius: '5px', border: '1px solid #4d6056', color: '#f0f3f1', background: '#17231c', padding: '7px', font: '13px system-ui' });
}

export function installStudioLoras(node, options = {}) {
    const lookup = options.findWidget || findWidget;
    if (!lookup(node, 'loras_json')) return null;
    if (node._h3StudioLoras) return node._h3StudioLoras;
    const panel = element('div', 'h3-studio-loras h3-section');
    panel.dataset.testid = 'h3-lora-panel';
    panel.title = 'Choose MiniMax H3-compatible LoRAs. Enabled rows apply from top to bottom.';
    Object.assign(panel.style, { display: 'flex', flexDirection: 'column', gap: '8px', width: '100%', height: 'auto', overflow: 'visible' });
    const heading = element('div', 'h3-section-title', 'LORAS');
    const note = element('p', 'h3-muted', 'Choose LoRAs made for MiniMax H3. Enabled rows apply in order, with their trigger words added to your prompt.');
    const toolbar = element('div');
    Object.assign(toolbar.style, { display: 'flex', gap: '8px', flexWrap: 'wrap' });
    const add = button('+ Add LoRA', 'add-lora', 'Add another LoRA row. Choose its file to enable it.');
    const upload = button('Upload LoRA', 'upload-lora', 'Upload a MiniMax H3-compatible .safetensors LoRA and add it to this mode.');
    const refresh = button('Refresh list', 'refresh-loras', 'Reload installed LoRA filenames without changing your rows.');
    toolbar.append(add, upload, refresh);
    const cards = element('div', 'h3-lora-rows');
    Object.assign(cards.style, { display: 'flex', flexDirection: 'column', gap: '8px', overflow: 'visible', height: 'auto' });
    const status = element('div', 'h3-lora-status');
    status.setAttribute('role', 'status');
    Object.assign(status.style, { minHeight: '18px', color: '#b9c8be', overflowWrap: 'anywhere' });
    const fileInput = element('input');
    Object.assign(fileInput, { type: 'file', name: 'lora_upload', accept: '.safetensors', hidden: true });
    fileInput.title = 'Choose one MiniMax H3 LoRA in .safetensors format, up to 4 GiB.';
    panel.append(heading, note, toolbar, cards, status, fileInput);
    const state = { rows: [], invalid: null, writing: false, uploading: false, removed: false,
        names: installedNames, revision: 0, request: 0, lastRaw: undefined, panel };
    const fieldId = `h3-studio-loras-${node.id}-${++panelNumber}`;
    const message = (text, error = false) => {
        if (state.removed) return;
        status.textContent = text;
        status.style.color = error ? '#f3b5a9' : '#b9c8be';
    };
    const resize = () => { options.onResize?.(); node._h3StudioPanel?.schedule(); };
    function hideRawWidget(item) {
        if (!item) return;
        item.hidden = true; item.computeSize = () => [0, -4];
        if (item.element?.style) item.element.style.setProperty('display', 'none', 'important');
        if (item.inputEl?.style) item.inputEl.style.setProperty('display', 'none', 'important');
        if (item.options) { item.options.getMinHeight = () => 0; item.options.getMaxHeight = () => 0; }
    }
    function availability() {
        add.disabled = state.uploading || !!state.invalid || state.rows.length >= MAX_ROWS;
        upload.disabled = state.uploading || !!state.invalid || (state.rows.length >= MAX_ROWS && state.rows.every(row => row.name));
        refresh.disabled = state.uploading;
    }
    function persist() {
        const item = lookup(node, 'loras_json');
        if (!item || state.removed) return;
        const raw = JSON.stringify(state.rows);
        state.writing = true;
        state.lastRaw = raw;
        ++state.revision;
        try {
            if (options.setValue) options.setValue(node, 'loras_json', raw);
            else { item.value = raw; item.callback?.(raw); node.graph?.change?.(); node.setDirtyCanvas?.(true, true); }
        } finally { state.writing = false; }
        availability(); resize();
    }
    function populateSelect(select, name) {
        const choices = [element('option', '', 'Choose an installed LoRA…')];
        choices[0].value = '';
        if (name && !state.names.includes(name)) {
            const saved = element('option', '', `${name} (not in installed list)`);
            saved.value = name; choices.push(saved);
        }
        for (const value of state.names) {
            const option = element('option', '', value); option.value = value; choices.push(option);
        }
        select.replaceChildren(...choices); select.value = name;
    }
    function updateOptions() {
        for (const select of cards.querySelectorAll('select')) {
            const row = state.rows[Number(select.dataset.row)];
            if (row) populateSelect(select, row.name);
        }
    }
    async function refreshOptions(force = false) {
        const request = ++state.request;
        message('Loading installed LoRAs…');
        try {
            const names = await getInstalledNames(force);
            if (state.removed || request !== state.request) return;
            state.names = names; updateOptions();
            message(names.length ? `${names.length} installed LoRAs available.` : 'No installed LoRAs yet. Use Upload LoRA to add one.');
        } catch (error) { if (request === state.request) message(error.message || String(error), true); }
    }
    function renderRows() {
        cards.replaceChildren();
        if (state.invalid) {
            cards.append(element('p', 'h3-muted', `${state.invalid} Your saved text has been kept.`));
            const reset = button('Reset invalid list', 'reset-loras', 'Replace the invalid list with an empty one. Its original text is kept in node properties.');
            reset.addEventListener('click', () => {
                (node.properties ??= {}).h3LoraInvalidBackup = lookup(node, 'loras_json')?.value;
                state.rows = []; state.invalid = null; persist(); renderRows(); message('LoRA list reset.');
            });
            cards.append(reset); availability(); resize(); return;
        }
        if (!state.rows.length) cards.append(element('p', 'h3-muted', 'No LoRAs added. Use + Add LoRA or Upload LoRA.'));
        state.rows.forEach((row, index) => {
            const card = element('div', 'h3-lora-row'); card.dataset.row = String(index);
            Object.assign(card.style, { display: 'flex', flexDirection: 'column', gap: '7px', padding: '9px', border: '1px solid #405e4c', borderRadius: '8px', background: '#1d3025' });
            const top = element('div');
            Object.assign(top.style, { display: 'grid', gridTemplateColumns: 'auto minmax(0,1fr) 76px auto', alignItems: 'center', gap: '7px' });
            const enableLabel = element('label');
            Object.assign(enableLabel.style, { display: 'flex', gap: '4px', alignItems: 'center', whiteSpace: 'nowrap' });
            const enabled = element('input');
            Object.assign(enabled, { type: 'checkbox', name: 'enabled', checked: row.enabled });
            Object.assign(enabled.style, { width: '16px', height: '16px', flex: 'none', accentColor: '#88c99a' });
            enabled.title = 'Enable this LoRA and its trigger words. Off skips both.';
            enabled.setAttribute('aria-label', `Enable LoRA ${index + 1}`);
            enableLabel.append(enabled, document.createTextNode('Use'));
            enabled.addEventListener('change', () => { row.enabled = enabled.checked; persist(); });
            const name = element('select'); name.name = 'lora_name'; name.dataset.row = String(index);
            name.title = 'Choose an installed MiniMax H3-compatible LoRA.';
            name.setAttribute('aria-label', `LoRA ${index + 1} file`);
            inputStyle(name); name.style.width = '100%'; populateSelect(name, row.name);
            name.addEventListener('change', () => {
                let next;
                try { next = cleanName(name.value); } catch (error) { name.value = row.name; message(error.message, true); return; }
                if (next && !state.names.includes(next) && next !== row.name) { name.value = row.name; message('Choose an installed LoRA or upload it first.', true); return; }
                const wasEmpty = !row.name; row.name = next;
                if (!next) row.enabled = false; else if (wasEmpty) row.enabled = true;
                enabled.checked = row.enabled; persist(); message(next ? 'LoRA selection saved.' : 'Empty row is disabled.');
            });
            const strength = element('input');
            Object.assign(strength, { type: 'number', name: 'strength', min: '-10', max: '10', step: '0.05', value: String(row.strength) });
            strength.title = 'LoRA strength. 1.0 is normal weight; 0 leaves model weights unchanged. Enabled trigger words still apply.';
            strength.setAttribute('aria-label', `LoRA ${index + 1} strength`); inputStyle(strength); strength.style.width = '76px';
            strength.addEventListener('change', () => {
                const next = strength.valueAsNumber;
                if (!Number.isFinite(next) || next < -10 || next > 10) { strength.value = String(row.strength); message('Strength must be between -10 and 10.', true); return; }
                row.strength = next; persist();
            });
            const remove = button('Remove', 'remove-lora', 'Remove this row from the workflow. The LoRA file stays installed.');
            remove.addEventListener('click', () => { state.rows.splice(index, 1); persist(); renderRows(); });
            top.append(enableLabel, name, strength, remove);
            const label = element('label', '', 'Trigger words'); label.style.fontWeight = '600';
            const triggers = element('textarea');
            Object.assign(triggers, { name: 'triggers', value: row.triggers, maxLength: 4096, id: `${fieldId}-triggers-${index}` });
            label.htmlFor = triggers.id;
            triggers.placeholder = 'Optional trigger words for this LoRA';
            triggers.title = 'Enter the author’s trigger words, separated by commas or lines. Enabled rows add their trigger words to the video prompt.';
            triggers.setAttribute('aria-label', `LoRA ${index + 1} trigger words`); inputStyle(triggers);
            Object.assign(triggers.style, { width: '100%', height: '55px', minHeight: '55px', resize: 'none', overflow: 'hidden', lineHeight: '1.4' });
            triggers.addEventListener('input', () => { row.triggers = triggers.value; persist(); });
            card.append(top, label, triggers); cards.append(card);
        });
        availability(); resize();
    }
    function update() {
        if (state.writing || state.removed) return;
        const item = lookup(node, 'loras_json'); hideRawWidget(item);
        if (!item || item.value === state.lastRaw) return;
        state.lastRaw = item.value; ++state.revision;
        try { state.rows = parseStudioLoraRows(item.value); state.invalid = null; }
        catch (error) { state.invalid = error.message || String(error); }
        renderRows();
    }
    add.addEventListener('click', () => {
        if (state.invalid || state.uploading || state.rows.length >= MAX_ROWS) return;
        state.rows.push({ enabled: false, name: '', strength: 1, triggers: '' });
        persist(); renderRows(); message('Choose a LoRA file to enable the new row.');
    });
    refresh.addEventListener('click', () => refreshOptions(true));
    upload.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', async () => {
        const file = fileInput.files?.[0]; fileInput.value = '';
        if (!file || state.uploading || state.removed) return;
        if (!file.name.toLowerCase().endsWith('.safetensors') || !file.size || file.size > MAX_UPLOAD_BYTES) { message('Choose one .safetensors file up to 4 GiB.', true); return; }
        update();
        if (state.invalid) return;
        const revision = state.revision, mode = lookup(node, 'mode')?.value;
        const raw = lookup(node, 'loras_json')?.value;
        state.uploading = true; availability(); upload.textContent = 'Uploading…'; message(`Uploading ${file.name}…`);
        try {
            const data = new FormData(); data.append('file', file);
            const response = await api.fetchApi('/h3-studio/upload-lora', { method: 'POST', body: data });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || `Upload failed (${response.status}).`);
            const filename = cleanName(result.filename);
            if (!filename) throw new Error('The upload did not return a LoRA filename.');
            installedRequest = null;
            if (state.removed) return;
            if (!state.names.includes(filename)) state.names = [...state.names, filename];
            updateOptions();
            if (revision !== state.revision || mode !== lookup(node, 'mode')?.value || raw !== lookup(node, 'loras_json')?.value) {
                message(`Uploaded ${filename}. Your newer LoRA selection was kept; choose the file from the list when needed.`); return;
            }
            let row = state.rows.find(item => !item.name);
            if (!row) {
                if (state.rows.length >= MAX_ROWS) { message('File uploaded. Remove a row, then choose it from the list.'); return; }
                row = { enabled: true, name: '', strength: 1, triggers: '' }; state.rows.push(row);
            }
            row.name = filename; row.enabled = true; persist(); renderRows();
            message(`Added ${filename}. Enter its trigger words if needed.`);
        } catch (error) { message(error.message || String(error), true); }
        finally { state.uploading = false; upload.textContent = 'Upload LoRA'; availability(); }
    });
    state.update = update;
    state.cleanup = () => { state.removed = true; ++state.request; ++state.revision; };
    node._h3StudioLoras = state;
    update(); refreshOptions();
    return state;
}

(() => {
  const panel = document.querySelector('#view-selfism');
  const summary = panel.querySelector('#sf-status');
  const log = panel.querySelector('#sf-log');
  const progress = panel.querySelector('#sf-progress');
  const message = panel.querySelector('#sf-message');
  const models = panel.querySelector('#sf-models');
  const nodes = panel.querySelector('#sf-nodes');
  let loaded = false;
  let running = false;
  const error = panel.querySelector('#sf-error');
  function button(text, profile, item = '') {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'secondary-button'; b.textContent = text;
    b.dataset.sfAction = profile; b.dataset.sfItem = item;
    return b;
  }
  function link(text, url) {
    const a = document.createElement('a'); a.textContent = text;
    a.href = url; a.target = '_blank'; a.rel = 'noopener noreferrer'; return a;
  }
  function catalog(data) {
    for (const [id, f] of Object.entries(data.files)) {
      const row = document.createElement('div'); row.className = 'sf-row';
      const label = document.createElement('div');
      label.append(link(f.name, f.url));
      const small = document.createElement('small'); small.textContent = f.destination;
      label.append(small); row.append(label, button('Download', 'model', id)); models.append(row);
    }
    for (const n of data.nodes) {
      const row = document.createElement('div'); row.className = 'sf-row';
      row.append(link(n.name, n.repo), button('Install', 'node', n.name)); nodes.append(row);
    }
  }
  async function json(url, options = {}) {
    const r = await fetch(url, { ...options, headers: {'Content-Type':'application/json'} });
    const d = await r.json(); if (!r.ok) throw Error(d.detail || `HTTP ${r.status}`); return d;
  }
  async function refresh() {
    try {
      const d = await json('/api/selfism');
      if (!loaded) { catalog(d.catalog); loaded = true; }
      running = d.job.status === 'running';
      summary.textContent = d.job.status === 'idle' ? 'Spremno za instalaciju' : `${d.job.status} · ${d.job.stage}`;
      progress.value = d.job.percent || 0;
      message.textContent = [d.job.message, d.job.current_file,
        d.job.total_bytes ? `${(d.job.downloaded_bytes / 1e9).toFixed(2)} / ${(d.job.total_bytes / 1e9).toFixed(2)} GB` : '',
        d.job.error, ...(d.job.warnings || [])].filter(Boolean).join('\n');
      const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 32;
      const text = d.log.join('\n');
      if (log.textContent !== text) { log.textContent = text; if (atEnd) log.scrollTop = log.scrollHeight; }
      panel.querySelectorAll('[data-sf-action]').forEach(b => b.disabled = running);
      panel.querySelector('#sf-cancel').disabled = !running;
    } catch (e) { error.textContent = e.message; }
  }
  panel.addEventListener('click', async event => {
    const b = event.target.closest('[data-sf-action]'); if (!b || running) return;
    error.textContent = ''; b.disabled = true;
    try {
      await json('/api/selfism/install', {method:'POST', body:JSON.stringify({
        profile:b.dataset.sfAction, item:b.dataset.sfItem || '', precision:(b.dataset.sfAction === 'full' ? panel.querySelector('#sf-full-precision') : panel.querySelector('#sf-precision')).value
      })});
      await refresh();
    } catch (e) { error.textContent = e.message; b.disabled = false; }
  });
  panel.querySelector('#sf-cancel').addEventListener('click', async () => {
    try { await json('/api/selfism/cancel', {method:'POST'}); await refresh(); }
    catch (e) { error.textContent = e.message; }
  });
  refresh();
  // No overlapping polling requests, even when the server is slow.
  async function poll() { if (!panel.hidden || running) await refresh(); setTimeout(poll,2000); }
  setTimeout(poll,2000);
})();

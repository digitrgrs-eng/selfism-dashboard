"""minimax swap 2 card: LBH author settings (full Ref2VA INT8, FL2V 4-step v0.1, character swap, 3D upscaler, KJ preview)."""
from __future__ import annotations
import asyncio
import hashlib
import importlib
import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from tests.test_minimax import start_install, R2V_CNR_TO_PACK

m = importlib.import_module('launcher.app')
s = importlib.import_module('launcher.selfism')
ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / 'catalog/selfism.json').read_text(encoding='utf-8'))
WF = ROOT / 'selfism_workflows/minimax_swap_2.json'
# sha256 of the user-supplied workflow; the shipped copy differs only by prompt_provider -> none and tiny_vae -> none.
SUPPLIED_SHA256 = '597880441ac3363b56720d35f1bf24196c418c6f5c01d2789a443419b6e6071e'
IDS = ['r2v-unet', 'mm-lora-fl2v-4step-v01', 'mm-lora-charswap', 'mm-enc-nvfp4', 'mm-vae-fp16', 'mm-vae-audio', 'mm-latent-up-3d', 'r2v-taeh3']
NODES = ['ComfyUI-VideoHelperSuite', 'ComfyUI-KJNodes', 'Comfyui_Minimax_h3_latent_Upscaler']


def test_catalog_lists():
    assert CATALOG['minimax_swap_2_files'] == IDS and CATALOG['minimax_swap_2_nodes'] == NODES
    names = {n['name'] for n in CATALOG['nodes']}
    assert set(NODES) <= names and all(k in CATALOG['files'] for k in IDS)
    assert len({CATALOG['files'][k]['destination'] for k in IDS}) == len(IDS)


def test_install_selects_files_and_nodes_and_card_is_on_both_pages(monkeypatch):
    r, captured, html = start_install(monkeypatch, {'profile': 'minimax_swap_2'})
    assert r.status_code == 200
    assert [f['id'] for f in captured['files']] == IDS
    assert captured['selfism_profile'] == 'minimax_swap_2' and captured['selfism_repair'] is False
    assert captured['pip_packages'] == [] and captured['model_links'] == []
    assert {n['name'] for n in captured['custom_nodes']} == set(NODES)
    total = sum(CATALOG['files'][k]['size_bytes'] for k in IDS)
    scripts = (ROOT / 'scripts/selfism-section.html').read_text(encoding='utf-8')
    for page in (html, scripts):
        assert 'data-sf-action="minimax_swap_2"' in page and '<h2>minimax swap 2</h2>' in page
        assert f'{total/1e9:.1f}'.replace('.', ',') + ' GB' in page
        assert 'data-sf-action="minimax_r2v"' in page and 'data-sf-action="god_mode"' in page


def test_workflow_unchanged_and_models_installed():
    raw = WF.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SUPPLIED_SHA256
    w = json.loads(raw)
    installed = {Path(CATALOG['files'][k]['destination']).name for k in IDS}
    used = set()
    for n in w['nodes']:
        v = n.get('widgets_values')
        if n['type'] in ('UNETLoader', 'CLIPLoader', 'VAELoader', 'LoraLoaderModelOnly', 'MinimaxH3LatentUpscaler3D'): used.add(v[0])
        if n['type'] == 'ModelPreviewOverrideKJ': used.add(v[5])
    assert used == installed
    assert CATALOG['files']['mm-latent-up-3d']['destination'].startswith('models/latent_upscale_models/')


def test_download_and_install_are_identical_and_nondestructive(monkeypatch, tmp_path):
    with TestClient(m.app) as client:
        r = client.get('/api/selfism/workflow/minimax_swap_2')
    assert r.status_code == 200 and r.content == WF.read_bytes()
    assert s.MINIMAX_SWAP_2_WORKFLOW_NAME == 'MiniMax_H3_LBH_Millie_AuthorSettings_OriginalAudio.json'
    assert s.MINIMAX_SWAP_2_WORKFLOW_NAME in r.headers['content-disposition']
    monkeypatch.setattr(m, 'COMFYUI_DIR', tmp_path)
    monkeypatch.setattr(m, 'COMFYUI_VENV', tmp_path / '.venv')
    python = m.COMFYUI_VENV / 'bin/python'
    python.parent.mkdir(parents=True); python.touch()
    ctrl = type(m.selfism_controller)()
    async def ready(*args): pass
    async def run(*args, **kwargs): return 0, 'torch==2.8.0'
    async def sources(host, files): return files, []
    monkeypatch.setattr(ctrl, '_wait_for_comfyui', ready)
    monkeypatch.setattr(ctrl, '_run_process', run)
    monkeypatch.setattr(m.JobController, '_install_workflow', ready)
    monkeypatch.setattr(s, 'resolve_sources', sources)
    saved = tmp_path / 'user/default/workflows/Selfism' / s.MINIMAX_SWAP_2_WORKFLOW_NAME
    wf = {'selfism_profile': 'minimax_swap_2', 'precision': 'fp8', 'files': [{'id': 'x'}], 'pip_packages': []}
    asyncio.run(ctrl._install_workflow(wf))
    assert saved.read_bytes() == WF.read_bytes()
    saved.write_text('{"user_edited":true}')
    asyncio.run(ctrl._install_workflow(wf))
    assert json.loads(saved.read_text()) == {'user_edited': True}

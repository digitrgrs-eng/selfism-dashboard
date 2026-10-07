"""R2V Hearmeman Full card: full HearmemanAI R2V workflow (References Manager) + pruned Ref2VA INT8 + 8-step turbo LoRA."""
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
WF = ROOT / 'selfism_workflows/minimax_h3_r2v_hearmeman_full.json'
# sha256 of the user-supplied workflow; the shipped copy differs only by prompt_provider -> none and tiny_vae -> none.
SUPPLIED_SHA256 = '435e76a95ebae3241fc1f0789c1e5d7241205a4875a34a04a8d668f8c446e8fa'
IDS = ['mm-ref2va', 'r2v-enc', 'mm-vae-fp16', 'mm-vae-audio', 'r2v-lora-turbo-8step', 'r2v-taeh3']
NODES = ['rgthree-comfy', 'ComfyUI-KJNodes', 'ComfyUI-VideoHelperSuite', 'ComfyUI-MiniMaxRefPack']


def test_catalog_lists():
    assert CATALOG['r2v_hearmeman_full_files'] == IDS and CATALOG['r2v_hearmeman_full_nodes'] == NODES
    names = {n['name'] for n in CATALOG['nodes']}
    assert set(NODES) <= names and all(k in CATALOG['files'] for k in IDS)
    assert len({CATALOG['files'][k]['destination'] for k in IDS}) == len(IDS)


def test_install_selects_files_and_nodes_and_card_is_on_both_pages(monkeypatch):
    r, captured, html = start_install(monkeypatch, {'profile': 'minimax_r2v_hearmeman_full'})
    assert r.status_code == 200
    assert [f['id'] for f in captured['files']] == IDS
    assert captured['selfism_profile'] == 'minimax_r2v_hearmeman_full' and captured['selfism_repair'] is False
    assert captured['pip_packages'] == [] and captured['model_links'] == []
    assert {n['name'] for n in captured['custom_nodes']} == set(NODES)
    total = sum(CATALOG['files'][k]['size_bytes'] for k in IDS)
    scripts = (ROOT / 'scripts/selfism-section.html').read_text(encoding='utf-8')
    for page in (html, scripts):
        assert 'data-sf-action="minimax_r2v_hearmeman_full"' in page and '<h2>R2V Hearmeman Full</h2>' in page
        assert f'{total/1e9:.1f}'.replace('.', ',') + ' GB' in page
        assert 'data-sf-action="minimax_r2v"' in page and 'data-sf-action="god_mode"' in page


def test_workflow_models_nodes_and_safe_defaults():
    raw = WF.read_bytes()
    assert hashlib.sha256(raw).hexdigest() != SUPPLIED_SHA256
    w = json.loads(raw)
    installed = {Path(CATALOG['files'][k]['destination']).name for k in IDS}
    used = set()
    for n in w['nodes']:
        v = n.get('widgets_values')
        if n['type'] in ('UNETLoader', 'CLIPLoader', 'VAELoader', 'LoraLoaderModelOnly'): used.add(v[0])
        if n['type'] == 'ModelPreviewOverrideKJ': assert v[5] == 'none' and v[2] is True
        if n['type'] == 'MiniMaxH3ReferencePack': assert v[3] == 'none'
        for mdl in (n.get('properties') or {}).get('models', []): assert mdl['name'] in installed, mdl
    assert used == installed - {'taeh3.safetensors'}
    cnr = {(n.get('properties') or {}).get('cnr_id') for n in w['nodes']} - {None, 'comfy-core'}
    assert cnr <= set(R2V_CNR_TO_PACK) and {R2V_CNR_TO_PACK[c] for c in cnr} == set(NODES)


def test_download_and_install_are_identical_and_nondestructive(monkeypatch, tmp_path):
    with TestClient(m.app) as client:
        r = client.get('/api/selfism/workflow/minimax_r2v_hearmeman_full')
    assert r.status_code == 200 and r.content == WF.read_bytes()
    assert s.MINIMAX_R2V_HEARMEMAN_FULL_WORKFLOW_NAME == 'MiniMax_H3_R2V_Hearmeman_Full.json'
    assert s.MINIMAX_R2V_HEARMEMAN_FULL_WORKFLOW_NAME in r.headers['content-disposition']
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
    saved = tmp_path / 'user/default/workflows/Selfism' / s.MINIMAX_R2V_HEARMEMAN_FULL_WORKFLOW_NAME
    wf = {'selfism_profile': 'minimax_r2v_hearmeman_full', 'precision': 'fp8', 'files': [{'id': 'x'}], 'pip_packages': []}
    asyncio.run(ctrl._install_workflow(wf))
    assert saved.read_bytes() == WF.read_bytes()
    saved.write_text('{"user_edited":true}')
    asyncio.run(ctrl._install_workflow(wf))
    assert json.loads(saved.read_text()) == {'user_edited': True}

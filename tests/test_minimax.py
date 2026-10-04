import asyncio
import hashlib
import importlib
import json
import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

m = importlib.import_module('launcher.app')
s = importlib.import_module('launcher.selfism')
ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT/'catalog/selfism.json').read_text(encoding='utf-8'))
WF_PATH = ROOT/'selfism_workflows/minimax_h3_simply_advanced.json'
# sha256 of the third-party "Simply Advanced" v1.4 workflow exactly as supplied; it must ship unchanged.
WF_SHA256 = '31979ea700c3e1297194ecc263e3d0adcecb34891410f77633b08f7bb4e74b60'
MM_IDS = ['mm-ref2va', 'mm-fl2va', 'mm-enc-int4', 'mm-enc-nvfp4', 'mm-vae-int8', 'mm-vae-fp16', 'mm-vae-audio',
          'mm-latent-up', 'mm-gemma-e4b', 'mm-gemma-e2b', 'mm-lora-turbo', 'mm-lora-taomate']
NODES = ['ComfyUI-KJNodes', 'rgthree-comfy', 'ComfyUI-Easy-Use', 'ComfyUI-Logic', 'ComfyUI-Spectrum-MiniMax-H3',
         'Comfyui_Minimax_h3_latent_Upscaler', 'comfyui-various', 'ComfyUI-VideoHelperSuite', 'comfyui_essential-er']


def all_nodes(w):
    return w['nodes'] + [n for g in w['definitions']['subgraphs'] for n in g['nodes']]


def start_install(monkeypatch, body):
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url, **kwargs):
            data = {'queue_running': [], 'queue_pending': []}
            return httpx.Response(200, json=data, request=httpx.Request('GET', url))
    monkeypatch.setattr(httpx, 'AsyncClient', Client)
    captured = {}
    async def start(workflow):
        captured.update(workflow)
        return {'status': 'running'}
    monkeypatch.setenv('SELFISM_AUTO_REPAIR', '0')
    monkeypatch.setenv('CIVITAI_TOKEN', 'test-only')
    monkeypatch.setattr(m.selfism_controller, 'start', start)
    for c, attr in [(m.controller, 'task'), (m.custom_model_controller, 'worker_task'),
                    (m.custom_node_controller, 'worker_task'), (m.comfy_service_controller, 'task'),
                    (m.selfism_controller, 'task')]:
        monkeypatch.setattr(c, attr, None)
    with TestClient(m.app) as client:
        r = client.post('/api/selfism/install', json=body)
        return r, captured, client.get('/').text


def test_minimax_catalog_metadata_and_r2_layout():
    files = CATALOG['files']
    for k in MM_IDS:
        f = files[k]
        assert re.fullmatch(r'[0-9a-f]{64}', f['sha256']), k
        assert f['size_bytes'] > 0 and f['destination'].startswith('models/'), k
        # The private-R2 resolver derives the key from the destination, so the layout must match the mirrored keys.
        assert f['destination'].removeprefix('models/').split('/')[0] in (
            'diffusion_models', 'text_encoders', 'vae', 'latent_upscale_models', 'loras'), k
    assert len({files[k]['destination'] for k in MM_IDS}) == len(MM_IDS)
    assert files['mm-lora-taomate']['auth'] == 'civitai'
    assert all(files[k]['url'].startswith('https://huggingface.co/') for k in MM_IDS if k != 'mm-lora-taomate')
    # the optional models stay individually installable
    assert set(MM_IDS) <= set(files)
    names = {n['name']: n for n in CATALOG['nodes']}
    assert set(CATALOG['minimax_nodes']) == set(NODES) and set(NODES) <= set(names)
    for n in NODES:
        assert re.fullmatch(r'[0-9a-f]{40}', names[n]['ref']) and names[n]['repo'].startswith('https://github.com/'), n


def test_minimax_install_selects_only_default_active_files(monkeypatch):
    r, captured, html = start_install(monkeypatch, {'profile': 'minimax'})
    assert r.status_code == 200
    ids = [f['id'] for f in captured['files']]
    assert ids == CATALOG['minimax_files'] == ['mm-ref2va', 'mm-enc-int4', 'mm-vae-int8', 'mm-vae-audio',
                                               'mm-latent-up', 'mm-gemma-e4b']
    assert captured['selfism_profile'] == 'minimax' and captured['selfism_repair'] is False
    assert {n['name'] for n in captured['custom_nodes']} == set(NODES)
    assert all(len(f['sha256']) == 64 for f in captured['files'])
    assert 'data-sf-action="minimax"' in html and 'MiniMax H3 Simply Advanced' in html
    total = sum(CATALOG['files'][k]['size_bytes'] for k in ids)
    assert f'{total/1e9:.1f}'.replace('.', ',') + ' GB' in html
    scripts = (ROOT/'scripts/selfism-section.html').read_text(encoding='utf-8')
    assert 'data-sf-action="minimax"' in scripts and f'{total/1e9:.1f}'.replace('.', ',') + ' GB' in scripts


def test_minimax_workflow_is_unchanged_and_default_models_are_covered():
    data = WF_PATH.read_bytes()
    assert hashlib.sha256(data).hexdigest() == WF_SHA256
    w = json.loads(data)
    needed = {Path(CATALOG['files'][k]['destination']).name for k in CATALOG['minimax_files']}
    nodes = all_nodes(w)
    # Subgraph "Load Models" instance: promoted widget values (unet, clip, video vae, audio vae) are what ComfyUI loads.
    inst = next(n for n in w['nodes'] if n['id'] == 4595)
    assert set(inst['widgets_values']) <= needed
    for n in nodes:
        if n.get('mode', 0) != 0: continue
        v = n.get('widgets_values')
        if n['type'] == 'MinimaxH3LatentUpscaler3D': assert v[0] in needed
        if n['type'] == 'CLIPLoader' and v[0].startswith('gemma4'): assert v[0] in needed
    # no LoRA rows are selected by default, so no LoRA is part of the default install
    for n in nodes:
        if n['type'] == 'Power Lora Loader (rgthree)':
            assert not any(isinstance(x, dict) and 'lora' in x for x in n['widgets_values'])


def test_minimax_workflow_download_and_install_is_byte_identical_and_nondestructive(monkeypatch, tmp_path):
    with TestClient(m.app) as client:
        r = client.get('/api/selfism/workflow/minimax')
    assert r.status_code == 200 and r.content == WF_PATH.read_bytes()
    assert s.MINIMAX_WORKFLOW_NAME in r.headers['content-disposition']
    monkeypatch.setattr(m, 'COMFYUI_DIR', tmp_path)
    monkeypatch.setattr(m, 'COMFYUI_VENV', tmp_path/'.venv')
    python = m.COMFYUI_VENV/'bin/python'
    python.parent.mkdir(parents=True); python.touch()
    ctrl = type(m.selfism_controller)()
    async def ready(*args): pass
    async def run(*args, **kwargs): return 0, 'torch==2.8.0'
    async def sources(host, files): return files, []
    monkeypatch.setattr(ctrl, '_wait_for_comfyui', ready)
    monkeypatch.setattr(ctrl, '_run_process', run)
    monkeypatch.setattr(m.JobController, '_install_workflow', ready)
    monkeypatch.setattr(s, 'resolve_sources', sources)
    saved = tmp_path/'user/default/workflows/Selfism'/s.MINIMAX_WORKFLOW_NAME
    asyncio.run(ctrl._install_workflow({'selfism_profile': 'minimax', 'precision': 'fp8', 'files': [{'id': 'x'}]}))
    assert saved.read_bytes() == WF_PATH.read_bytes()
    saved.write_text('{"user_edited":true}')
    asyncio.run(ctrl._install_workflow({'selfism_profile': 'minimax', 'precision': 'fp8', 'files': [{'id': 'x'}]}))
    assert json.loads(saved.read_text()) == {'user_edited': True}

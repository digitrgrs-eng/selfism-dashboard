import asyncio
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
WORKFLOW = json.loads((ROOT/'selfism_workflows/full.json').read_text(encoding='utf-8'))


def all_nodes(w):
    return w['nodes'] + [n for g in w['definitions']['subgraphs'] for n in g['nodes']]


def start_install(monkeypatch, body):
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url, **kwargs):
            if url.endswith('/queue'):
                data = {'queue_running': [], 'queue_pending': []}
            else:
                version = int(url.rsplit('/', 1)[1])
                f = next(x for x in CATALOG['files'].values() if x.get('civitai_version') == version)
                data = {'files': [{'id': f['civitai_file'], 'hashes': {'SHA256': f['sha256'].upper()}}]}
            return httpx.Response(200, json=data, request=httpx.Request('GET', url))
    captured = {}
    async def start(workflow):
        captured.update(workflow)
        return {'status': 'running'}
    monkeypatch.setenv('SELFISM_AUTO_REPAIR', '0')
    monkeypatch.setenv('CIVITAI_TOKEN', 'test-only')
    monkeypatch.setattr(httpx, 'AsyncClient', Client)
    monkeypatch.setattr(m.selfism_controller, 'start', start)
    for c, attr in [(m.controller, 'task'), (m.custom_model_controller, 'worker_task'),
                    (m.custom_node_controller, 'worker_task'), (m.comfy_service_controller, 'task'),
                    (m.selfism_controller, 'task')]:
        monkeypatch.setattr(c, attr, None)
    with TestClient(m.app) as client:
        r = client.post('/api/selfism/install', json=body)
        return r, captured, client.get('/').text


def test_full_catalog_metadata_is_real():
    files = CATALOG['files']
    for key in CATALOG['full_files'] + ['int8', 'fp8']:
        f = files[key]
        assert re.fullmatch(r'[0-9a-f]{64}', f['sha256']), key
        assert f['size_bytes'] > 0, key
    assert len({files[k]['destination'] for k in CATALOG['full_files']}) == len(CATALOG['full_files'])
    assert 'bf16' not in CATALOG['full_files']
    names = {n['name'] for n in CATALOG['nodes']}
    assert set(CATALOG['full_nodes']) <= names


@pytest.mark.parametrize('body,expected', [({'profile': 'full'}, 'int8'),
                                          ({'profile': 'full', 'precision': 'int8'}, 'int8'),
                                          ({'profile': 'full', 'precision': 'fp8'}, 'fp8')])
def test_full_install_selects_int8_default_or_fp8(monkeypatch, body, expected):
    r, captured, html = start_install(monkeypatch, body)
    assert r.status_code == 200
    ids = [f['id'] for f in captured['files']]
    assert ids[0] == expected and not ({'int8', 'fp8', 'bf16'} - {expected}) & set(ids)
    assert set(CATALOG['full_files']) <= set(ids)
    assert captured['precision'] == expected and captured['selfism_profile'] == 'full'
    assert captured['selfism_repair'] is True
    assert {n['name'] for n in captured['custom_nodes']} == set(CATALOG['full_nodes'])
    assert all(len(f['sha256']) == 64 for f in captured['files'])
    assert {l['source'] for l in captured['model_links']} <= {f['destination'] for f in captured['files']}
    assert 'data-sf-action="full"' in html and 'Instaliraj sve' in html
    assert '/api/selfism/workflow/full' in html


def test_full_rejects_bf16(monkeypatch):
    r, captured, _ = start_install(monkeypatch, {'profile': 'full', 'precision': 'bf16'})
    assert r.status_code == 400 and not captured


def test_full_card_size_matches_catalog():
    files = CATALOG['files']
    base = sum(files[k]['size_bytes'] for k in CATALOG['full_files'])
    html = (ROOT/'launcher/static/index.html').read_text(encoding='utf-8')
    assert f'{(base+files["int8"]["size_bytes"])/1e9:.0f} GB (INT8)' in html
    assert f'{(base+files["fp8"]["size_bytes"])/1e9:.1f}'.replace('.', ',') + ' GB (FP8)' in html


def test_full_workflow_download_and_static_model_coverage():
    with TestClient(m.app) as client:
        w = client.get('/api/selfism/workflow/full').json()
    assert w == WORKFLOW
    names = {Path(CATALOG['files'][k]['destination']).name for k in CATALOG['full_files'] + ['int8', 'fp8']}
    names |= {'ComfyUI_temp'}
    for n in all_nodes(w):
        t, v = n['type'], n.get('widgets_values')
        if t in ('UNETLoader', 'VAELoader', 'CLIPLoader', 'UpscaleModelLoader', 'SAMLoader',
                 'Krea2ControlLoRALoader', 'DepthAnythingV2Preprocessor', 'LoraLoaderModelOnly'):
            assert Path(v[0]).name in names, (t, v[0])
        if t == 'UltralyticsDetectorProvider':
            assert Path(v[0]).name in names, v[0]
        if t == 'ArtfatLLMPrompter':
            assert v[0] in names and v[1] in names
        if t == 'DWPreprocessor':
            assert v[4] in names and v[5] in names
        if t == 'Power Lora Loader (rgthree)':
            for row in v:
                if isinstance(row, dict) and 'lora' in row:
                    assert row['lora'] in names, row['lora']
    assert not any('\\' in json.dumps(n.get('widgets_values')) and '.safetensors' in json.dumps(n.get('widgets_values'))
                   for n in all_nodes(w) if n['type'] in ('UNETLoader', 'Power Lora Loader (rgthree)'))


def test_full_workflow_saved_once_with_chosen_precision_and_nondestructive(monkeypatch, tmp_path):
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
    folder = tmp_path/'user/default/workflows/Selfism'
    asyncio.run(ctrl._install_workflow({'selfism_profile': 'full', 'precision': 'fp8', 'files': [{'id': 'x'}]}))
    saved = folder/'Selfism_FULL_fp8_v1.json'
    w = json.loads(saved.read_text(encoding='utf-8'))
    assert [n['widgets_values'][0] for n in w['nodes'] if n['type'] == 'UNETLoader'] == [
        'selforaV21NightFix_selforaV21Fp8.safetensors']
    saved.write_text('{"user_edited":true}')
    asyncio.run(ctrl._install_workflow({'selfism_profile': 'full', 'precision': 'fp8', 'files': [{'id': 'x'}]}))
    assert json.loads(saved.read_text()) == {'user_edited': True}
    asyncio.run(ctrl._install_workflow({'selfism_profile': 'full', 'precision': 'int8', 'files': [{'id': 'x'}]}))
    w = json.loads((folder/'Selfism_FULL_int8_v1.json').read_text(encoding='utf-8'))
    assert [n['widgets_values'][0] for n in w['nodes'] if n['type'] == 'UNETLoader'] == [
        'selforaV21NightFix_selfora21Int8.safetensors']

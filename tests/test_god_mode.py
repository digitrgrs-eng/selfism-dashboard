"""GOD Mode card: Wan 2.2 Animate workflow + R2-backed catalog assets."""
from __future__ import annotations
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
CATALOG = json.loads((ROOT / 'catalog/selfism.json').read_text(encoding='utf-8'))
GOD_WF = ROOT / 'selfism_workflows/wan22_animate_god_mode.json'
GOD_IDS = [
    'wan-animate-14b-bf16', 'umt5-xxl-fp16', 'pusa-lora', 'vitpose-h-bin', 'lightx2v-t2v-lora',
    'wan-relight-lora', 'clip-vision-h', 'lightx2v-i2v-4step', 'fun-mps-lora', 'sam2-hiera-bp',
    'wan-vae-bf16', 'yolov10m', 'rife49', 'vitpose-h-onnx',
]
GOD_NODES = [
    'ComfyUI-WanVideoWrapper', 'ComfyUI-WanAnimatePreprocess', 'ComfyUI-segment-anything-2',
    'ComfyUI-Frame-Interpolation', 'ComfyMath', 'ComfyUI-KJNodes', 'ComfyUI-Easy-Use',
    'ComfyUI-VideoHelperSuite', 'rgthree-comfy', 'ComfyUI-Custom-Scripts',
]
GOD_MODELS = {
    'wan2.2_animate_14B_bf16.safetensors',
    'umt5_xxl_fp16.safetensors',
    'clip_vision_h.safetensors',
    'Wan2_1_VAE_bf16.safetensors',
    'sam2.1_hiera_base_plus.safetensors',
    'vitpose_h_wholebody_model.onnx',
    'vitpose_h_wholebody_data.bin',
    'yolov10m.onnx',
    'wan2.2_animate_14B_relight_lora_bf16.safetensors',
    'lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors',
    'wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors',
    'Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors',
    'Wan2.2-Fun-A14B-InP-low-noise-MPS.safetensors',
    'rife49.pth',
}
CNR_TO_PACK = {
    'ComfyUI-WanVideoWrapper': 'ComfyUI-WanVideoWrapper',
    'comfyui-wanvideowrapper': 'ComfyUI-WanVideoWrapper',
    'ComfyUI-WanAnimatePreprocess': 'ComfyUI-WanAnimatePreprocess',
    'ComfyUI-segment-anything-2': 'ComfyUI-segment-anything-2',
    'comfyui-frame-interpolation': 'ComfyUI-Frame-Interpolation',
    'ComfyMath': 'ComfyMath',
    'comfyui-kjnodes': 'ComfyUI-KJNodes',
    'comfyui-easy-use': 'ComfyUI-Easy-Use',
    'comfyui-videohelpersuite': 'ComfyUI-VideoHelperSuite',
    'rgthree-comfy': 'rgthree-comfy',
    'comfyui-custom-scripts': 'ComfyUI-Custom-Scripts',
}
AUX_TO_PACK = {
    'kijai/ComfyUI-WanVideoWrapper': 'ComfyUI-WanVideoWrapper',
    'kijai/ComfyUI-WanAnimatePreprocess': 'ComfyUI-WanAnimatePreprocess',
    'lehych-ai/ComfyUI-WanAnimatePreprocess': 'ComfyUI-WanAnimatePreprocess',
    'kijai/ComfyUI-segment-anything-2': 'ComfyUI-segment-anything-2',
    'Fannovel16/ComfyUI-Frame-Interpolation': 'ComfyUI-Frame-Interpolation',
    'evanspearman/ComfyMath': 'ComfyMath',
    'kijai/ComfyUI-KJNodes': 'ComfyUI-KJNodes',
    'aining2022/ComfyUI_Swwan': 'ComfyUI-KJNodes',
    'yolain/ComfyUI-Easy-Use': 'ComfyUI-Easy-Use',
    'Kosinkadink/ComfyUI-VideoHelperSuite': 'ComfyUI-VideoHelperSuite',
    'rgthree/rgthree-comfy': 'rgthree-comfy',
    'pythongosssss/ComfyUI-Custom-Scripts': 'ComfyUI-Custom-Scripts',
}
REEL_V3_IDS = ['mm-ref2va', 'r2v-enc', 'mm-vae-fp16', 'mm-vae-audio']
REEL_V2_IDS = ['mm-ref2va', 'r2v-enc', 'mm-vae-fp16', 'mm-vae-audio', 'r2v-lora-turbo-8step']


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


def test_god_mode_catalog_files_nodes_links_and_unchanged_siblings():
    files = CATALOG['files']
    assert CATALOG['god_mode_files'] == GOD_IDS
    assert CATALOG['god_mode_nodes'] == GOD_NODES
    assert CATALOG.get('god_mode_pip') == []
    assert CATALOG['god_mode_links'] == [{
        'source': 'models/rife/rife49.pth',
        'destination': 'custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife49.pth',
    }]
    for k in GOD_IDS:
        f = files[k]
        assert re.fullmatch(r'[0-9a-f]{64}', f['sha256']) and f['size_bytes'] > 0 and f['auth'] == 'none', k
        assert f['destination'].startswith('models/') and f['url'].startswith('https://huggingface.co/'), k
        assert f['id'] == k
    installed = {Path(files[k]['destination']).name for k in GOD_IDS}
    assert installed == GOD_MODELS
    # R2 key layout from destination
    for k in GOD_IDS:
        top = files[k]['destination'].removeprefix('models/').split('/')[0]
        assert top in ('diffusion_models', 'text_encoders', 'clip_vision', 'vae', 'loras', 'sam2', 'detection', 'rife'), k
    names = {n['name']: n for n in CATALOG['nodes']}
    assert set(GOD_NODES) <= set(names)
    for n in GOD_NODES:
        assert names[n]['repo'].startswith('https://github.com/') and re.fullmatch(r'[0-9a-f]{40}', names[n]['ref']), n
    # sibling MiniMax cards unchanged
    assert CATALOG['reel_recreation_v3_files'] == REEL_V3_IDS
    assert CATALOG['reel_recreation_v2_files'] == REEL_V2_IDS
    assert CATALOG['r2v_swap_highres_nodes'] == ['rgthree-comfy', 'ComfyUI-VideoHelperSuite', 'ComfyUI-KJNodes']


def test_god_mode_install_selects_files_nodes_and_ui_card(monkeypatch):
    r, captured, html = start_install(monkeypatch, {'profile': 'god_mode'})
    assert r.status_code == 200
    assert [f['id'] for f in captured['files']] == GOD_IDS
    assert captured['selfism_profile'] == 'god_mode' and captured['selfism_repair'] is False
    assert captured['pip_packages'] == []
    assert {n['name'] for n in captured['custom_nodes']} == set(GOD_NODES)
    assert captured['model_links'] == CATALOG['god_mode_links']
    total = sum(CATALOG['files'][k]['size_bytes'] for k in GOD_IDS)
    scripts = (ROOT / 'scripts/selfism-section.html').read_text(encoding='utf-8')
    gb = f'{total / 1e9:.1f}'.replace('.', ',') + ' GB'
    for page in (html, scripts):
        assert 'data-sf-action="god_mode"' in page
        assert '>GOD Mode</h2>' in page
        assert 'Instaliraj</button>' in page
        assert gb in page
        # siblings stay
        assert 'data-sf-action="minimax_reel_recreation_v3"' in page
        assert 'data-sf-action="minimax_reel_recreation_v2"' in page


def test_god_mode_workflow_covers_catalog_models_and_nodes():
    w = json.loads(GOD_WF.read_bytes())
    assert w.get('id')
    used = set()
    for n in w['nodes']:
        named = n.get('widgets_values_named') or {}
        vals = n.get('widgets_values')
        for v in list(named.values()) + (vals if isinstance(vals, list) else list(vals.values()) if isinstance(vals, dict) else []):
            if isinstance(v, str) and v.lower().endswith(('.safetensors', '.onnx', '.pth', '.bin', '.ckpt', '.pt')) and v.lower() != 'none':
                used.add(v)
    # vitpose data.bin is required beside the onnx but not named in widgets
    assert (used | {'vitpose_h_wholebody_data.bin'}) == GOD_MODELS
    installed = {Path(CATALOG['files'][k]['destination']).name for k in CATALOG['god_mode_files']}
    assert used <= installed | {'vitpose_h_wholebody_data.bin'}
    packs = set()
    for n in w['nodes']:
        props = n.get('properties') or {}
        cnr = props.get('cnr_id')
        aux = props.get('aux_id')
        if cnr in (None, 'comfy-core'):
            if aux in AUX_TO_PACK:
                packs.add(AUX_TO_PACK[aux])
            continue
        if cnr in CNR_TO_PACK:
            packs.add(CNR_TO_PACK[cnr])
        elif aux in AUX_TO_PACK:
            packs.add(AUX_TO_PACK[aux])
    assert packs <= set(GOD_NODES)
    for required in ('ComfyUI-WanVideoWrapper', 'ComfyUI-WanAnimatePreprocess', 'ComfyUI-segment-anything-2',
                     'ComfyUI-Frame-Interpolation', 'ComfyUI-VideoHelperSuite'):
        assert required in packs


def test_god_mode_workflow_download_and_install_are_byte_identical_and_nondestructive(monkeypatch, tmp_path):
    with TestClient(m.app) as client:
        r = client.get('/api/selfism/workflow/god_mode')
    assert r.status_code == 200 and r.content == GOD_WF.read_bytes()
    assert s.GOD_MODE_WORKFLOW_NAME in r.headers['content-disposition']
    monkeypatch.setattr(m, 'COMFYUI_DIR', tmp_path)
    monkeypatch.setattr(m, 'COMFYUI_VENV', tmp_path / '.venv')
    python = m.COMFYUI_VENV / 'bin/python'
    python.parent.mkdir(parents=True)
    python.touch()
    ctrl = type(m.selfism_controller)()
    async def ready(*args): pass
    async def run(*args, **kwargs): return 0, 'torch==2.8.0'
    async def sources(host, files): return files, []
    monkeypatch.setattr(ctrl, '_wait_for_comfyui', ready)
    monkeypatch.setattr(ctrl, '_run_process', run)
    monkeypatch.setattr(m.JobController, '_install_workflow', ready)
    monkeypatch.setattr(s, 'resolve_sources', sources)
    saved = tmp_path / 'user/default/workflows/Selfism' / s.GOD_MODE_WORKFLOW_NAME
    wf = {'selfism_profile': 'god_mode', 'precision': 'fp8', 'files': [{'id': 'x'}], 'pip_packages': []}
    asyncio.run(ctrl._install_workflow(wf))
    assert saved.read_bytes() == GOD_WF.read_bytes()
    assert not (tmp_path / 'user/default/workflows/Selfism' / s.MINIMAX_REEL_RECREATION_V3_WORKFLOW_NAME).exists()
    saved.write_text('{"user_edited":true}')
    asyncio.run(ctrl._install_workflow(wf))
    assert json.loads(saved.read_text()) == {'user_edited': True}


def test_god_mode_does_not_touch_cu130_bake_pins():
    dockerfile = (ROOT / 'Dockerfile').read_text(encoding='utf-8')
    bake = (ROOT / 'docker/bake_comfyui.sh').read_text(encoding='utf-8')
    assert 'cu130' in dockerfile or 'cu130' in bake
    # GOD Mode changes must not rewrite torch pins
    assert 'torch==2.10.0+cu130' in bake or 'cu130' in bake
    assert 'god_mode' not in bake.lower()

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
         'Comfyui_Minimax_h3_latent_Upscaler', 'comfyui-various', 'ComfyUI-VideoHelperSuite', 'comfyui_essential-er',
         'ComfyUI-LLM-text-processor', 'ComfyUI-MiniMaxH3Mod']
# ComfyUI-Logic must stay on the 1.0.0 line: the workflow uses node type "Bool", and the later 1.0.1 pin registers it as
# "Bool-🔬", which ComfyUI Manager then reports as a missing node.
LOGIC_REF = 'f95b332091ec1d6a8b1df4bedf1b054e9807f5e2'
# VHS_LoadVideo's `format: H3` exists from Kosinkadink/ComfyUI-VideoHelperSuite 593fcb0 ("Support h3 as input format");
# older pins fail validation with "Value not in list: format: 'H3'". The workflow was saved with this exact commit.
VHS_REF = '4d907bee61e92c2e65af3bd6383a4e4d356126d1'
# Every non-core node type used by the workflow, mapped to the catalog pack that provides it. The workflow's own
# cnr_id / aux_id properties name the source pack; types without either come from the pack that registers them.
TYPE_TO_PACK = {
    'Bool': 'ComfyUI-Logic', 'JWDatetimeString': 'comfyui-various',
    'LLMTextProcessor': 'ComfyUI-LLM-text-processor',
    'MiniMaxH3RefModApply': 'ComfyUI-MiniMaxH3Mod', 'MiniMaxH3RefModsLoader': 'ComfyUI-MiniMaxH3Mod',
    'MinimaxH3LatentUpscaler3D': 'Comfyui_Minimax_h3_latent_Upscaler', 'SpectrumApplyMiniMaxH3': 'ComfyUI-Spectrum-MiniMax-H3',
    'VHS_LoadVideo': 'ComfyUI-VideoHelperSuite', 'ResizeImageMaskAlt': 'comfyui_essential-er',
    'Any Switch (rgthree)': 'rgthree-comfy', 'Power Lora Loader (rgthree)': 'rgthree-comfy',
    'GetNode': 'ComfyUI-KJNodes', 'SetNode': 'ComfyUI-KJNodes', 'ImageBatchMulti': 'ComfyUI-KJNodes',
}
CNR_TO_PACK = {
    'comfyui-logic': 'ComfyUI-Logic', 'jameswalker-nodes': 'comfyui-various', 'comfyui-various': 'comfyui-various',
    'ComfyUI-LLM-text-processor': 'ComfyUI-LLM-text-processor', 'ComfyUI-MiniMaxH3Mod': 'ComfyUI-MiniMaxH3Mod',
    'comfyui-kjnodes': 'ComfyUI-KJNodes', 'rgthree-comfy': 'rgthree-comfy', 'comfyui-easy-use': 'ComfyUI-Easy-Use',
    'comfyui-videohelpersuite': 'ComfyUI-VideoHelperSuite', 'comfyui-spectrum-minimax-h3': 'ComfyUI-Spectrum-MiniMax-H3',
}
AUX_TO_PACK = {
    'altoiddealer/ComfyUI-Easy-Use-alt': 'ComfyUI-Easy-Use', 'altoiddealer/comfyui_essential-er': 'comfyui_essential-er',
    'kijai/ComfyUI-KJNodes': 'ComfyUI-KJNodes',
    'LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler': 'Comfyui_Minimax_h3_latent_Upscaler',
}


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
    assert names['ComfyUI-Logic']['ref'] == LOGIC_REF
    assert names['ComfyUI-VideoHelperSuite']['ref'] == VHS_REF
    assert CATALOG['minimax_pip'] == ['soundfile']
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
    assert captured['pip_packages'] == ['soundfile']
    assert all(len(f['sha256']) == 64 for f in captured['files'])
    for label in ('LLM Text Processor', 'MiniMax H3 RefMod', 'soundfile'):
        assert label in html and label in (ROOT/'scripts/selfism-section.html').read_text(encoding='utf-8')
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


def test_every_non_core_node_in_the_workflow_is_installed_by_the_minimax_card():
    w = json.loads(WF_PATH.read_bytes())
    installed = set(CATALOG['minimax_nodes'])
    seen = {}
    for n in all_nodes(w):
        props = n.get('properties') or {}
        cnr, aux = props.get('cnr_id'), props.get('aux_id')
        if cnr and cnr != 'comfy-core':
            assert cnr in CNR_TO_PACK, f"{n['type']}: unmapped cnr_id {cnr}"
            seen[cnr] = CNR_TO_PACK[cnr]
        if aux:
            assert aux in AUX_TO_PACK, f"{n['type']}: unmapped aux_id {aux}"
            seen[aux] = AUX_TO_PACK[aux]
        if n['type'] in TYPE_TO_PACK: seen[n['type']] = TYPE_TO_PACK[n['type']]
    # the three packs ComfyUI Manager reported as missing on a freshly installed pod are really used by the workflow
    assert {'comfyui-logic', 'jameswalker-nodes', 'ComfyUI-LLM-text-processor', 'ComfyUI-MiniMaxH3Mod'} <= set(seen)
    assert set(seen.values()) <= installed, set(seen.values()) - installed
    assert {'Bool', 'JWDatetimeString', 'LLMTextProcessor', 'MiniMaxH3RefModApply', 'MiniMaxH3RefModsLoader'} <= set(seen)


def test_minimax_install_pip_installs_soundfile_after_the_packs_and_checks_the_import(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'COMFYUI_DIR', tmp_path)
    monkeypatch.setattr(m, 'COMFYUI_VENV', tmp_path/'.venv')
    python = m.COMFYUI_VENV/'bin/python'
    python.parent.mkdir(parents=True); python.touch()
    ctrl = type(m.selfism_controller)()
    calls = []
    async def ready(*args): calls.append('packs')
    async def run(*args, **kwargs):
        calls.append([str(a) for a in args[1:]])
        return 0, 'torch==2.8.0'
    async def sources(host, files): return files, []
    monkeypatch.setattr(ctrl, '_wait_for_comfyui', lambda: ready())
    monkeypatch.setattr(ctrl, '_run_process', run)
    monkeypatch.setattr(m.JobController, '_install_workflow', ready)
    monkeypatch.setattr(s, 'resolve_sources', sources)
    wf = {'selfism_profile': 'minimax', 'precision': 'fp8', 'files': [{'id': 'x'}], 'pip_packages': ['soundfile']}
    asyncio.run(ctrl._install_workflow(wf))
    pip = [c for c in calls if isinstance(c, list) and c[:3] == ['-m', 'pip', 'install']]
    assert len(pip) == 1 and pip[0][-1] == 'soundfile'
    assert calls.index('packs') < calls.index(pip[0]) and ['-c', 'import soundfile'] in calls
    # a failing pip install must fail the card instead of being reported as success
    async def failing(*args, **kwargs): return (1, 'boom') if 'pip' in args else (0, 'torch==2.8.0')
    monkeypatch.setattr(ctrl, '_run_process', failing)
    with pytest.raises(RuntimeError, match='soundfile'):
        asyncio.run(ctrl._install_workflow(wf))
    # catalog values never reach the shell unchecked
    with pytest.raises(RuntimeError, match='Unsafe'):
        asyncio.run(ctrl._install_workflow({**wf, 'pip_packages': ['soundfile; rm -rf /']}))


def test_vhs_pin_is_the_commit_the_workflow_was_saved_with_and_supports_the_h3_format():
    w = json.loads(WF_PATH.read_bytes())
    loaders = [n for n in all_nodes(w) if n['type'] == 'VHS_LoadVideo']
    assert loaders and all(n['widgets_values']['format'] == 'H3' for n in loaders)
    assert {n['properties']['ver'] for n in loaders} == {VHS_REF}
    ref = {n['name']: n['ref'] for n in CATALOG['nodes']}['ComfyUI-VideoHelperSuite']
    assert ref == VHS_REF
    # 593fcb07 (Support h3 as input format) is an ancestor of the pin; the previous pin predates it
    assert ref != '4ee72c065db22c9d96c2427954dc69e7b908444b'

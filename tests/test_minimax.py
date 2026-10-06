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


# ---- MiniMax H3 R2V Turbo (HearmemanAI) card -------------------------------------------------------------------
R2V_WF = ROOT/'selfism_workflows/minimax_h3_r2v_turbo_hearmeman.json'
# sha256 of the HearmemanAI workflow exactly as supplied. The shipped copy differs by three string literals only: the
# UNETLoader value lost its "diffusion_models/" prefix (the file is installed at models/diffusion_models/<name>, so the
# prefixed value is "not in list"), and that node's download-metadata entry pointed at the wrong (fl2va) file.
R2V_ORIGINAL_SHA256 = '95a4390a4a50082bc381bd2b4aa8f42792f0ac476830c6073527ed504d10e827'
R2V_IDS = ['r2v-unet', 'r2v-enc', 'mm-vae-fp16', 'mm-vae-audio', 'r2v-lora-turbo']
R2V_NODES = ['rgthree-comfy', 'ComfyUI-KJNodes', 'ComfyUI-VideoHelperSuite', 'ComfyUI-MiniMaxRefPack']
R2V_CNR_TO_PACK = {'rgthree-comfy': 'rgthree-comfy', 'comfyui-kjnodes': 'ComfyUI-KJNodes',
                   'comfyui-videohelpersuite': 'ComfyUI-VideoHelperSuite', 'comfyui-minimaxrefpack': 'ComfyUI-MiniMaxRefPack'}


def test_r2v_catalog_files_nodes_and_r2_layout():
    files = CATALOG['files']
    assert CATALOG['r2v_files'] == R2V_IDS and CATALOG['r2v_nodes'] == R2V_NODES
    for k in R2V_IDS:
        f = files[k]
        assert re.fullmatch(r'[0-9a-f]{64}', f['sha256']) and f['size_bytes'] > 0 and f['auth'] == 'none', k
        assert f['destination'].startswith('models/') and f['url'].startswith('https://huggingface.co/'), k
    # destinations double as the R2 keys (destination minus "models/") that were uploaded for these files
    assert {k: files[k]['destination'].removeprefix('models/') for k in R2V_IDS if k.startswith('r2v-')} == {
        'r2v-unet': 'diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors',
        'r2v-enc': 'text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
        'r2v-lora-turbo': 'loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors'}
    # taeh3 stays in the catalog for optional install, but is not part of the card (preview is pinned to none)
    assert 'r2v-taeh3' in files and files['r2v-taeh3']['destination'] == 'models/vae_approx/taeh3.safetensors'
    assert len({files[k]['destination'] for k in R2V_IDS}) == len(R2V_IDS)
    names = {n['name']: n for n in CATALOG['nodes']}
    assert set(R2V_NODES) <= set(names)
    for n in R2V_NODES:
        assert re.fullmatch(r'[0-9a-f]{40}', names[n]['ref']) and names[n]['repo'].startswith('https://github.com/'), n
    # shared packs keep the pins of the Simply Advanced card; the new pack is the 0.3.5 release the workflow was saved with
    assert names['ComfyUI-VideoHelperSuite']['ref'] == VHS_REF
    assert names['ComfyUI-MiniMaxRefPack']['ref'] == '7012734eabf6f98063d6eaf8ce1f9264ee803664'
    assert names['ComfyUI-MiniMaxRefPack']['repo'] == 'https://github.com/Hearmeman24/ComfyUI-MiniMaxRefPack.git'


def test_r2v_install_selects_the_workflow_files_only(monkeypatch):
    r, captured, html = start_install(monkeypatch, {'profile': 'minimax_r2v'})
    assert r.status_code == 200
    assert [f['id'] for f in captured['files']] == R2V_IDS
    assert captured['selfism_profile'] == 'minimax_r2v' and captured['selfism_repair'] is False
    assert captured['pip_packages'] == []
    assert {n['name'] for n in captured['custom_nodes']} == set(R2V_NODES)
    total = sum(CATALOG['files'][k]['size_bytes'] for k in R2V_IDS)
    scripts = (ROOT/'scripts/selfism-section.html').read_text(encoding='utf-8')
    for page in (html, scripts):
        assert 'data-sf-action="minimax_r2v"' in page and 'MiniMax H3 R2V Turbo (Hearmeman)' in page
        assert f'{total/1e9:.1f}'.replace('.', ',') + ' GB' in page
    # the optional NSFW LoRAs are not part of the card
    assert not any('HM' in CATALOG['files'][k]['destination'] for k in R2V_IDS)


def test_r2v_workflow_keeps_the_original_graph_with_safe_defaults_and_every_model_is_installed():
    w = json.loads(R2V_WF.read_bytes())
    assert 'definitions' not in w and not any(n['type'] == 'ResizeImageMaskNode' for n in w['nodes'])
    installed = {Path(CATALOG['files'][k]['destination']).name for k in CATALOG['r2v_files']}
    used = {}
    for n in w['nodes']:
        v = n.get('widgets_values')
        if n['type'] in ('UNETLoader', 'CLIPLoader', 'VAELoader', 'LoraLoaderModelOnly'): used[n['id']] = v[0]
        if n['type'] == 'ModelPreviewOverrideKJ':
            assert v[5] == 'none' and v[2] is True  # tiny_vae off; suppress default preview
        if n['type'] == 'Power Lora Loader (rgthree)':
            on = [x['lora'] for x in v if isinstance(x, dict) and x.get('on')]
            assert on == []  # hmmotion and every HM* row are off
            assert any(isinstance(x, dict) and x.get('lora') == 'hmmotion_minimax-h3_epoch40.safetensors' and x.get('on') is False for x in v)
    assert set(used.values()) == installed and len(used) == 5, set(used.values()) ^ installed
    cnr = {(n.get('properties') or {}).get('cnr_id') for n in w['nodes']} - {None, 'comfy-core'}
    assert {R2V_CNR_TO_PACK[c] for c in cnr} <= set(CATALOG['r2v_nodes']) and cnr <= set(R2V_CNR_TO_PACK)
    for n in w['nodes']:
        for mdl in (n.get('properties') or {}).get('models', []):
            assert mdl['name'] in installed, mdl



def test_r2v_workflow_download_and_install_are_byte_identical_and_nondestructive(monkeypatch, tmp_path):
    with TestClient(m.app) as client:
        r = client.get('/api/selfism/workflow/minimax_r2v')
    assert r.status_code == 200 and r.content == R2V_WF.read_bytes()
    assert s.MINIMAX_R2V_WORKFLOW_NAME in r.headers['content-disposition']
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
    saved = tmp_path/'user/default/workflows/Selfism'/s.MINIMAX_R2V_WORKFLOW_NAME
    wf = {'selfism_profile': 'minimax_r2v', 'precision': 'fp8', 'files': [{'id': 'x'}], 'pip_packages': []}
    asyncio.run(ctrl._install_workflow(wf))
    assert saved.read_bytes() == R2V_WF.read_bytes()
    saved.write_text('{"user_edited":true}')
    asyncio.run(ctrl._install_workflow(wf))
    assert json.loads(saved.read_text()) == {'user_edited': True}


# ---- MiniMax H3 R2V Swap Low-VRAM (Hearmeman) card -------------------------------------------------------------
SWAP_WF = ROOT/'selfism_workflows/minimax_h3_r2v_swap_lowvram.json'
SWAP_IDS = ['mm-ref2va', 'mm-enc-nvfp4', 'mm-vae-fp16', 'mm-vae-audio', 'r2v-lora-turbo']
SWAP_NODES = ['rgthree-comfy', 'ComfyUI-VideoHelperSuite']
# the R2V Turbo card's workflow (d08fc47) must stay untouched by the swap card
R2V_SHIPPED_SHA256 = '3588c6d1542652be0ae79e103e38409034f096d847fb0e8841677de539071d57'
SWAP_PROMPT_START = 'subject_definitions: <Subject 1> is the original woman in <Video 1>, shown in <Picture 1>.'


def swap_nodes():
    w = json.loads(SWAP_WF.read_bytes())
    return w, {n['id']: n for n in w['nodes']}


def test_swap_catalog_files_nodes_and_r2_layout():
    files = CATALOG['files']
    assert CATALOG['r2v_swap_files'] == SWAP_IDS and CATALOG['r2v_swap_nodes'] == SWAP_NODES
    for k in SWAP_IDS:
        f = files[k]
        assert re.fullmatch(r'[0-9a-f]{64}', f['sha256']) and f['size_bytes'] > 0 and f['auth'] == 'none', k
        assert f['destination'].startswith('models/') and f['url'].startswith('https://huggingface.co/'), k
    # destinations double as the R2 keys (destination minus "models/") mirrored to selfism-models
    assert {k: files[k]['destination'].removeprefix('models/') for k in SWAP_IDS} == {
        'mm-ref2va': 'diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors',
        'mm-enc-nvfp4': 'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
        'mm-vae-fp16': 'vae/minimax_h3_video_vae_fp16.safetensors',
        'mm-vae-audio': 'vae/minimax_h3_audio_vae_fp32.safetensors',
        'r2v-lora-turbo': 'loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors'}
    assert files['mm-enc-nvfp4']['url'] == 'https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'
    names = {n['name']: n for n in CATALOG['nodes']}
    assert set(SWAP_NODES) <= set(names)
    assert names['ComfyUI-VideoHelperSuite']['ref'] == VHS_REF  # VHS_LoadVideo format 'H3'
    # the R2V Turbo card is unchanged
    assert CATALOG['r2v_files'] == R2V_IDS and CATALOG['r2v_nodes'] == R2V_NODES
    assert hashlib.sha256(R2V_WF.read_bytes()).hexdigest() == R2V_SHIPPED_SHA256


def test_swap_install_selects_the_workflow_files_only(monkeypatch):
    r, captured, html = start_install(monkeypatch, {'profile': 'minimax_r2v_swap_lowvram'})
    assert r.status_code == 200
    assert [f['id'] for f in captured['files']] == SWAP_IDS
    assert captured['selfism_profile'] == 'minimax_r2v_swap_lowvram' and captured['selfism_repair'] is False
    assert captured['pip_packages'] == []
    assert {n['name'] for n in captured['custom_nodes']} == set(SWAP_NODES)
    total = sum(CATALOG['files'][k]['size_bytes'] for k in SWAP_IDS)
    scripts = (ROOT/'scripts/selfism-section.html').read_text(encoding='utf-8')
    for page in (html, scripts):
        assert 'data-sf-action="minimax_r2v_swap_lowvram"' in page and 'MiniMax H3 R2V Swap Low-VRAM (Hearmeman)' in page
        assert f'{total/1e9:.1f}'.replace('.', ',') + ' GB' in page
        assert 'data-sf-action="minimax_r2v"' in page  # the R2V Turbo card stays
    assert not any('HM' in CATALOG['files'][k]['destination'] for k in SWAP_IDS)


def test_swap_workflow_graph_links_and_every_model_is_installed():
    w, N = swap_nodes()
    assert 'definitions' not in w and w['last_node_id'] >= max(N)
    types = {n['type'] for n in w['nodes']}
    # no References Manager / prompt display / preview override, no taeh3
    assert not types & {'MiniMaxH3ReferencePack', 'Display Any (rgthree)', 'ModelPreviewOverrideKJ', 'ResizeImageMaskNode'}
    assert 'taeh3' not in SWAP_WF.read_text(encoding='utf-8') and 'hmmotion' not in SWAP_WF.read_text(encoding='utf-8')
    # links are consistent in both directions
    L = {l[0]: l for l in w['links']}
    assert len(L) == len(w['links']) and w['last_link_id'] >= max(L)
    for lid, src, ss, dst, ds, typ in w['links']:
        assert lid in N[src]['outputs'][ss]['links'] and N[dst]['inputs'][ds]['link'] == lid, lid
    for n in w['nodes']:
        for i in n.get('inputs', []):
            assert i['link'] is None or L[i['link']][3] == n['id'], (n['id'], i['name'])
    def src(nid, name):
        i = next(x for x in N[nid]['inputs'] if x['name'] == name)
        l = L[i['link']]
        return N[l[1]], l[2]
    r2v = next(n for n in w['nodes'] if n['type'] == 'MiniMaxH3ReferenceToVideo')
    p1, _ = src(r2v['id'], 'ref_images.ref_image_0'); p2, _ = src(r2v['id'], 'ref_images.ref_image_1')
    v1, vs = src(r2v['id'], 'ref_videos.ref_video_0'); va, vas = src(r2v['id'], 'ref_video_audios.ref_video_audio_0')
    assert (p1['type'], p1['title'], p1['widgets_values'][1]) == ('LoadImage', 'Picture 1 (original woman from video)', 'image')
    assert (p2['type'], p2['title']) == ('LoadImage', 'Picture 2 (Millie / new person)')
    assert v1['type'] == 'VHS_LoadVideo' and v1['title'] == 'Video 1 (source reel)' and vs == 0
    assert va is v1 and vas == 2  # the reel's own soundtrack
    vv = v1['widgets_values']
    assert (vv['custom_width'], vv['custom_height'], vv['force_rate'], vv['format']) == (640, 0, 24, 'H3')
    # duration -> length (17k+5 frames) feeds both the generation length and the reel's frame cap
    math, ms = src(r2v['id'], 'length')
    assert math['type'] == 'ComfyMathExpression' and src(v1['id'], 'frame_load_cap') == (math, ms)
    dur, _ = src(math['id'], 'values.a')
    assert dur['type'] == 'PrimitiveFloat' and dur['widgets_values'] == [5]
    a = 5
    length = eval(math['widgets_values'][0], {'max': max, 'round': round, 'a': a})
    assert length == 124 and length % 17 == 5 and vv['frame_load_cap'] == length
    res, _ = src(r2v['id'], 'width')
    assert res['type'] == 'ResolutionSelector' and res['widgets_values'] == ['9:16 (Portrait Widescreen)', 0.35, 32]
    assert src(r2v['id'], 'height')[0] is res
    prompt, _ = src(r2v['id'], 'prompt')
    assert prompt['type'] == 'PrimitiveStringMultiline' and prompt['widgets_values'][0].startswith(SWAP_PROMPT_START)
    assert prompt['widgets_values'][0].rstrip().endswith('non_diegetic_music: N/A')
    assert r2v['widgets_values'][0] == prompt['widgets_values'][0] and r2v['widgets_values'][4] == 'match'
    # models: pruned UNET (bare name), NVFP4 encoder (type minimax), turbo LoRA 4 steps with the original sampler
    installed = {Path(CATALOG['files'][k]['destination']).name for k in CATALOG['r2v_swap_files']}
    used = {}
    for n in w['nodes']:
        v = n.get('widgets_values')
        if n['type'] in ('UNETLoader', 'CLIPLoader', 'VAELoader', 'LoraLoaderModelOnly'): used[n['id']] = v[0]
        if n['type'] == 'UNETLoader': assert v == ['minimax_h3_ref2va_pruned_int8_convrot.safetensors', 'default']
        if n['type'] == 'CLIPLoader': assert v == ['qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'minimax', 'default']
        if n['type'] == 'LoraLoaderModelOnly': assert v == ['minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors', 0.85]
        if n['type'] == 'BetaSamplingScheduler': assert v[0] == 4
        if n['type'] == 'KSamplerSelect': assert v == ['seeds_2']
        if n['type'] == 'Power Lora Loader (rgthree)':
            assert not [x for x in v if isinstance(x, dict) and x.get('on')]
        for mdl in (n.get('properties') or {}).get('models', []):
            assert mdl['name'] in installed, mdl
    assert set(used.values()) == installed and len(used) == 5, set(used.values()) ^ installed
    cnr = {(n.get('properties') or {}).get('cnr_id') for n in w['nodes']} - {None, 'comfy-core'}
    assert {R2V_CNR_TO_PACK[c] for c in cnr} == set(CATALOG['r2v_swap_nodes'])


def test_swap_workflow_download_and_install_are_byte_identical_and_nondestructive(monkeypatch, tmp_path):
    with TestClient(m.app) as client:
        r = client.get('/api/selfism/workflow/minimax_r2v_swap_lowvram')
    assert r.status_code == 200 and r.content == SWAP_WF.read_bytes()
    assert s.MINIMAX_R2V_SWAP_WORKFLOW_NAME in r.headers['content-disposition']
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
    saved = tmp_path/'user/default/workflows/Selfism'/s.MINIMAX_R2V_SWAP_WORKFLOW_NAME
    wf = {'selfism_profile': 'minimax_r2v_swap_lowvram', 'precision': 'fp8', 'files': [{'id': 'x'}], 'pip_packages': []}
    asyncio.run(ctrl._install_workflow(wf))
    assert saved.read_bytes() == SWAP_WF.read_bytes()
    assert not (tmp_path/'user/default/workflows/Selfism'/s.MINIMAX_R2V_WORKFLOW_NAME).exists()
    saved.write_text('{"user_edited":true}')
    asyncio.run(ctrl._install_workflow(wf))
    assert json.loads(saved.read_text()) == {'user_edited': True}


# ---- MiniMax H3 R2V Swap High-Res (Hearmeman) card -------------------------------------------------------------
HIGHRES_WF = ROOT/'selfism_workflows/minimax_h3_r2v_swap_highres.json'
HIGHRES_IDS = ['mm-ref2va', 'mm-enc-nvfp4', 'mm-vae-fp16', 'mm-vae-audio', 'r2v-lora-turbo']
HIGHRES_NODES = ['rgthree-comfy', 'ComfyUI-VideoHelperSuite', 'ComfyUI-KJNodes']
# Low-VRAM and R2V Turbo workflows must stay byte-identical after the High-Res card lands
SWAP_SHIPPED_SHA256 = '752961044d48a695d4a2dfb218ee45d43a9ef81bdde02428023d67194340b3ad'
HIGHRES_PROMPT_START = 'subject_definitions:\n<Subject 1> is the woman whose appearance comes only from <Picture 2>'


def highres_nodes():
    w = json.loads(HIGHRES_WF.read_bytes())
    return w, {n['id']: n for n in w['nodes']}


def test_highres_catalog_files_nodes_and_unchanged_siblings():
    files = CATALOG['files']
    assert CATALOG['r2v_swap_highres_files'] == HIGHRES_IDS
    assert CATALOG['r2v_swap_highres_nodes'] == HIGHRES_NODES
    assert CATALOG['r2v_swap_highres_files'] == CATALOG['r2v_swap_files']
    for k in HIGHRES_IDS:
        f = files[k]
        assert re.fullmatch(r'[0-9a-f]{64}', f['sha256']) and f['size_bytes'] > 0 and f['auth'] == 'none', k
        assert f['destination'].startswith('models/') and f['url'].startswith('https://huggingface.co/'), k
    names = {n['name']: n for n in CATALOG['nodes']}
    assert set(HIGHRES_NODES) <= set(names)
    assert names['ComfyUI-KJNodes']['ref'] == 'd3cfe21625e5170126ce06fbfcfe1d88108688c3'
    assert names['ComfyUI-VideoHelperSuite']['ref'] == VHS_REF
    # sibling cards unchanged
    assert CATALOG['r2v_swap_files'] == SWAP_IDS and CATALOG['r2v_swap_nodes'] == SWAP_NODES
    assert CATALOG['r2v_files'] == R2V_IDS and CATALOG['r2v_nodes'] == R2V_NODES
    assert hashlib.sha256(R2V_WF.read_bytes()).hexdigest() == R2V_SHIPPED_SHA256
    assert hashlib.sha256(SWAP_WF.read_bytes()).hexdigest() == SWAP_SHIPPED_SHA256


def test_highres_install_selects_the_workflow_files_and_kjnodes(monkeypatch):
    r, captured, html = start_install(monkeypatch, {'profile': 'minimax_r2v_swap_highres'})
    assert r.status_code == 200
    assert [f['id'] for f in captured['files']] == HIGHRES_IDS
    assert captured['selfism_profile'] == 'minimax_r2v_swap_highres' and captured['selfism_repair'] is False
    assert captured['pip_packages'] == []
    assert {n['name'] for n in captured['custom_nodes']} == set(HIGHRES_NODES)
    total = sum(CATALOG['files'][k]['size_bytes'] for k in HIGHRES_IDS)
    scripts = (ROOT/'scripts/selfism-section.html').read_text(encoding='utf-8')
    for page in (html, scripts):
        assert 'data-sf-action="minimax_r2v_swap_highres"' in page
        assert 'MiniMax H3 R2V Swap High-Res (Hearmeman)' in page
        assert f'{total/1e9:.1f}'.replace('.', ',') + ' GB' in page
        assert 'data-sf-action="minimax_r2v_swap_lowvram"' in page
        assert 'data-sf-action="minimax_r2v"' in page


def test_highres_workflow_graph_chunk_lowvram_chain_and_defaults():
    w, N = highres_nodes()
    assert 'definitions' not in w and w['last_node_id'] >= max(N)
    types = {n['type'] for n in w['nodes']}
    assert not types & {'MiniMaxH3ReferencePack', 'Display Any (rgthree)', 'ModelPreviewOverrideKJ', 'ResizeImageMaskNode'}
    assert 'PathchSageAttentionKJ' not in types and 'SageAttention' not in ''.join(types)
    assert 'taeh3' not in HIGHRES_WF.read_text(encoding='utf-8') and 'hmmotion' not in HIGHRES_WF.read_text(encoding='utf-8')
    L = {l[0]: l for l in w['links']}
    assert len(L) == len(w['links']) and w['last_link_id'] >= max(L)
    for lid, src, ss, dst, ds, typ in w['links']:
        assert lid in N[src]['outputs'][ss]['links'] and N[dst]['inputs'][ds]['link'] == lid, lid
    for n in w['nodes']:
        for i in n.get('inputs', []):
            assert i['link'] is None or L[i['link']][3] == n['id'], (n['id'], i['name'])

    def src(nid, name):
        i = next(x for x in N[nid]['inputs'] if x['name'] == name)
        l = L[i['link']]
        return N[l[1]], l[2]

    chunk = next(n for n in w['nodes'] if n['type'] == 'MiniMaxChunkFeedForward')
    low = next(n for n in w['nodes'] if n['type'] == 'MiniMaxLowVRAMAttention')
    assert chunk['widgets_values'] == [4, 2048] and chunk['title'] == 'Chunk FeedForward'
    assert low['widgets_values'] == [4] and low['title'] == 'Low VRAM Attention'
    assert (chunk.get('properties') or {}).get('cnr_id') == 'comfyui-kjnodes'
    assert (low.get('properties') or {}).get('cnr_id') == 'comfyui-kjnodes'
    turbo, _ = src(chunk['id'], 'model')
    assert turbo['type'] == 'LoraLoaderModelOnly'
    mid, _ = src(low['id'], 'model')
    assert mid is chunk
    guider = next(n for n in w['nodes'] if n['type'] == 'BasicGuider')
    sched = next(n for n in w['nodes'] if n['type'] == 'BetaSamplingScheduler')
    assert src(guider['id'], 'model')[0] is low
    assert src(sched['id'], 'model')[0] is low

    r2v = next(n for n in w['nodes'] if n['type'] == 'MiniMaxH3ReferenceToVideo')
    p1, _ = src(r2v['id'], 'ref_images.ref_image_0'); p2, _ = src(r2v['id'], 'ref_images.ref_image_1')
    v1, vs = src(r2v['id'], 'ref_videos.ref_video_0'); va, vas = src(r2v['id'], 'ref_video_audios.ref_video_audio_0')
    assert (p1['type'], p1['title']) == ('LoadImage', 'Picture 1 (original woman from video)')
    assert (p2['type'], p2['title']) == ('LoadImage', 'Picture 2 (Millie / new person)')
    assert v1['type'] == 'VHS_LoadVideo' and v1['title'] == 'Video 1 (source reel)' and vs == 0
    assert va is v1 and vas == 2
    vv = v1['widgets_values']
    assert (vv['custom_width'], vv['custom_height'], vv['force_rate'], vv['format']) == (512, 0, 24, 'H3')
    math, ms = src(r2v['id'], 'length')
    assert math['type'] == 'ComfyMathExpression' and src(v1['id'], 'frame_load_cap') == (math, ms)
    dur, _ = src(math['id'], 'values.a')
    assert dur['type'] == 'PrimitiveFloat' and dur['widgets_values'] == [5]
    res, _ = src(r2v['id'], 'width')
    assert res['type'] == 'ResolutionSelector' and res['widgets_values'] == ['9:16 (Portrait Widescreen)', 0.98, 32]
    assert src(r2v['id'], 'height')[0] is res
    prompt, _ = src(r2v['id'], 'prompt')
    assert prompt['type'] == 'PrimitiveStringMultiline'
    assert prompt['widgets_values'][0].startswith(HIGHRES_PROMPT_START)
    assert prompt['widgets_values'][0].rstrip().endswith('non_diegetic_music:\nN/A')
    assert r2v['widgets_values'][0] == prompt['widgets_values'][0] and r2v['widgets_values'][4] == 'max'
    assert r2v['widgets_values'][1:4] == [768, 1344, 124]
    combine = next(n for n in w['nodes'] if n['type'] == 'VHS_VideoCombine')
    assert combine['widgets_values']['filename_prefix'] == 'MiniMax_R2V_Swap_HighRes'

    installed = {Path(CATALOG['files'][k]['destination']).name for k in CATALOG['r2v_swap_highres_files']}
    used = {}
    for n in w['nodes']:
        v = n.get('widgets_values')
        if n['type'] in ('UNETLoader', 'CLIPLoader', 'VAELoader', 'LoraLoaderModelOnly'): used[n['id']] = v[0]
        if n['type'] == 'UNETLoader': assert v == ['minimax_h3_ref2va_pruned_int8_convrot.safetensors', 'default']
        if n['type'] == 'CLIPLoader': assert v == ['qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'minimax', 'default']
        if n['type'] == 'LoraLoaderModelOnly': assert v == ['minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors', 0.85]
        if n['type'] == 'BetaSamplingScheduler': assert v[0] == 4
        if n['type'] == 'Power Lora Loader (rgthree)':
            assert not [x for x in v if isinstance(x, dict) and x.get('on')]
        for mdl in (n.get('properties') or {}).get('models', []):
            assert mdl['name'] in installed, mdl
    assert set(used.values()) == installed and len(used) == 5, set(used.values()) ^ installed
    cnr = {(n.get('properties') or {}).get('cnr_id') for n in w['nodes']} - {None, 'comfy-core'}
    assert {R2V_CNR_TO_PACK[c] for c in cnr} == set(CATALOG['r2v_swap_highres_nodes'])
    assert cnr <= set(R2V_CNR_TO_PACK)


def test_highres_workflow_download_and_install_are_byte_identical_and_nondestructive(monkeypatch, tmp_path):
    with TestClient(m.app) as client:
        r = client.get('/api/selfism/workflow/minimax_r2v_swap_highres')
    assert r.status_code == 200 and r.content == HIGHRES_WF.read_bytes()
    assert s.MINIMAX_R2V_SWAP_HIGHRES_WORKFLOW_NAME in r.headers['content-disposition']
    monkeypatch.setattr(m, 'COMFYUI_DIR', tmp_path)
    monkeypatch.setattr(m, 'COMFYUI_VENV', tmp_path/'.venv')
    python = m.COMFYUI_VENV/'bin/python'
    python.parent.mkdir(parents=True); python.touch()
    # workspace args file: install should append --reserve-vram 8
    work = tmp_path/'runpod-slim'
    work.mkdir()
    args = work/'comfyui_args.txt'
    args.write_text('# keep\n--preview-method none\n', encoding='utf-8')
    monkeypatch.setenv('SELFISM_WORKSPACE', str(work))
    ctrl = type(m.selfism_controller)()
    async def ready(*args): pass
    async def run(*args, **kwargs): return 0, 'torch==2.8.0'
    async def sources(host, files): return files, []
    monkeypatch.setattr(ctrl, '_wait_for_comfyui', ready)
    monkeypatch.setattr(ctrl, '_run_process', run)
    monkeypatch.setattr(m.JobController, '_install_workflow', ready)
    monkeypatch.setattr(s, 'resolve_sources', sources)
    saved = tmp_path/'user/default/workflows/Selfism'/s.MINIMAX_R2V_SWAP_HIGHRES_WORKFLOW_NAME
    wf = {'selfism_profile': 'minimax_r2v_swap_highres', 'precision': 'fp8', 'files': [{'id': 'x'}], 'pip_packages': []}
    asyncio.run(ctrl._install_workflow(wf))
    assert saved.read_bytes() == HIGHRES_WF.read_bytes()
    assert '--reserve-vram 8' in args.read_text(encoding='utf-8')
    assert '--preview-method none' in args.read_text(encoding='utf-8')
    assert not (tmp_path/'user/default/workflows/Selfism'/s.MINIMAX_R2V_SWAP_WORKFLOW_NAME).exists()
    assert not (tmp_path/'user/default/workflows/Selfism'/s.MINIMAX_R2V_WORKFLOW_NAME).exists()
    saved.write_text('{"user_edited":true}')
    before = args.read_text(encoding='utf-8')
    asyncio.run(ctrl._install_workflow(wf))
    assert json.loads(saved.read_text()) == {'user_edited': True}
    assert args.read_text(encoding='utf-8') == before  # do not duplicate --reserve-vram 8


# ---- MiniMax H3 Reel Recreation v2 card ------------------------------------------------------------------------
REEL_WF = ROOT/'selfism_workflows/minimax_h3_reel_recreation_v2.json'
REEL_IDS = ['mm-ref2va', 'r2v-enc', 'mm-vae-fp16', 'mm-vae-audio', 'r2v-lora-turbo-8step']
REEL_NODES = ['ComfyUI-VideoHelperSuite']
# sibling R2V / Low-VRAM / High-Res workflows must stay byte-identical
HIGHRES_SHIPPED_SHA256 = 'e310f6d61e5e3fc064b20106f77503d121506e99009b309e5d4019e0f5e15ff8'
REEL_FIXED_START = 'subject_definitions: <Subject 1> is the original main person in <Video 1>'


def reel_nodes():
    w = json.loads(REEL_WF.read_bytes())
    return w, {n['id']: n for n in w['nodes']}


def test_reel_catalog_files_nodes_and_unchanged_siblings():
    files = CATALOG['files']
    assert CATALOG['reel_recreation_v2_files'] == REEL_IDS
    assert CATALOG['reel_recreation_v2_nodes'] == REEL_NODES
    for k in REEL_IDS:
        f = files[k]
        assert re.fullmatch(r'[0-9a-f]{64}', f['sha256']) and f['size_bytes'] > 0 and f['auth'] == 'none', k
        assert f['destination'].startswith('models/') and f['url'].startswith('https://huggingface.co/'), k
    assert {k: files[k]['destination'].removeprefix('models/') for k in REEL_IDS} == {
        'mm-ref2va': 'diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors',
        'r2v-enc': 'text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
        'mm-vae-fp16': 'vae/minimax_h3_video_vae_fp16.safetensors',
        'mm-vae-audio': 'vae/minimax_h3_audio_vae_fp32.safetensors',
        'r2v-lora-turbo-8step': 'loras/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors'}
    assert files['r2v-lora-turbo-8step']['url'] == (
        'https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/'
        'minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors')
    names = {n['name']: n for n in CATALOG['nodes']}
    assert set(REEL_NODES) <= set(names)
    assert names['ComfyUI-VideoHelperSuite']['ref'] == VHS_REF
    # sibling cards unchanged
    assert CATALOG['r2v_swap_files'] == SWAP_IDS and CATALOG['r2v_swap_nodes'] == SWAP_NODES
    assert CATALOG['r2v_swap_highres_files'] == HIGHRES_IDS and CATALOG['r2v_swap_highres_nodes'] == HIGHRES_NODES
    assert CATALOG['r2v_files'] == R2V_IDS and CATALOG['r2v_nodes'] == R2V_NODES
    assert hashlib.sha256(R2V_WF.read_bytes()).hexdigest() == R2V_SHIPPED_SHA256
    assert hashlib.sha256(SWAP_WF.read_bytes()).hexdigest() == SWAP_SHIPPED_SHA256
    assert hashlib.sha256(HIGHRES_WF.read_bytes()).hexdigest() == HIGHRES_SHIPPED_SHA256


def test_reel_install_selects_the_workflow_files_only(monkeypatch):
    r, captured, html = start_install(monkeypatch, {'profile': 'minimax_reel_recreation_v2'})
    assert r.status_code == 200
    assert [f['id'] for f in captured['files']] == REEL_IDS
    assert captured['selfism_profile'] == 'minimax_reel_recreation_v2' and captured['selfism_repair'] is False
    assert captured['pip_packages'] == []
    assert {n['name'] for n in captured['custom_nodes']} == set(REEL_NODES)
    total = sum(CATALOG['files'][k]['size_bytes'] for k in REEL_IDS)
    scripts = (ROOT/'scripts/selfism-section.html').read_text(encoding='utf-8')
    for page in (html, scripts):
        assert 'data-sf-action="minimax_reel_recreation_v2"' in page
        assert 'MiniMax H3 Reel Recreation v2' in page
        assert f'{total/1e9:.1f}'.replace('.', ',') + ' GB' in page
        assert 'data-sf-action="minimax_r2v_swap_lowvram"' in page
        assert 'data-sf-action="minimax_r2v_swap_highres"' in page
        assert 'data-sf-action="minimax_r2v"' in page


def test_reel_workflow_graph_links_models_prompt_and_original_audio():
    w, N = reel_nodes()
    assert w['id'] == '13d16ef8-5c1d-4e99-9d66-bdea8d294cf3'
    assert 'definitions' not in w and w['last_node_id'] >= max(N)
    types = {n['type'] for n in w['nodes']}
    assert not types & {'MiniMaxH3ReferencePack', 'Display Any (rgthree)', 'ModelPreviewOverrideKJ',
                        'ResolutionSelector', 'Power Lora Loader (rgthree)', 'VAEDecodeAudio',
                        'BetaSamplingScheduler', 'ExtendIntermediateSigmas'}
    assert 'OpenRouter' not in REEL_WF.read_text(encoding='utf-8')
    assert 'ReferencePack' not in REEL_WF.read_text(encoding='utf-8')
    L = {l[0]: l for l in w['links']}
    assert len(L) == len(w['links']) and w['last_link_id'] >= max(L)
    for lid, src, ss, dst, ds, typ in w['links']:
        assert lid in N[src]['outputs'][ss]['links'] and N[dst]['inputs'][ds]['link'] == lid, lid
    for n in w['nodes']:
        for i in n.get('inputs', []):
            assert i['link'] is None or L[i['link']][3] == n['id'], (n['id'], i['name'])

    def src(nid, name):
        i = next(x for x in N[nid]['inputs'] if x['name'] == name)
        l = L[i['link']]
        return N[l[1]], l[2]

    r2v = next(n for n in w['nodes'] if n['type'] == 'MiniMaxH3ReferenceToVideo')
    p1, _ = src(r2v['id'], 'ref_images.ref_image_0'); p2, _ = src(r2v['id'], 'ref_images.ref_image_1')
    v1, vs = src(r2v['id'], 'ref_videos.ref_video_0'); va, vas = src(r2v['id'], 'ref_video_audios.ref_video_audio_0')
    assert (p1['type'], p1['title']) == ('LoadImage', 'Picture 1 (persona front)')
    assert (p2['type'], p2['title']) == ('LoadImage', 'Picture 2 (persona 3/4)')
    assert v1['type'] == 'VHS_LoadVideo' and v1['title'] == 'Video 1 (source reel)' and vs == 0
    assert va is v1 and vas == 2
    vv = v1['widgets_values']
    assert (vv['custom_width'], vv['custom_height'], vv['force_rate'], vv['format']) == (640, 0, 24, 'H3')
    math, ms = src(r2v['id'], 'length')
    assert math['type'] == 'ComfyMathExpression' and src(v1['id'], 'frame_load_cap') == (math, ms)
    dur, _ = src(math['id'], 'values.a')
    assert dur['type'] == 'PrimitiveFloat' and dur['widgets_values'] == [5]
    a = 5
    length = eval(math['widgets_values'][0], {'max': max, 'round': round, 'a': a})
    assert length == 124 and length % 17 == 5 and vv['frame_load_cap'] == length
    # size widgets on R2V (no ResolutionSelector)
    assert r2v['widgets_values'][1:5] == [768, 1344, 124, 'match']
    assert next(x for x in r2v['inputs'] if x['name'] == 'width')['link'] is None
    assert next(x for x in r2v['inputs'] if x['name'] == 'height')['link'] is None
    # fixed roles + scene concatenated into prompt
    concat, _ = src(r2v['id'], 'prompt')
    assert concat['type'] == 'StringConcatenate'
    fixed, _ = src(concat['id'], 'string_a'); scene, _ = src(concat['id'], 'string_b')
    assert fixed['type'] == 'PrimitiveStringMultiline' and scene['type'] == 'PrimitiveStringMultiline'
    assert fixed['widgets_values'][0].startswith(REEL_FIXED_START)
    assert 'audio_roles:' in fixed['widgets_values'][0]
    assert scene['widgets_values'][0].lstrip().startswith('detailed_description:')
    assert concat['widgets_values'][2] == '\n\n'
    # original source audio also feeds VideoCombine (no VAEDecodeAudio)
    combine = next(n for n in w['nodes'] if n['type'] == 'VHS_VideoCombine')
    ca, cas = src(combine['id'], 'audio')
    assert ca is v1 and cas == 2
    assert combine['widgets_values']['filename_prefix'] == 'MiniMax_H3/Reel_Recreation_v2/source_audio'
    # models + sampler
    installed = {Path(CATALOG['files'][k]['destination']).name for k in CATALOG['reel_recreation_v2_files']}
    used = {}
    for n in w['nodes']:
        v = n.get('widgets_values')
        if n['type'] in ('UNETLoader', 'CLIPLoader', 'VAELoader', 'LoraLoaderModelOnly'): used[n['id']] = v[0]
        if n['type'] == 'UNETLoader': assert v == ['minimax_h3_ref2va_pruned_int8_convrot.safetensors', 'default']
        if n['type'] == 'CLIPLoader': assert v == ['qwen3vl_32b_minimax_h3_int8_convrot.safetensors', 'minimax', 'default']
        if n['type'] == 'LoraLoaderModelOnly':
            assert v == ['minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors', 0.85]
        if n['type'] == 'BasicScheduler': assert v == ['simple', 8, 1]
        if n['type'] == 'KSamplerSelect': assert v == ['euler']
        if n['type'] == 'RandomNoise': assert v == [703673408085144, 'fixed']
        for mdl in (n.get('properties') or {}).get('models', []):
            assert mdl['name'] in installed, mdl
    assert set(used.values()) == installed and len(used) == 5, set(used.values()) ^ installed
    cnr = {(n.get('properties') or {}).get('cnr_id') for n in w['nodes']} - {None, 'comfy-core'}
    assert cnr == {'comfyui-videohelpersuite'}
    assert {R2V_CNR_TO_PACK[c] for c in cnr} == set(CATALOG['reel_recreation_v2_nodes'])


def test_reel_workflow_download_and_install_are_byte_identical_and_nondestructive(monkeypatch, tmp_path):
    with TestClient(m.app) as client:
        r = client.get('/api/selfism/workflow/minimax_reel_recreation_v2')
    assert r.status_code == 200 and r.content == REEL_WF.read_bytes()
    assert s.MINIMAX_REEL_RECREATION_V2_WORKFLOW_NAME in r.headers['content-disposition']
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
    saved = tmp_path/'user/default/workflows/Selfism'/s.MINIMAX_REEL_RECREATION_V2_WORKFLOW_NAME
    wf = {'selfism_profile': 'minimax_reel_recreation_v2', 'precision': 'fp8', 'files': [{'id': 'x'}], 'pip_packages': []}
    asyncio.run(ctrl._install_workflow(wf))
    assert saved.read_bytes() == REEL_WF.read_bytes()
    assert not (tmp_path/'user/default/workflows/Selfism'/s.MINIMAX_R2V_WORKFLOW_NAME).exists()
    assert not (tmp_path/'user/default/workflows/Selfism'/s.MINIMAX_R2V_SWAP_WORKFLOW_NAME).exists()
    assert not (tmp_path/'user/default/workflows/Selfism'/s.MINIMAX_R2V_SWAP_HIGHRES_WORKFLOW_NAME).exists()
    assert not (tmp_path/'user/default/workflows/Selfism'/s.MINIMAX_REEL_RECREATION_V3_WORKFLOW_NAME).exists()
    saved.write_text('{"user_edited":true}')
    asyncio.run(ctrl._install_workflow(wf))
    assert json.loads(saved.read_text()) == {'user_edited': True}


# ---- MiniMax H3 Reel Recreation v3 card ------------------------------------------------------------------------
REEL_V3_WF = ROOT/'selfism_workflows/minimax_h3_reel_recreation_v3.json'
REEL_V3_IDS = ['mm-ref2va', 'r2v-enc', 'mm-vae-fp16', 'mm-vae-audio']
REEL_V3_NODES = ['ComfyUI-VideoHelperSuite']
REEL_V2_SHIPPED_SHA256 = 'd1c571648070f2117b0fc48cd5aec64b9d93b1ab803b1a4d9f5124f821fa067e'
REEL_V3_FIXED_START = 'subject_definitions: <Subject 1> is the original main person to be replaced in <Video 1>'


def reel_v3_nodes():
    w = json.loads(REEL_V3_WF.read_bytes())
    return w, {n['id']: n for n in w['nodes']}


def test_reel_v3_catalog_files_nodes_and_unchanged_siblings():
    files = CATALOG['files']
    assert CATALOG['reel_recreation_v3_files'] == REEL_V3_IDS
    assert CATALOG['reel_recreation_v3_nodes'] == REEL_V3_NODES
    for k in REEL_V3_IDS:
        f = files[k]
        assert re.fullmatch(r'[0-9a-f]{64}', f['sha256']) and f['size_bytes'] > 0 and f['auth'] == 'none', k
        assert f['destination'].startswith('models/') and f['url'].startswith('https://huggingface.co/'), k
    assert {k: files[k]['destination'].removeprefix('models/') for k in REEL_V3_IDS} == {
        'mm-ref2va': 'diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors',
        'r2v-enc': 'text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
        'mm-vae-fp16': 'vae/minimax_h3_video_vae_fp16.safetensors',
        'mm-vae-audio': 'vae/minimax_h3_audio_vae_fp32.safetensors'}
    assert 'r2v-lora-turbo-8step' not in CATALOG['reel_recreation_v3_files']
    names = {n['name']: n for n in CATALOG['nodes']}
    assert set(REEL_V3_NODES) <= set(names)
    assert names['ComfyUI-VideoHelperSuite']['ref'] == VHS_REF
    # sibling cards unchanged (including Reel Recreation v2)
    assert CATALOG['reel_recreation_v2_files'] == REEL_IDS and CATALOG['reel_recreation_v2_nodes'] == REEL_NODES
    assert CATALOG['r2v_swap_files'] == SWAP_IDS and CATALOG['r2v_swap_nodes'] == SWAP_NODES
    assert CATALOG['r2v_swap_highres_files'] == HIGHRES_IDS and CATALOG['r2v_swap_highres_nodes'] == HIGHRES_NODES
    assert CATALOG['r2v_files'] == R2V_IDS and CATALOG['r2v_nodes'] == R2V_NODES
    assert hashlib.sha256(R2V_WF.read_bytes()).hexdigest() == R2V_SHIPPED_SHA256
    assert hashlib.sha256(SWAP_WF.read_bytes()).hexdigest() == SWAP_SHIPPED_SHA256
    assert hashlib.sha256(HIGHRES_WF.read_bytes()).hexdigest() == HIGHRES_SHIPPED_SHA256
    assert hashlib.sha256(REEL_WF.read_bytes()).hexdigest() == REEL_V2_SHIPPED_SHA256


def test_reel_v3_install_selects_the_workflow_files_only(monkeypatch):
    r, captured, html = start_install(monkeypatch, {'profile': 'minimax_reel_recreation_v3'})
    assert r.status_code == 200
    assert [f['id'] for f in captured['files']] == REEL_V3_IDS
    assert captured['selfism_profile'] == 'minimax_reel_recreation_v3' and captured['selfism_repair'] is False
    assert captured['pip_packages'] == []
    assert {n['name'] for n in captured['custom_nodes']} == set(REEL_V3_NODES)
    total = sum(CATALOG['files'][k]['size_bytes'] for k in REEL_V3_IDS)
    scripts = (ROOT/'scripts/selfism-section.html').read_text(encoding='utf-8')
    for page in (html, scripts):
        assert 'data-sf-action="minimax_reel_recreation_v3"' in page
        assert 'MiniMax H3 Reel Recreation v3' in page
        assert f'{total/1e9:.1f}'.replace('.', ',') + ' GB' in page
        assert 'data-sf-action="minimax_reel_recreation_v2"' in page
        assert 'data-sf-action="minimax_r2v_swap_lowvram"' in page
        assert 'data-sf-action="minimax_r2v_swap_highres"' in page
        assert 'data-sf-action="minimax_r2v"' in page


def test_reel_v3_workflow_graph_first_frame_guide_no_turbo_and_original_audio():
    w, N = reel_v3_nodes()
    assert w['id'] == '243e1b66-47bd-4749-9fd5-4806b0bfbd61'
    assert 'definitions' not in w and w['last_node_id'] >= max(N)
    types = {n['type'] for n in w['nodes']}
    assert 'MiniMaxH3AddGuide' in types
    assert not types & {'MiniMaxH3ReferencePack', 'Display Any (rgthree)', 'ModelPreviewOverrideKJ',
                        'ResolutionSelector', 'Power Lora Loader (rgthree)', 'VAEDecodeAudio',
                        'BetaSamplingScheduler', 'ExtendIntermediateSigmas', 'LoraLoaderModelOnly'}
    text = REEL_V3_WF.read_text(encoding='utf-8')
    assert 'OpenRouter' not in text and 'ReferencePack' not in text
    assert 'turbo' not in text.lower() or 'no Turbo LoRA' in text  # titles may mention absence
    assert 'minimax_h3_ref2v_turbo' not in text
    L = {l[0]: l for l in w['links']}
    assert len(L) == len(w['links']) and w['last_link_id'] >= max(L)
    for lid, src, ss, dst, ds, typ in w['links']:
        assert lid in N[src]['outputs'][ss]['links'] and N[dst]['inputs'][ds]['link'] == lid, lid
    for n in w['nodes']:
        for i in n.get('inputs', []):
            assert i['link'] is None or L[i['link']][3] == n['id'], (n['id'], i['name'])

    def src(nid, name):
        i = next(x for x in N[nid]['inputs'] if x['name'] == name)
        l = L[i['link']]
        return N[l[1]], l[2]

    r2v = next(n for n in w['nodes'] if n['type'] == 'MiniMaxH3ReferenceToVideo')
    guide = next(n for n in w['nodes'] if n['type'] == 'MiniMaxH3AddGuide')
    assert guide['widgets_values'] == [0]
    p1, _ = src(r2v['id'], 'ref_images.ref_image_0'); p2, _ = src(r2v['id'], 'ref_images.ref_image_1')
    v1, vs = src(r2v['id'], 'ref_videos.ref_video_0'); va, vas = src(r2v['id'], 'ref_video_audios.ref_video_audio_0')
    assert (p1['type'], p1['title']) == ('LoadImage', '01 - RECREATED FIRST FRAME / your persona IN the scene')
    assert (p2['type'], p2['title']) == ('LoadImage', '02 - PERSONA FACE / same identity as Picture 1')
    assert v1['type'] == 'VHS_LoadVideo' and 'SOURCE REEL' in v1['title'] and vs == 0
    assert va is v1 and vas == 2
    # Picture 1 also guides frame 0
    g_img, _ = src(guide['id'], 'image')
    g_pos, _ = src(guide['id'], 'positive')
    g_lat, _ = src(guide['id'], 'latent')
    g_vae, _ = src(guide['id'], 'vae')
    assert g_img is p1 and g_pos is r2v and g_lat is r2v
    assert g_vae['type'] == 'VAELoader' and g_vae['widgets_values'] == ['minimax_h3_video_vae_fp16.safetensors']
    guider = next(n for n in w['nodes'] if n['type'] == 'BasicGuider')
    g_cond, _ = src(guider['id'], 'conditioning')
    assert g_cond is guide
    vv = v1['widgets_values']
    assert (vv['custom_width'], vv['custom_height'], vv['force_rate'], vv['format']) == (640, 0, 24, 'H3')
    math, ms = src(r2v['id'], 'length')
    assert math['type'] == 'ComfyMathExpression' and src(v1['id'], 'frame_load_cap') == (math, ms)
    dur, _ = src(math['id'], 'values.a')
    assert dur['type'] == 'PrimitiveFloat' and dur['widgets_values'] == [5]
    a = 5
    length = eval(math['widgets_values'][0], {'max': max, 'round': round, 'a': a})
    assert length == 124 and length % 17 == 5 and vv['frame_load_cap'] == length
    assert r2v['widgets_values'][1:5] == [768, 1344, 124, 'match']
    # width/height are widget-only on this graph (no linked ResolutionSelector)
    assert not any(x['name'] in ('width', 'height') and x.get('link') is not None for x in r2v['inputs'])
    concat, _ = src(r2v['id'], 'prompt')
    assert concat['type'] == 'StringConcatenate'
    fixed, _ = src(concat['id'], 'string_a'); scene, _ = src(concat['id'], 'string_b')
    assert fixed['type'] == 'PrimitiveStringMultiline' and scene['type'] == 'PrimitiveStringMultiline'
    assert fixed['widgets_values'][0].startswith(REEL_V3_FIXED_START)
    assert 'audio_roles:' in fixed['widgets_values'][0]
    assert scene['widgets_values'][0].lstrip().startswith('detailed_description:')
    assert concat['widgets_values'][2] == '\n\n'
    combine = next(n for n in w['nodes'] if n['type'] == 'VHS_VideoCombine')
    ca, cas = src(combine['id'], 'audio')
    assert ca is v1 and cas == 2
    assert combine['widgets_values']['filename_prefix'] == 'MiniMax_H3/Reel_Recreation_v3/first_frame_original_audio'
    installed = {Path(CATALOG['files'][k]['destination']).name for k in CATALOG['reel_recreation_v3_files']}
    used = {}
    for n in w['nodes']:
        v = n.get('widgets_values')
        if n['type'] in ('UNETLoader', 'CLIPLoader', 'VAELoader'): used[n['id']] = v[0]
        if n['type'] == 'UNETLoader': assert v == ['minimax_h3_ref2va_pruned_int8_convrot.safetensors', 'default']
        if n['type'] == 'CLIPLoader': assert v == ['qwen3vl_32b_minimax_h3_int8_convrot.safetensors', 'minimax', 'default']
        if n['type'] == 'BasicScheduler': assert v == ['simple', 20, 1]
        if n['type'] == 'KSamplerSelect': assert v == ['res_multistep']
        if n['type'] == 'RandomNoise': assert v == [703673408085144, 'fixed']
        for mdl in (n.get('properties') or {}).get('models', []):
            assert mdl['name'] in installed, mdl
    assert set(used.values()) == installed and len(used) == 4, set(used.values()) ^ installed
    # base model feeds scheduler + guider directly (no LoRA)
    sched = next(n for n in w['nodes'] if n['type'] == 'BasicScheduler')
    unet = next(n for n in w['nodes'] if n['type'] == 'UNETLoader')
    assert src(sched['id'], 'model') == (unet, 0)
    assert src(guider['id'], 'model') == (unet, 0)
    cnr = {(n.get('properties') or {}).get('cnr_id') for n in w['nodes']} - {None, 'comfy-core'}
    assert cnr == {'comfyui-videohelpersuite'}
    assert {R2V_CNR_TO_PACK[c] for c in cnr} == set(CATALOG['reel_recreation_v3_nodes'])


def test_reel_v3_workflow_download_and_install_are_byte_identical_and_nondestructive(monkeypatch, tmp_path):
    with TestClient(m.app) as client:
        r = client.get('/api/selfism/workflow/minimax_reel_recreation_v3')
    assert r.status_code == 200 and r.content == REEL_V3_WF.read_bytes()
    assert s.MINIMAX_REEL_RECREATION_V3_WORKFLOW_NAME in r.headers['content-disposition']
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
    saved = tmp_path/'user/default/workflows/Selfism'/s.MINIMAX_REEL_RECREATION_V3_WORKFLOW_NAME
    wf = {'selfism_profile': 'minimax_reel_recreation_v3', 'precision': 'fp8', 'files': [{'id': 'x'}], 'pip_packages': []}
    asyncio.run(ctrl._install_workflow(wf))
    assert saved.read_bytes() == REEL_V3_WF.read_bytes()
    assert not (tmp_path/'user/default/workflows/Selfism'/s.MINIMAX_REEL_RECREATION_V2_WORKFLOW_NAME).exists()
    assert not (tmp_path/'user/default/workflows/Selfism'/s.MINIMAX_R2V_WORKFLOW_NAME).exists()
    assert not (tmp_path/'user/default/workflows/Selfism'/s.MINIMAX_R2V_SWAP_WORKFLOW_NAME).exists()
    assert not (tmp_path/'user/default/workflows/Selfism'/s.MINIMAX_R2V_SWAP_HIGHRES_WORKFLOW_NAME).exists()
    saved.write_text('{"user_edited":true}')
    asyncio.run(ctrl._install_workflow(wf))
    assert json.loads(saved.read_text()) == {'user_edited': True}

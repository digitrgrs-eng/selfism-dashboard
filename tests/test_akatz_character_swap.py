"""akatz character swap card: official H3 Ref2VA template + akatz-ai Character Swap LoRA v1 (core nodes only)."""
from __future__ import annotations
import asyncio
import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from tests.test_minimax import start_install

m = importlib.import_module('launcher.app')
s = importlib.import_module('launcher.selfism')
ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / 'catalog/selfism.json').read_text(encoding='utf-8'))
WF = ROOT / 'selfism_workflows/akatz_character_swap.json'
IDS = ['mm-ref2va', 'mm-lora-charswap', 'r2v-lora-turbo', 'mm-enc-nvfp4', 'mm-vae-int8', 'mm-vae-audio']
# Every node type is built into ComfyUI (>= 0.37.0 per the author; image bakes v0.38.1).
CORE_TYPES = {'SaveVideo', 'ResolutionSelector', 'MarkdownNote', 'VAELoader', 'VAEDecodeAudio', 'VAEDecode',
              'KSamplerSelect', 'BasicScheduler', 'SamplerCustomAdvanced', 'BasicGuider', 'UNETLoader',
              'CLIPLoader', 'RandomNoise', 'CreateVideo', 'ComfyMathExpression', 'PrimitiveFloat',
              'MiniMaxH3ReferenceToVideo', 'LoadImage', 'PrimitiveStringMultiline', 'ComfySwitchNode',
              'PrimitiveInt', 'LoraLoaderModelOnly', 'PrimitiveBoolean', 'LoadVideo', 'GetVideoComponents'}


def _wf():
    return json.loads(WF.read_text(encoding='utf-8'))


def test_catalog_lists():
    assert CATALOG['akatz_character_swap_files'] == IDS and CATALOG['akatz_character_swap_nodes'] == []
    assert all(k in CATALOG['files'] for k in IDS)
    assert len({CATALOG['files'][k]['destination'] for k in IDS}) == len(IDS)


def test_install_selects_files_and_card_is_on_both_pages(monkeypatch):
    r, captured, html = start_install(monkeypatch, {'profile': 'akatz_character_swap'})
    assert r.status_code == 200
    assert [f['id'] for f in captured['files']] == IDS
    assert captured['selfism_profile'] == 'akatz_character_swap' and captured['selfism_repair'] is False
    assert captured['pip_packages'] == [] and captured['model_links'] == []
    assert captured['custom_nodes'] == []
    total = sum(CATALOG['files'][k]['size_bytes'] for k in IDS)
    scripts = (ROOT / 'scripts/selfism-section.html').read_text(encoding='utf-8')
    for page in (html, scripts):
        assert 'data-sf-action="akatz_character_swap"' in page and '<h2>akatz character swap</h2>' in page
        assert f'{total/1e9:.1f}'.replace('.', ',') + ' GB' in page
        assert 'data-sf-action="minimax_swap_3"' in page and 'data-sf-action="god_mode"' in page


def test_workflow_core_nodes_models_and_author_settings():
    w = _wf()
    assert {n['type'] for n in w['nodes']} <= CORE_TYPES
    installed = {Path(CATALOG['files'][k]['destination']).name for k in IDS}
    used = {n['widgets_values'][0] for n in w['nodes']
            if n['type'] in ('UNETLoader', 'CLIPLoader', 'VAELoader', 'LoraLoaderModelOnly')}
    assert used == installed
    nodes = {n['id']: n for n in w['nodes']}
    assert nodes[147]['widgets_values'] == ['h3_character_swap_pro4500_1000.safetensors', 1] and nodes[147]['mode'] == 0
    assert nodes[145]['mode'] == 4 and nodes[146]['widgets_values'] == [False]  # turbo bypassed and off
    assert nodes[123]['widgets_values'] == ['res_multistep'] and nodes[143]['widgets_values'][0] == 20
    assert nodes[115]['widgets_values'][0] == '9:16 (Portrait Widescreen)'
    assert '<Video 1>' in nodes[138]['widgets_values'][0] and '<Picture 1>' in nodes[138]['widgets_values'][0]


def test_original_audio_switch_and_link_integrity():
    w = _wf()
    nodes = {n['id']: n for n in w['nodes']}
    links = {l[0]: l for l in w['links']}
    assert w['last_node_id'] >= max(nodes) and w['last_link_id'] >= max(links)
    for lid, a, ao, b, bi, _ in links.values():
        assert lid in nodes[a]['outputs'][ao]['links'] and nodes[b]['inputs'][bi]['link'] == lid
    for n in nodes.values():
        for o in n.get('outputs', []):
            assert all(l in links for l in o.get('links') or [])
        for i in n.get('inputs', []):
            assert i.get('link') is None or i['link'] in links
    sw = nodes[150]
    assert sw['type'] == 'ComfySwitchNode' and sw['widgets_values'] == [True]
    by_name = {i['name']: links[i['link']] for i in sw['inputs'] if i['link']}
    assert by_name['on_true'][1:3] == [148, 1] and nodes[148]['type'] == 'GetVideoComponents'
    assert by_name['on_false'][1:3] == [121, 0] and nodes[121]['type'] == 'VAEDecodeAudio'
    audio_in = nodes[130]['inputs'][1]
    assert nodes[130]['type'] == 'CreateVideo' and links[audio_in['link']][1] == 150
    note = next(n for n in w['nodes'] if n['type'] == 'MarkdownNote' and n['title'].startswith('UPUTSTVO'))
    text = note['widgets_values'][0]
    assert 'Swap the woman in <Video 1> with the character in <Picture 1>.' in text
    assert '24 fps' in text and '4–5 s' in text and 'Turbo je isključen' in text and 'Originalni zvuk' in text


def test_download_and_install_are_identical_and_nondestructive(monkeypatch, tmp_path):
    with TestClient(m.app) as client:
        r = client.get('/api/selfism/workflow/akatz_character_swap')
    assert r.status_code == 200 and r.content == WF.read_bytes()
    assert s.AKATZ_CHARACTER_SWAP_WORKFLOW_NAME == 'MiniMax_H3_akatz_Character_Swap_v1.json'
    assert s.AKATZ_CHARACTER_SWAP_WORKFLOW_NAME in r.headers['content-disposition']
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
    saved = tmp_path / 'user/default/workflows/Selfism' / s.AKATZ_CHARACTER_SWAP_WORKFLOW_NAME
    wf = {'selfism_profile': 'akatz_character_swap', 'precision': 'fp8', 'files': [{'id': 'x'}], 'pip_packages': []}
    asyncio.run(ctrl._install_workflow(wf))
    assert saved.read_bytes() == WF.read_bytes()
    saved.write_text('{"user_edited":true}')
    asyncio.run(ctrl._install_workflow(wf))
    assert json.loads(saved.read_text()) == {'user_edited': True}

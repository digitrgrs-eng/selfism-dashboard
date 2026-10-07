"""minimax swap 3 card: MiniMax H3 Studio (SAM3.1 masked video inpainting, bundled local Studio node packs)."""
from __future__ import annotations
import asyncio
import hashlib
import importlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.test_minimax import start_install

m = importlib.import_module('launcher.app')
s = importlib.import_module('launcher.selfism')
ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / 'catalog/selfism.json').read_text(encoding='utf-8'))
WF = ROOT / 'selfism_workflows/minimax_swap_3.json'
# sha256 of the user-supplied "MiniMax H3 Studio.json"; shipped byte-for-byte.
SUPPLIED_SHA256 = 'c3bdb4888fdce505466d7f978df7c8535df8ce36c701d359f93c016fe6e5c36c'
IDS = ['mm-ref2va', 'mm-lora-turbo-v4-600', 'r2v-lora-turbo-8step', 'mm-enc-nvfp4', 'mm-vae-int8', 'mm-vae-audio', 'mm-sam31']
PACKS = ['ComfyUI-H3-Studio', 'ComfyUI-H3-Studio-Support']
# sha256 of the user's local node files (CONTENTS.json of the "minimax swap" folder).
PACK_FILES = {
    'ComfyUI-H3-Studio/__init__.py': '15264e35545fba14e4b113c4a34c205c652266ad2406c0bf534f1914a946dfc0',
    'ComfyUI-H3-Studio/studio_generate.py': 'c3cce44ac7c53f28c1a42a5c424604a88fc93f4b90ec0ef515b6d42496ba3bb1',
    'ComfyUI-H3-Studio/studio_media.py': '0dec6db0498712d1523ca48cac388ab00c9abbe48f51154e8c1ad0b3f22a97b8',
    'ComfyUI-H3-Studio/web/studio.js': '49a48420d0be10e4d57127a7b5b4bf71ec8daf74dc563dc56deb719185d8bf5b',
    'ComfyUI-H3-Studio/requirements.txt': '5bbbf856ada0d65defdfd9775d763703b11b0fa4a9ad5ad854d797b57857f48b',
    'ComfyUI-H3-Studio-Support/__init__.py': '4a7f9c46ed6e5042655d700609d37b7775ddda0705e2ceae87344006126f91f3',
    'ComfyUI-H3-Studio-Support/safe.py': '64e28d73537f5ea0060dd6d9a9af13e4df84651d242ec13b6c4d60ca88fabebe',
    'ComfyUI-H3-Studio-Support/mask_controls.py': 'c3c2bb45c94dfd31eb54d9e2adc220aee740773bd80ad185edfaf6e6d9471575',
    'ComfyUI-H3-Studio-Support/requirements.txt': '15c45379b4ad5810d20faa8b66c177ac026ce3f473c9730088f206017e146297',
}


def test_catalog_lists():
    assert CATALOG['minimax_swap_3_files'] == IDS and CATALOG['minimax_swap_3_nodes'] == []
    assert CATALOG['minimax_swap_3_bundled_nodes'] == PACKS
    assert all(k in CATALOG['files'] for k in IDS)
    assert len({CATALOG['files'][k]['destination'] for k in IDS}) == len(IDS)
    assert CATALOG['files']['mm-sam31']['destination'] == 'models/checkpoints/sam3.1_multiplex_fp16.safetensors'
    assert CATALOG['files']['mm-lora-turbo-v4-600']['destination'] == 'models/loras/minimax_h3_turbo_v4_step600_pruned_comfyui.safetensors'


def test_bundled_packs_are_byte_identical_and_in_the_image():
    for rel, digest in PACK_FILES.items():
        assert hashlib.sha256((ROOT / 'bundled_nodes' / rel).read_bytes()).hexdigest() == digest, rel
    assert 'COPY --link bundled_nodes/' in (ROOT / 'Dockerfile').read_text()
    req = ''.join((ROOT / 'bundled_nodes' / p / 'requirements.txt').read_text() for p in PACKS).lower()
    assert 'torch' not in req and 'cu1' not in req


def test_install_selects_files_and_card_is_on_both_pages(monkeypatch):
    r, captured, html = start_install(monkeypatch, {'profile': 'minimax_swap_3'})
    assert r.status_code == 200
    assert [f['id'] for f in captured['files']] == IDS
    assert captured['selfism_profile'] == 'minimax_swap_3' and captured['selfism_repair'] is False
    assert captured['pip_packages'] == [] and captured['model_links'] == [] and captured['custom_nodes'] == []
    total = sum(CATALOG['files'][k]['size_bytes'] for k in IDS)
    scripts = (ROOT / 'scripts/selfism-section.html').read_text(encoding='utf-8')
    for page in (html, scripts):
        assert 'data-sf-action="minimax_swap_3"' in page and '<h2>minimax swap 3</h2>' in page
        assert f'{total/1e9:.1f}'.replace('.', ',') + ' GB' in page
        assert 'data-sf-action="minimax_swap_2"' in page and 'data-sf-action="god_mode"' in page


def test_workflow_unchanged_and_models_installed():
    raw = WF.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SUPPLIED_SHA256
    w = json.loads(raw)
    types = {n['type'] for n in w['nodes']} | {n['type'] for g in w['definitions']['subgraphs'] for n in g['nodes']}
    assert {'H3StudioMedia', 'H3StudioMaskPreview', 'H3StudioModelDownloads', 'H3StudioSource', 'H3StudioMasks', 'H3StudioGenerate'} <= types
    installed = {Path(CATALOG['files'][k]['destination']).name for k in IDS}
    gen = (ROOT / 'bundled_nodes/ComfyUI-H3-Studio/studio_generate.py').read_text()
    media = (ROOT / 'bundled_nodes/ComfyUI-H3-Studio/studio_media.py').read_text()
    # Every model the Video inpainting / Reference to video path loads is installed.
    for name in ('REF_MODEL', 'REF_TURBO', 'VIDEO_VAE', 'AUDIO_VAE', 'TEXT_ENCODER'):
        value = next(l.split("'")[1] for l in gen.splitlines() if l.startswith(name + ' ='))
        assert value in installed, name
    assert "SAM_CHECKPOINT = 'sam3.1_multiplex_fp16.safetensors'" in media
    loras = json.loads(next(n for n in w['nodes'] if n['id'] == 2)['widgets_values'][-1])
    assert {row['name'] for row in loras if row['enabled']} <= installed


def test_helper_installs_requirements_with_comfy_python_and_copies_both_packs(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'COMFYUI_DIR', tmp_path)
    old = tmp_path / 'custom_nodes/ComfyUI-H3-Studio'
    old.mkdir(parents=True); (old / '__init__.py').write_text('# previous')
    calls = []
    ctrl = type(m.selfism_controller)()
    async def run(*command, **kwargs): calls.append(command); return 0, 'ready'
    monkeypatch.setattr(ctrl, '_run_process', run)
    python = tmp_path / '.venv-cu128/bin/python'
    asyncio.run(ctrl._install_h3_studio_nodes(python))
    assert all(call[0] == python for call in calls)
    assert calls[0][1:4] == ('-m', 'pip', 'install') and 'torch' not in ' '.join(map(str, calls[0]))
    for pack in PACKS:
        assert (tmp_path / 'custom_nodes' / pack / '__init__.py').read_bytes() == (ROOT / 'bundled_nodes' / pack / '__init__.py').read_bytes()
    backups = list((tmp_path / 'user/h3_studio_backups').glob('*/ComfyUI-H3-Studio/__init__.py'))
    assert len(backups) == 1 and backups[0].read_text() == '# previous'


def test_failed_requirements_leave_existing_nodes(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'COMFYUI_DIR', tmp_path)
    ctrl = type(m.selfism_controller)()
    async def run(*args, **kwargs): return 1, 'conflict'
    monkeypatch.setattr(ctrl, '_run_process', run)
    with pytest.raises(RuntimeError, match='H3 Studio dependencies failed'):
        asyncio.run(ctrl._install_h3_studio_nodes(tmp_path / '.venv-cu128/bin/python'))
    assert not (tmp_path / 'custom_nodes/ComfyUI-H3-Studio').exists()


def test_download_and_install_are_identical_and_nondestructive(monkeypatch, tmp_path):
    with TestClient(m.app) as client:
        r = client.get('/api/selfism/workflow/minimax_swap_3')
    assert r.status_code == 200 and r.content == WF.read_bytes()
    assert s.MINIMAX_SWAP_3_WORKFLOW_NAME == 'MiniMax_H3_Studio_Swap.json'
    assert s.MINIMAX_SWAP_3_WORKFLOW_NAME in r.headers['content-disposition']
    monkeypatch.setattr(m, 'COMFYUI_DIR', tmp_path)
    monkeypatch.setattr(m, 'COMFYUI_VENV', tmp_path / '.venv')
    python = m.COMFYUI_VENV / 'bin/python'
    python.parent.mkdir(parents=True); python.touch()
    ctrl = type(m.selfism_controller)()
    async def ready(*args): pass
    async def run(*args, **kwargs): return 0, 'torch==2.10.0'
    async def sources(host, files): return files, []
    monkeypatch.setattr(ctrl, '_wait_for_comfyui', ready)
    monkeypatch.setattr(ctrl, '_run_process', run)
    monkeypatch.setattr(m.JobController, '_install_workflow', ready)
    monkeypatch.setattr(s, 'resolve_sources', sources)
    saved = tmp_path / 'user/default/workflows/Selfism' / s.MINIMAX_SWAP_3_WORKFLOW_NAME
    wf = {'selfism_profile': 'minimax_swap_3', 'precision': 'fp8', 'files': [{'id': 'x'}], 'pip_packages': []}
    asyncio.run(ctrl._install_workflow(wf))
    assert saved.read_bytes() == WF.read_bytes()
    assert (tmp_path / 'custom_nodes/ComfyUI-H3-Studio-Support/safe.py').is_file()
    saved.write_text('{"user_edited":true}')
    asyncio.run(ctrl._install_workflow(wf))
    assert json.loads(saved.read_text()) == {'user_edited': True}

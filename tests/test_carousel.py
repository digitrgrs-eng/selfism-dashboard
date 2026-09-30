import asyncio
import copy
import importlib
import json
import hashlib
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

m = importlib.import_module('launcher.app')
s = importlib.import_module('launcher.selfism')
ROOT = Path(__file__).resolve().parents[1]


def test_carousel_full_install_and_model_coverage(monkeypatch):
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url, **kwargs):
            data = ({'queue_running': [], 'queue_pending': []} if url.endswith('/queue') else
                    {'files': [{'id': 3114237, 'hashes': {'SHA256': 'a' * 64}}]})
            return httpx.Response(200, json=data, request=httpx.Request('GET', url))
    captured = {}
    async def start(workflow):
        captured.update(workflow)
        return {'status': 'running'}
    monkeypatch.setenv('SELFISM_AUTO_REPAIR', '0')
    monkeypatch.setenv('CIVITAI_TOKEN', 'test-only')
    monkeypatch.setattr(httpx, 'AsyncClient', Client)
    monkeypatch.setattr(m.selfism_controller, 'start', start)
    with TestClient(m.app) as client:
        assert client.post('/api/selfism/install', json={'profile': 'carousel', 'precision': 'bf16'}).status_code == 200
        w = client.get('/api/selfism/workflow/carousel').json()
        assert 'data-sf-action="carousel"' in client.get('/').text
    files = {f['id']: f for f in captured['files']}
    assert captured['precision'] == 'fp8'
    assert not {'bf16', 'int8', 'krea-turbo', 'edit'} & files.keys()
    assert {'fp8', 'encoder', 'vae', 'llm', 'mmproj', 'upscale', 'depth', 'depth-anything',
            'qwen-edit-2511', 'qwen-edit-encoder', 'qwen-vae', 'person-seg'} == files.keys()
    assert all(len(f['sha256']) == 64 for f in files.values())
    assert len({f['destination'] for f in files.values()}) == len(files)
    names = {Path(f['destination']).name for f in files.values()}
    nodes = w['nodes'] + [n for g in w['definitions']['subgraphs'] for n in g['nodes']]
    for n in nodes:
        if n['type'] in ('UNETLoader', 'VAELoader', 'CLIPLoader', 'UpscaleModelLoader',
                         'Krea2ControlLoRALoader', 'DepthAnythingV2Preprocessor'):
            assert n['widgets_values'][0] in names
    assert captured['model_links'][0]['source'] == files['depth-anything']['destination']
    assert captured['selfism_repair'] is True
    assert (ROOT/'bundled_nodes/ComfyUI-AIO-Carousel/__init__.py').is_file()
    assert 'COPY --link bundled_nodes/' in (ROOT/'Dockerfile').read_text()


def spec():
    original = {'name': 'test', 'destination': 'models/vae/test', 'size_bytes': 12,
                'sha256': 'a'*64, 'url': 'https://origin/model', 'auth': 'none'}
    selected = dict(original, url='https://r2/model?signature=secret', parallel=True,
                    verify=True, _selfism_source='R2', _selfism_original=copy.deepcopy(original))
    return original, selected


def test_transfer_failure_falls_back_in_order_without_changing_hash_or_path(monkeypatch):
    original, selected = spec()
    calls = []
    async def download(self, client, file, *args):
        calls.append(file)
        if file['url'] != original['url']:
            raise httpx.ConnectError('failed https://r2/model?signature=secret')
        return 12
    async def rapid(host, files):
        assert files == [original]
        return [dict(original, url='https://rapid/model?token=hidden', parallel=True, verify=True)], ['RapidCache selected']
    monkeypatch.setattr(m.JobController, '_download_file', download)
    monkeypatch.setattr(s, 'resolve_rapidcache', rapid)
    ctrl = type(m.selfism_controller)()
    assert asyncio.run(ctrl._download_file(None, selected, 0, 1, 0, 12, 88)) == 12
    assert [f['url'].split('/')[2] for f in calls] == ['r2', 'rapid', 'origin']
    assert all(f['sha256'] == original['sha256'] and f['destination'] == original['destination'] for f in calls)
    assert not ctrl.state.warnings
    assert '_selfism_original' not in calls[0]


def test_r2_success_does_not_request_rapidcache(monkeypatch):
    _, selected = spec()
    async def download(*args): return 12
    async def rapid(*args): raise AssertionError('Unnecessary RapidCache lookup')
    monkeypatch.setattr(m.JobController, '_download_file', download)
    monkeypatch.setattr(s, 'resolve_rapidcache', rapid)
    assert asyncio.run(type(m.selfism_controller)()._download_file(None, selected)) == 12


def test_real_transfer_retries_http_failure_and_verifies_origin_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'COMFYUI_DIR', tmp_path)
    monkeypatch.setattr(m, 'ARIA2C_PATH', None)
    monkeypatch.setattr(m, '_scratch_dir', None)
    original, selected = spec()
    payload = b'test-payload'
    original['sha256'] = selected['sha256'] = hashlib.sha256(payload).hexdigest()
    selected['_selfism_original'] = original
    requested = []
    def response(request):
        requested.append(request.url.host)
        return httpx.Response(403) if request.url.host == 'r2' else httpx.Response(200, content=payload)
    async def rapid(host, files): return files, []
    monkeypatch.setattr(s, 'resolve_rapidcache', rapid)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            return await type(m.selfism_controller)()._download_file(client, selected, 0, 1, 0, 12, 88)
    assert asyncio.run(run()) == len(payload)
    assert requested == ['r2', 'origin']
    assert (tmp_path/original['destination']).read_bytes() == payload


@pytest.mark.parametrize('error', [m.InstallCancelled(), OSError('disk full')])
def test_cancel_and_disk_errors_do_not_trigger_network_fallback(monkeypatch, error):
    _, selected = spec()
    async def download(*args): raise error
    async def rapid(*args): raise AssertionError('Must not fallback on cancellation or disk error')
    monkeypatch.setattr(m.JobController, '_download_file', download)
    monkeypatch.setattr(s, 'resolve_rapidcache', rapid)
    with pytest.raises(type(error)):
        asyncio.run(type(m.selfism_controller)()._download_file(None, selected))


def test_final_error_redacts_signed_url(monkeypatch):
    original, _ = spec()
    async def download(*args):
        raise httpx.ConnectError('failed https://origin/model?signature=secret')
    monkeypatch.setattr(m.JobController, '_download_file', download)
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(type(m.selfism_controller)()._download_file(None, original))
    assert 'secret' not in str(exc.value)


def test_corrupt_partials_removed_but_models_kept(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'COMFYUI_DIR', tmp_path)
    monkeypatch.setattr(m, 'scratch_dir', lambda: None)
    original, selected = spec()
    final = tmp_path/original['destination']
    final.parent.mkdir(parents=True)
    final.write_bytes(b'existing-model')
    partial = final.with_name(final.name+'.part')
    control = final.with_name(final.name+'.part.aria2')
    partial.write_bytes(b'corrupt'); control.write_bytes(b'control')
    async def download(self, client, file, *args):
        if file['url'] != original['url']:
            raise RuntimeError('Checksum verification failed')
        assert not partial.exists() and not control.exists()
        assert final.read_bytes() == b'existing-model'
        return 12
    async def rapid(host, files): return files, []
    monkeypatch.setattr(m.JobController, '_download_file', download)
    monkeypatch.setattr(s, 'resolve_rapidcache', rapid)
    assert asyncio.run(type(m.selfism_controller)()._download_file(None, selected)) == 12


def test_helper_uses_comfy_python_and_preserves_previous_copy(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'COMFYUI_DIR', tmp_path)
    target = tmp_path/'custom_nodes/ComfyUI-AIO-Carousel'
    target.mkdir(parents=True)
    (target/'__init__.py').write_text('# previous local version')
    calls = []
    ctrl = type(m.selfism_controller)()
    async def run(*command, **kwargs): calls.append(command); return 0, 'ready'
    monkeypatch.setattr(ctrl, '_run_process', run)
    python = tmp_path/'.venv-cu128/bin/python'
    asyncio.run(ctrl._install_carousel_helper(python))
    assert all(call[0] == python for call in calls)
    assert calls[0][1:4] == ('-m', 'pip', 'install')
    backups = list((tmp_path/'user/carousel_backups').glob('*/__init__.py'))
    assert len(backups) == 1 and backups[0].read_text() == '# previous local version'
    assert (target/'__init__.py').read_bytes() == (ROOT/'bundled_nodes/ComfyUI-AIO-Carousel/__init__.py').read_bytes()


def test_failed_helper_install_leaves_existing_nodes(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'COMFYUI_DIR', tmp_path)
    target = tmp_path/'custom_nodes/ComfyUI-AIO-Carousel'
    target.mkdir(parents=True); (target/'__init__.py').write_text('# original')
    ctrl = type(m.selfism_controller)()
    async def run(*args, **kwargs): return 1, 'pip dependency conflict'
    monkeypatch.setattr(ctrl, '_run_process', run)
    with pytest.raises(RuntimeError, match='Carousel dependencies failed'):
        asyncio.run(ctrl._install_carousel_helper(tmp_path/'.venv-cu128/bin/python'))
    assert (target/'__init__.py').read_text() == '# original'


def test_installed_workflow_keeps_qwen_loader_and_existing_user_edits(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'COMFYUI_DIR', tmp_path)
    monkeypatch.setattr(m, 'COMFYUI_VENV', tmp_path/'.venv-cu128')
    python = m.COMFYUI_VENV/'bin/python'
    python.parent.mkdir(parents=True); python.touch()
    lora = tmp_path/'models/loras/millie.safetensors'
    lora.parent.mkdir(parents=True); lora.write_bytes(b'private')
    ctrl = type(m.selfism_controller)()
    async def ready(*args): pass
    async def run(*args, **kwargs): return 0, 'torch==2.8.0'
    async def sources(host, files): return files, []
    monkeypatch.setattr(ctrl, '_wait_for_comfyui', ready)
    monkeypatch.setattr(ctrl, '_run_process', run)
    monkeypatch.setattr(m.JobController, '_install_workflow', ready)
    monkeypatch.setattr(s, 'resolve_sources', sources)
    workflow = {'selfism_profile': 'carousel', 'precision': 'fp8', 'files': [{'id': 'test'}]}
    asyncio.run(ctrl._install_workflow(workflow))
    destination = tmp_path/'user/default/workflows/Selfism/Selfism_AIO_m1lli3_CAROUSEL_QWEN_2511_v1.json'
    w = json.loads(destination.read_text(encoding='utf-8'))
    assert {n['widgets_values'][0] for n in w['nodes'] if n['type'] == 'UNETLoader'} == {
        'selforaV21NightFix_selforaV21Fp8.safetensors', 'qwen_image_edit_2511_fp8mixed.safetensors'}
    loaders = [n for n in w['nodes'] if n['type'] == 'Power Lora Loader (rgthree)']
    assert any(row.get('lora') == 'millie.safetensors' and row.get('on')
               for n in loaders for row in n['widgets_values'] if isinstance(row, dict))
    destination.write_text('{"user_edited":true}')
    asyncio.run(ctrl._install_workflow(workflow))
    assert json.loads(destination.read_text()) == {'user_edited': True}
    assert lora.read_bytes() == b'private'

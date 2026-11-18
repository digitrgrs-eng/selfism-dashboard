"""Qwen 2.1 Dataset Generator (24 images) card: the published 1050R/10sorLabs graph, GPU-only SeedVR2."""
import asyncio
import hashlib
import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from test_full import CATALOG, ROOT, m, start_install
from test_hearmeman_video import widget_models, workflow_nodes
import launcher.selfism as s

PROFILE = 'qwen21_dataset_generator'
WORKFLOW = ROOT / 'selfism_workflows/qwen21_dataset_generator.json'
ORIGINAL = ROOT / 'selfism_workflows/sources/qwen21_dataset_generator_original.json'
ORIGINAL_SHA = '673fab8cc03a18e81ca2bc717338a5312c0723abe8e2dbc8067f245e478905cd'
OUT = 'Qwen21_Dataset_Generator_24.json'
FILES = {
    'redqw21-unlocked-v2-fp8': ('models/diffusion_models/redgptK2T2W6A8REDQW21_redqw21UNLOCKEDV2_fp8.safetensors', 7116801326),
    'qwen3vl-8b-bf16': ('models/text_encoders/qwen3vl_8b_bf16.safetensors', 17534334616),
    'qwen21-vae': ('models/vae/qwen_image_2.1_vae_bf16.safetensors', 675509688),
    'nsfw-qwen-thesealpacas-v2': ('models/loras/civitai_lora_3357315_3244894.safetensors', 79744352),
    'lenovo-qwen21': ('models/loras/lenovo_qwen21.safetensors', 79743896),
    'bfs-head-qwen21': ('models/loras/bfs_head_v1.1_qwen_2.1.safetensors', 260096144),
    'krea-turbo': ('models/diffusion_models/krea2_turbo_fp8_scaled.safetensors', 13141730784),
    'encoder': ('models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors', 6020316384),
    'edit-vae': ('models/vae/krea2RealVae_v10.safetensors', 507591212),
    'seedvr2-7b-fp16': ('models/SEEDVR2/seedvr2_ema_7b_fp16.safetensors', 16479334424),
    'seedvr2-vae-fp16': ('models/SEEDVR2/ema_vae_fp16.safetensors', 501324814),
}
NODES = ['rgthree-comfy', 'ComfyUI-KJNodes', 'ComfyUI-Easy-Use', 'ComfyUI-Custom-Scripts',
         'ComfyUI-SeedVR2_VideoUpscaler']
PACK_TYPES = {
    'Fast Groups Bypasser (rgthree)': 'rgthree-comfy', 'Power Lora Loader (rgthree)': 'rgthree-comfy',
    'Label (rgthree)': 'rgthree-comfy',
    'SetNode': 'ComfyUI-KJNodes', 'GetNode': 'ComfyUI-KJNodes', 'StringConstantMultiline': 'ComfyUI-KJNodes',
    'easy indexAnything': 'ComfyUI-Easy-Use', 'easy stringToIntList': 'ComfyUI-Easy-Use',
    'MathExpression|pysssss': 'ComfyUI-Custom-Scripts',
    'SeedVR2LoadDiTModel': 'ComfyUI-SeedVR2_VideoUpscaler', 'SeedVR2LoadVAEModel': 'ComfyUI-SeedVR2_VideoUpscaler',
    'SeedVR2VideoUpscaler': 'ComfyUI-SeedVR2_VideoUpscaler',
}
# Everything else is ComfyUI core (all present in the baked v0.38.1).
CORE = {'BatchImagesNode', 'CLIPLoader', 'CLIPTextEncode', 'CustomCombo', 'EmptyLatentImage', 'GetImageSize',
        'ImageScaleBy', 'ImageStitch', 'KSampler', 'LatentUpscaleBy', 'LoadImage', 'LoraLoaderModelOnly',
        'ModelSamplingAuraFlow', 'Note', 'PreviewAny', 'PreviewImage', 'PrimitiveFloat', 'PrimitiveStringMultiline',
        'RegexExtract', 'SaveImage', 'StringReplace', 'TextEncodeQwenImage21', 'TextGenerate', 'UNETLoader',
        'VAEDecode', 'VAEEncode', 'VAELoader'}
OFFLOAD = {'SeedVR2LoadDiTModel': 4, 'SeedVR2LoadVAEModel': 9, 'SeedVR2VideoUpscaler': 11}


def _strip(data):
    """Published graph minus the card's edits (offload devices, description, example images)."""
    data = json.loads(json.dumps(data))
    for group in data['definitions']['subgraphs']:
        for node in group['nodes']:
            if node['type'] in OFFLOAD:
                node['widgets_values'][OFFLOAD[node['type']]] = None
                node['widgets_values_named']['offload_device'] = None
    for node in data['nodes']:
        if node['id'] == 5910 or node['type'] == 'LoadImage':
            node['widgets_values'][0] = None
            node.pop('widgets_values_named', None)
    return data


def test_catalog_pins_every_model_the_workflow_loads():
    assert CATALOG[PROFILE + '_files'] == list(FILES)
    for key, (dest, size) in FILES.items():
        item = CATALOG['files'][key]
        assert item['id'] == key and item['destination'] == dest and item['size_bytes'] == size
        assert re.fullmatch(r'[0-9a-f]{64}', item['sha256'])
    lenovo = CATALOG['files']['lenovo-qwen21']
    # Danrisi's official repo, pinned; same bytes as Civitai "Lenovo UltraReal v1.0 Qwen2.1".
    assert lenovo['url'].startswith('https://huggingface.co/Danrisi/lenovo_qwen2.1/resolve/')
    assert re.search(r'/resolve/[0-9a-f]{40}/lenovo_qwen21\.safetensors$', lenovo['url'])
    assert lenovo['sha256'] == '7f1e95c3ec1e9c7cea197f7b6f80f601af8e6b479f6ead78ab16df542f447c0e'
    assert lenovo['auth'] == 'none'
    # Shares every other model with the Auto Carousel card.
    assert set(FILES) - set(CATALOG['qwen21_auto_carousel_files']) == {'lenovo-qwen21', 'bfs-head-qwen21'}


def test_workflow_is_the_published_graph_with_only_the_card_edits():
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == ORIGINAL_SHA
    original = json.loads(ORIGINAL.read_text(encoding='utf-8'))
    data = json.loads(WORKFLOW.read_text(encoding='utf-8'))
    assert _strip(data) == _strip(original)
    _, nodes = workflow_nodes(WORKFLOW)
    assert len(nodes) == 924


def test_seedvr2_runs_on_gpu_only():
    _, nodes = workflow_nodes(WORKFLOW)
    seen = 0
    for node in nodes:
        if node['type'] in OFFLOAD:
            seen += 1
            assert node['widgets_values'][OFFLOAD[node['type']]] == 'cuda:0'
            assert node['widgets_values_named']['offload_device'] == 'cuda:0'
            if 'device' in node['widgets_values_named']:
                assert node['widgets_values_named']['device'] == 'cuda:0'
        for value in (node.get('widgets_values_named') or {}).values():
            assert value != 'cpu', node['type']
    assert seen == 72


def test_description_and_inputs_are_neutral():
    data = json.loads(WORKFLOW.read_text(encoding='utf-8'))
    by_id = {node['id']: node for node in data['nodes']}
    assert by_id[5910]['title'] == 'Model description' and by_id[5910]['widgets_values'] == ['Adult woman.']
    loads = {node['title']: node['widgets_values'][0] for node in data['nodes'] if node['type'] == 'LoadImage'}
    assert loads == {'Load Face': 'example.png', 'Load Body+Face': 'example.png'}


def test_every_node_and_model_is_provided():
    data, nodes = workflow_nodes(WORKFLOW)
    subgraphs = {group['id'] for group in data['definitions']['subgraphs']}
    used_packs = set()
    for node in nodes:
        if node['type'] in subgraphs:
            continue
        if node['type'] in PACK_TYPES:
            used_packs.add(PACK_TYPES[node['type']])
        else:
            assert node['type'] in CORE, node['type']
    assert used_packs == set(NODES) and CATALOG[PROFILE + '_nodes'] == NODES
    assert set(NODES) <= {n['name'] for n in CATALOG['nodes']}
    assert widget_models(nodes) == {Path(dest).name for dest, _ in FILES.values()}
    loras = {row['lora']: (row['on'], row['strength']) for node in nodes
             if node['type'] == 'Power Lora Loader (rgthree)'
             for row in node['widgets_values'] if isinstance(row, dict) and 'lora' in row}
    assert loras['lenovo_qwen21.safetensors'] == (True, 0.7)
    assert loras['civitai_lora_3357315_3244894.safetensors'][0] is True


def test_install_payload_card_and_download(monkeypatch):
    response, captured, html = start_install(monkeypatch, {'profile': PROFILE, 'precision': 'int8'})
    assert response.status_code == 200
    assert captured['selfism_profile'] == PROFILE and captured['precision'] == 'fp8'
    assert captured['pip_packages'] == [] and captured['model_links'] == []
    assert [item['id'] for item in captured['files']] == list(FILES)
    assert sorted(node['name'] for node in captured['custom_nodes']) == sorted(NODES)
    total = sum(size for _, size in FILES.values())
    for page in ((ROOT / 'launcher/static/index.html').read_text(encoding='utf-8'),
                 (ROOT / 'scripts/selfism-section.html').read_text(encoding='utf-8'), html):
        card = page.split('sf-card-qwen21-dataset-generator')[1].split('</article>')[0]
        assert '<h2>Qwen 2.1 Dataset Generator (24 slike)</h2>' in card
        assert f'data-sf-action="{PROFILE}"' in card and '11 modela' in card and '5 paketa' in card
        assert f'{total / 1e9:.1f}'.replace('.', ',') in card and 'CIVITAI_TOKEN' in card
        assert '48 GB' in card and 'cuda:0' in card
        assert f'/api/selfism/workflow/{PROFILE}' in card
        for other in ('sf-card-qwen21-edit', 'sf-card-qwen21-swap', 'sf-card-qwen21-auto-carousel',
                      'sf-card-krast-org-pose-millie'):
            assert other in page
    with TestClient(m.app) as client:
        downloaded = client.get(f'/api/selfism/workflow/{PROFILE}')
        assert downloaded.status_code == 200 and downloaded.content == WORKFLOW.read_bytes()
        assert OUT in downloaded.headers['content-disposition']
        other = client.get('/api/selfism/workflow/qwen21_auto_carousel')
        assert other.content == (ROOT / 'selfism_workflows/qwen21_auto_carousel_10sor.json').read_bytes()


def test_installer_saves_without_overwriting_user_edits(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'COMFYUI_DIR', tmp_path)
    monkeypatch.setattr(m, 'COMFYUI_VENV', tmp_path / '.venv')
    python = m.COMFYUI_VENV / 'bin/python'; python.parent.mkdir(parents=True); python.touch()
    ctrl = type(m.selfism_controller)()
    async def ready(*args): pass
    async def run(*args, **kwargs): return 0, 'torch==2.10.0'
    async def sources(host, files): return files, []
    monkeypatch.setattr(ctrl, '_wait_for_comfyui', ready)
    monkeypatch.setattr(ctrl, '_run_process', run)
    monkeypatch.setattr(m.JobController, '_install_workflow', ready)
    monkeypatch.setattr(s, 'resolve_sources', sources)
    wf = {'selfism_profile': PROFILE, 'precision': 'fp8', 'files': [{'id': 'x'}]}
    asyncio.run(ctrl._install_workflow(wf))
    path = tmp_path / 'user/default/workflows/Selfism' / OUT
    assert path.read_bytes() == WORKFLOW.read_bytes()
    path.write_text('user edit')
    asyncio.run(ctrl._install_workflow(wf))
    assert path.read_text() == 'user edit'

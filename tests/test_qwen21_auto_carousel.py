"""Qwen 2.1 Auto Carousel (10SOR) card: the 4-slot REDQW21 carousel workflow, byte-for-byte."""
import hashlib
import json
import re
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from test_full import CATALOG, ROOT, m, start_install
from test_hearmeman_video import widget_models, workflow_nodes

PROFILE = 'qwen21_auto_carousel'
WORKFLOW = ROOT / 'selfism_workflows/qwen21_auto_carousel_10sor.json'
SHA = '08f15a1f0248b2022c7d2a5ad06afc37447e8a0e3a4b51648373494b8188d4f7'
FILES = {
    'redqw21-unlocked-v2-fp8': ('models/diffusion_models/redgptK2T2W6A8REDQW21_redqw21UNLOCKEDV2_fp8.safetensors', 7116801326),
    'qwen3vl-8b-bf16': ('models/text_encoders/qwen3vl_8b_bf16.safetensors', 17534334616),
    'qwen21-vae': ('models/vae/qwen_image_2.1_vae_bf16.safetensors', 675509688),
    'nsfw-qwen-thesealpacas-v2': ('models/loras/civitai_lora_3357315_3244894.safetensors', 79744352),
    'krea-turbo': ('models/diffusion_models/krea2_turbo_fp8_scaled.safetensors', 13141730784),
    'encoder': ('models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors', 6020316384),
    'edit-vae': ('models/vae/krea2RealVae_v10.safetensors', 507591212),
    'seedvr2-7b-fp16': ('models/SEEDVR2/seedvr2_ema_7b_fp16.safetensors', 16479334424),
    'seedvr2-vae-fp16': ('models/SEEDVR2/ema_vae_fp16.safetensors', 501324814),
}
NODES = ['rgthree-comfy', 'ComfyUI-KJNodes', 'ComfyUI-Easy-Use', 'ComfyUI-Custom-Scripts',
         'ComfyUI-SeedVR2_VideoUpscaler', 'ComfyUI-10SOR-Regeneration-Gate']
# Node types and the pack that provides them (everything else is ComfyUI core v0.38.1).
PACK_TYPES = {
    '10SORRegenerationGate': 'ComfyUI-10SOR-Regeneration-Gate',
    'Fast Groups Muter (rgthree)': 'rgthree-comfy', 'Power Lora Loader (rgthree)': 'rgthree-comfy',
    'SetNode': 'ComfyUI-KJNodes', 'GetNode': 'ComfyUI-KJNodes', 'StringConstantMultiline': 'ComfyUI-KJNodes',
    'easy indexAnything': 'ComfyUI-Easy-Use', 'easy stringToIntList': 'ComfyUI-Easy-Use',
    'MathExpression|pysssss': 'ComfyUI-Custom-Scripts',
    'SeedVR2LoadDiTModel': 'ComfyUI-SeedVR2_VideoUpscaler', 'SeedVR2LoadVAEModel': 'ComfyUI-SeedVR2_VideoUpscaler',
    'SeedVR2VideoUpscaler': 'ComfyUI-SeedVR2_VideoUpscaler',
}
CORE = {'BatchImagesNode', 'CLIPLoader', 'CLIPTextEncode', 'CustomCombo', 'EmptyLatentImage', 'GetImageSize',
        'ImageScaleBy', 'KSampler', 'LoadImage', 'ModelSamplingAuraFlow', 'Note', 'PreviewAny', 'PreviewImage',
        'PrimitiveBoolean', 'RegexExtract', 'SaveImage', 'SeedNode', 'TextEncodeQwenImage21', 'TextGenerate',
        'UNETLoader', 'VAEDecode', 'VAEEncode', 'VAELoader'}


def test_catalog_pins_every_model_the_workflow_loads():
    assert CATALOG[PROFILE + '_files'] == list(FILES)
    for key, (dest, size) in FILES.items():
        item = CATALOG['files'][key]
        assert item['id'] == key and item['destination'] == dest and item['size_bytes'] == size
        assert re.fullmatch(r'[0-9a-f]{64}', item['sha256'])
    red = CATALOG['files']['redqw21-unlocked-v2-fp8']
    assert red['auth'] == 'civitai' and (red['civitai_version'], red['civitai_file']) == (3370753, 3258921)
    lora = CATALOG['files']['nsfw-qwen-thesealpacas-v2']
    # Same bytes as Civitai version 3357315 / file 3244894 (Civitai SHA256 83ABA822...).
    assert lora['sha256'] == '83aba822ad8ee10a1e836828f5088259d93bce7b5f3419b46dbc0258c2be46a9'
    assert re.search(r'^https://huggingface.co/.+/resolve/[0-9a-f]{40}/', lora['url'])
    assert re.search(r'^https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/[0-9a-f]{40}/',
                     CATALOG['files']['qwen3vl-8b-bf16']['url'])


def test_workflow_is_the_published_bytes_and_every_node_is_provided():
    assert hashlib.sha256(WORKFLOW.read_bytes()).hexdigest() == SHA
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
    assert used_packs == set(NODES)
    assert CATALOG[PROFILE + '_nodes'] == NODES
    catalog_nodes = {n['name']: n for n in CATALOG['nodes']}
    gate = catalog_nodes['ComfyUI-10SOR-Regeneration-Gate']
    assert gate['repo'] == 'https://github.com/10sorlabs/ComfyUI-10SOR-Regeneration-Gate.git'
    assert gate['ref'] == '7d891c3c1abb64cd51913816ecf3eeca343574bc'    # the commit the workflow was saved with
    assert set(NODES) <= set(catalog_nodes)
    used = widget_models(nodes)
    assert used == {Path(dest).name for dest, _ in FILES.values()}
    loras = {row['lora'] for node in nodes if node['type'] == 'Power Lora Loader (rgthree)'
             for row in node['widgets_values'] if isinstance(row, dict) and 'lora' in row}
    assert loras == {'civitai_lora_3357315_3244894.safetensors'}


def test_ten_sor_gate_is_installed_at_card_time_and_base_is_unchanged():
    sys.path.insert(0, str(ROOT / 'scripts'))
    import base_image
    names = [n['name'] for n in json.loads((ROOT / 'docker/base-nodes.json').read_text())['nodes']]
    assert 'ComfyUI-10SOR-Regeneration-Gate' not in names
    assert (ROOT / 'docker/base-nodes.json').read_text() == base_image.nodes_document()


def test_install_payload_card_and_download(monkeypatch):
    response, captured, html = start_install(monkeypatch, {'profile': PROFILE, 'precision': 'int8'})
    assert response.status_code == 200
    assert captured['selfism_profile'] == PROFILE and captured['precision'] == 'fp8'
    assert captured['selfism_repair'] is False
    assert captured['pip_packages'] == [] and captured['model_links'] == []
    assert [item['id'] for item in captured['files']] == list(FILES)
    assert sorted(node['name'] for node in captured['custom_nodes']) == sorted(NODES)
    total = sum(size for _, size in FILES.values())
    for page in ((ROOT / 'launcher/static/index.html').read_text(encoding='utf-8'),
                 (ROOT / 'scripts/selfism-section.html').read_text(encoding='utf-8'), html):
        card = page.split('sf-card-qwen21-auto-carousel')[1].split('</article>')[0]
        assert '<h2>Qwen 2.1 Auto Carousel (10SOR)</h2>' in card
        assert f'data-sf-action="{PROFILE}"' in card and '9 modela' in card and '6 paketa' in card
        assert f'{total / 1e9:.1f}'.replace('.', ',') in card and 'CIVITAI_TOKEN' in card
        assert f'/api/selfism/workflow/{PROFILE}' in card
        for other in ('sf-card-qwen21-edit', 'sf-card-qwen21-swap'):
            assert other in page
    with TestClient(m.app) as client:
        downloaded = client.get(f'/api/selfism/workflow/{PROFILE}')
        assert downloaded.status_code == 200 and downloaded.content == WORKFLOW.read_bytes()
        assert 'Qwen21_Auto_Carousel_10SOR.json' in downloaded.headers['content-disposition']

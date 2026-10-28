"""Qwen Image 2.1 Edit (Full v2) card: AusBoss Generate + Edit v2.0, shipped byte-for-byte."""
import hashlib
import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from test_full import CATALOG, ROOT, m, start_install
from test_hearmeman_video import widget_models, workflow_nodes

PROFILE = 'qwen21_edit_full'
WORKFLOW = ROOT / 'selfism_workflows/qwen21_edit_full.json'
# Civitai model 2962616, version v2.0 file qwenImage21GenerateEdit_v20.json (SHA256 published by Civitai).
WORKFLOW_SHA = 'dd7f3d8f15f886f62fad353206e142cc87d196af4ac452e747bfd3a53c4589f0'
FILENAME = 'AusBoss_Qwen21_GenerateEdit_Full_v2.json'
AUSBOSS_REF = '714b638ee8324c76bbd0604fae9f2fbab8d96e50'  # Release 2.20.0 (registry Active)
CORE_TYPES = {
    'TextEncodeQwenImage21', 'QwenImage21Cache', 'TextGenerate', 'ComfySwitchNode', 'ManualSigmas',
    'BasicScheduler', 'CLIPLoader', 'KSamplerSelect', 'SamplerCustom', 'StringFormat', 'UNETLoader',
    'VAEDecode', 'VAEEncode', 'VAELoader', 'Reroute',
}
EXPECTED = {
    'qwen_image_2.1_int8_convrot.safetensors': ('models/diffusion_models', 7256783064),
    'qwen3vl_8b_int8_convrot.safetensors': ('models/text_encoders', 9350798360),
    'qwen_image_2.1_vae_bf16.safetensors': ('models/vae', 675509688),
    'qwen-image-2.1-consistency.safetensors': ('models/loras', 159436496),
    'qwen-image-2.1-restore.safetensors': ('models/loras', 159436584),
    'Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors': ('models/loras', 679604800),
}


def lora_rows(nodes):
    rows = []
    for node in nodes:
        if node['type'] == 'AUSBOSS_NODES_LoraLoader':
            rows += json.loads(node['widgets_values'][0])
    return rows


def test_catalog_files_are_pinned_with_private_r2_keys():
    keys = CATALOG[PROFILE + '_files']
    assert len(keys) == 6
    seen = {}
    for key in keys:
        item = CATALOG['files'][key]
        assert item['id'] == key
        assert re.fullmatch(r'[0-9a-f]{64}', item['sha256']), key
        assert item['auth'] == 'none'
        assert item['url'].startswith('https://huggingface.co/')
        assert re.search(r'/resolve/[0-9a-f]{40}/', item['url']), key
        r2_key = item['destination'].removeprefix('models/')
        assert r2_key != item['destination'] and not r2_key.startswith('models/')
        seen[Path(item['destination']).name] = (str(Path(item['destination']).parent), item['size_bytes'])
    assert seen == EXPECTED
    assert PROFILE + '_links' not in CATALOG


def test_ausboss_pack_is_pinned_and_baked():
    nodes = {node['name']: node for node in CATALOG['nodes']}
    pack = nodes['ComfyUI-AusBoss']
    assert pack['repo'] == 'https://github.com/ausboss/ComfyUI-AusBoss.git'
    assert pack['ref'] == AUSBOSS_REF
    assert pack['install_requirements'] is False
    assert CATALOG[PROFILE + '_nodes'] == ['ComfyUI-AusBoss']
    bake = (ROOT / 'docker/bake_comfyui.sh').read_text(encoding='utf-8')
    assert 'ComfyUI-AusBoss' in bake.split('BAKED_NODE_PACKS=(', 1)[1].split(')', 1)[0]


def test_workflow_is_the_published_v2_with_carla_defaults():
    assert hashlib.sha256(WORKFLOW.read_bytes()).hexdigest() == WORKFLOW_SHA
    data, nodes = workflow_nodes(WORKFLOW)
    installed = {Path(CATALOG['files'][k]['destination']).name for k in CATALOG[PROFILE + '_files']}
    loras = {row['name'] for row in lora_rows(nodes)}
    assert loras == {name for name, (folder, _) in EXPECTED.items() if folder == 'models/loras'}
    assert widget_models(nodes) | loras == installed
    subgraphs = {group['id'] for group in data['definitions']['subgraphs']}
    for node in nodes:
        if node['type'] in subgraphs:
            continue
        cnr = (node.get('properties') or {}).get('cnr_id')
        if node['type'].startswith('AUSBOSS_NODES_'):
            assert cnr == 'ausboss-nodes'
        else:
            assert node['type'] in CORE_TYPES, node['type']
            assert cnr in (None, 'comfy-core')
    by_id = {node['id']: node for node in data['nodes']}
    # Edit a picture ON: crop/pad loader and compare run, size comes from the picture.
    assert by_id[6]['mode'] == 0 and by_id[54]['mode'] == 0
    assert by_id[39]['widgets_values_named'] == {'size from your picture': True, 'fast': False}
    # Auto prompt OFF (bypassed), Upscale OFF (bypassed).
    assert by_id[41]['mode'] == 4 and by_id[15]['mode'] == 4
    assert all(by_id[i]['mode'] == 4 for i in (7, 40, 12))
    # Fast OFF: 25 steps, Viggle off. Consistency 0.5 on, Restore off.
    assert by_id[68]['widgets_values'] == [25]
    rows = {row['name']: row for row in lora_rows([by_id[11]])}
    assert rows['Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors']['enabled'] is False
    assert rows['qwen-image-2.1-consistency.safetensors'] == {
        'name': 'qwen-image-2.1-consistency.safetensors', 'strength': 0.5, 'enabled': True, 'triggers': ''}
    assert rows['qwen-image-2.1-restore.safetensors']['enabled'] is False


def test_install_payload_card_and_download(monkeypatch):
    response, captured, html = start_install(monkeypatch, {'profile': PROFILE, 'precision': 'int8'})
    assert response.status_code == 200
    assert captured['selfism_profile'] == PROFILE
    assert captured['precision'] == 'fp8'
    assert captured['selfism_repair'] is False
    assert captured['selfism_repair_paths_only'] is False
    assert captured['pip_packages'] == []
    assert captured['model_links'] == []
    assert [item['id'] for item in captured['files']] == CATALOG[PROFILE + '_files']
    assert [node['name'] for node in captured['custom_nodes']] == ['ComfyUI-AusBoss']
    pages = [
        (ROOT / 'launcher/static/index.html').read_text(encoding='utf-8'),
        (ROOT / 'scripts/selfism-section.html').read_text(encoding='utf-8'),
        html,
    ]
    total = sum(CATALOG['files'][k]['size_bytes'] for k in CATALOG[PROFILE + '_files'])
    size = f'{total / 1e9:.1f}'.replace('.', ',')
    for page in pages:
        card = page.split('sf-card-qwen21-edit')[1].split('</article>')[0]
        assert '<h2>Qwen Image 2.1 Edit (Full v2)</h2>' in card
        assert f'data-sf-action="{PROFILE}"' in card
        assert f'/api/selfism/workflow/{PROFILE}' in card
        assert '6 modela' in card and size in card and '2.20.0' in card
    with TestClient(m.app) as client:
        downloaded = client.get(f'/api/selfism/workflow/{PROFILE}')
    assert downloaded.status_code == 200
    assert downloaded.content == WORKFLOW.read_bytes()
    assert FILENAME in downloaded.headers['content-disposition']


def test_existing_cards_are_unchanged():
    html = (ROOT / 'launcher/static/index.html').read_text(encoding='utf-8')
    for profile in ('hearmeman_scail2_v2', 'hearmeman_wan_v5', 'hearmeman_i2i', 'krast', 'full'):
        assert f'data-sf-action="{profile}"' in html

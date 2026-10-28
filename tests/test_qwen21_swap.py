"""Qwen Image 2.1 Swap (Person + Outfit) card: AusBoss Person Swap v1.0 and Outfit Swap v1.1, byte-for-byte."""
import hashlib
import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from test_full import CATALOG, ROOT, m, start_install
from test_hearmeman_video import widget_models, workflow_nodes
from test_qwen21_edit import CORE_TYPES, lora_rows

PROFILE = 'qwen21_swap'
# Civitai SHA256 of qwenImage21PersonSwap_v10.json (model 2980685) and qwenImage21OutfitSwap_v11.json (2983126).
WORKFLOWS = {
    'qwen21_person_swap': ('119159019f16f447463764589133db314b016446107c406fe6ffe338033b2f9e',
                           'AusBoss_Qwen21_PersonSwap_v1.0.json'),
    'qwen21_outfit_swap': ('f1bbdf3ad2f2e1e064836950a550c7ed29c4f3e8753034c0d829258886849e98',
                           'AusBoss_Qwen21_OutfitSwap_v1.1.json'),
}
SWAP_CORE = CORE_TYPES | {'KSampler', 'LoadImage', 'PrimitiveBoolean', 'RegexMatch', 'RegexReplace',
                          'SplitImageWithAlpha', 'StringConcatenate', 'MarkdownNote'}
SHARED = ['qwen21-int8', 'qwen3vl-8b-int8', 'qwen21-vae', 'qwen21-viggle-turbo-lora']
NEW = {
    'qwen-image-2.1-outfit-swap.safetensors': 159436488,
    'bfs_body_swap_v1.0_qwen_2.1.safetensors': 209753576,
    'bfs_head_v1.1_qwen_2.1.safetensors': 260096144,
    'p_qwen_image_2.1_8step_v0.1-comfyui-T8.safetensors': 335633936,
}


def names(keys):
    return {Path(CATALOG['files'][k]['destination']).name for k in keys}


def test_catalog_reuses_the_edit_models_and_pins_the_new_loras():
    keys = CATALOG[PROFILE + '_files']
    assert keys[:4] == SHARED and len(keys) == 8
    assert set(SHARED) <= set(CATALOG['qwen21_edit_full_files'])
    assert CATALOG[PROFILE + '_nodes'] == ['ComfyUI-AusBoss']
    for key in keys[4:]:
        item = CATALOG['files'][key]
        assert item['id'] == key and item['auth'] == 'none'
        assert re.fullmatch(r'[0-9a-f]{64}', item['sha256'])
        assert re.search(r'^https://huggingface.co/.+/resolve/[0-9a-f]{40}/', item['url'])
        assert item['destination'].startswith('models/loras/')
        assert NEW[Path(item['destination']).name] == item['size_bytes']
    assert names(keys[4:]) == set(NEW)


def test_both_workflows_are_published_bytes_and_match_the_card():
    installed = names(CATALOG[PROFILE + '_files'])
    used_all = set()
    for profile, (sha, _) in WORKFLOWS.items():
        path = ROOT / 'selfism_workflows' / f'{profile}.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
        data, nodes = workflow_nodes(path)
        used = widget_models(nodes) | {row['name'] for row in lora_rows(nodes)}
        assert used <= installed
        used_all |= used
        subgraphs = {group['id'] for group in data['definitions']['subgraphs']}
        for node in nodes:
            if node['type'] in subgraphs:
                continue
            if node['type'].startswith('AUSBOSS_NODES_'):
                assert node['properties']['cnr_id'] == 'ausboss-nodes'
            else:
                assert node['type'] in SWAP_CORE, node['type']
    assert used_all == installed


def test_author_defaults_are_kept():
    p = {n['id']: n for n in json.loads((ROOT / 'selfism_workflows/qwen21_person_swap.json').read_text())['nodes']}
    o = {n['id']: n for n in json.loads((ROOT / 'selfism_workflows/qwen21_outfit_swap.json').read_text())['nodes']}
    assert p[5]['widgets_values'] == [True]          # Swap the whole body
    assert p[53]['widgets_values'] == [25] and o[71]['widgets_values'] == [25]
    assert p[12]['mode'] == 4 and p[14]['mode'] == 4  # second character off
    assert o[6]['widgets_values'] == [True]           # Describe the outfit for me (local Qwen3-VL)
    for node, on in ((p[10], 'bfs_body_swap_v1.0_qwen_2.1.safetensors'), (p[11], 'bfs_head_v1.1_qwen_2.1.safetensors'),
                     (o[70], 'qwen-image-2.1-outfit-swap.safetensors')):
        rows = {row['name']: row['enabled'] for row in lora_rows([node])}
        assert rows.pop(on) is True
        assert set(rows.values()) == {False}


def test_install_payload_card_and_downloads(monkeypatch):
    response, captured, html = start_install(monkeypatch, {'profile': PROFILE, 'precision': 'int8'})
    assert response.status_code == 200
    assert captured['selfism_profile'] == PROFILE and captured['precision'] == 'fp8'
    assert captured['selfism_repair'] is False and captured['selfism_repair_paths_only'] is False
    assert captured['pip_packages'] == [] and captured['model_links'] == []
    assert [item['id'] for item in captured['files']] == CATALOG[PROFILE + '_files']
    assert [node['name'] for node in captured['custom_nodes']] == ['ComfyUI-AusBoss']
    total = sum(CATALOG['files'][k]['size_bytes'] for k in CATALOG[PROFILE + '_files'])
    shared = sum(CATALOG['files'][k]['size_bytes'] for k in SHARED)
    for page in (
        (ROOT / 'launcher/static/index.html').read_text(encoding='utf-8'),
        (ROOT / 'scripts/selfism-section.html').read_text(encoding='utf-8'),
        html,
    ):
        card = page.split('sf-card-qwen21-swap')[1].split('</article>')[0]
        assert '<h2>Qwen Image 2.1 Swap (Person + Outfit)</h2>' in card
        assert f'data-sf-action="{PROFILE}"' in card
        assert '8 modela' in card and '2.20.0' in card and 'nekomercijalna' in card
        assert f'{total / 1e9:.1f}'.replace('.', ',') in card
        assert f'{shared / 1e9:.1f}'.replace('.', ',') in card
        for profile in WORKFLOWS:
            assert f'/api/selfism/workflow/{profile}' in card
        assert 'sf-card-qwen21-edit' in page
    with TestClient(m.app) as client:
        for profile, (_, filename) in WORKFLOWS.items():
            downloaded = client.get(f'/api/selfism/workflow/{profile}')
            assert downloaded.status_code == 200
            assert downloaded.content == (ROOT / 'selfism_workflows' / f'{profile}.json').read_bytes()
            assert filename in downloaded.headers['content-disposition']

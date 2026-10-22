"""Hearmeman Wan Animate V5 (SAM3) and SCAIL 2 God Mode V2 cards."""
import hashlib
import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from test_full import CATALOG, ROOT, m, start_install

WAN = 'hearmeman_wan_v5'
SCAIL = 'hearmeman_scail2_v2'
WAN_SHA = 'd29ba4b87a15d29ad5fff11b4fb8865616c8a7e60a2d565dc30238d2d61b3114'
SCAIL_SHA = '8977dbd059cefd7ec244abd1c9e39a768afbb37faa821d17d72dc82455d420a2'
PLACEHOLDER = 'Your_Character_LoRA_Here.safetensors'
CNR = {
    'comfyui-kjnodes': 'ComfyUI-KJNodes',
    'comfyui_controlnet_aux': 'comfyui_controlnet_aux',
    'ComfyUI-WanAnimatePreprocess': 'ComfyUI-WanAnimatePreprocess',
    'ComfyUI-WanVideoWrapper': 'ComfyUI-WanVideoWrapper',
    'rgthree-comfy': 'rgthree-comfy',
    'hearmemanai-upscale': 'hearmemanai-upscale',
    'openrouter_node': 'openrouter_node',
    'comfyui-videohelpersuite': 'ComfyUI-VideoHelperSuite',
}
AUX = {
    'kijai/ComfyUI-KJNodes': 'ComfyUI-KJNodes',
    'Fannovel16/comfyui_controlnet_aux': 'comfyui_controlnet_aux',
    'kijai/ComfyUI-WanAnimatePreprocess': 'ComfyUI-WanAnimatePreprocess',
    'kijai/ComfyUI-WanVideoWrapper': 'ComfyUI-WanVideoWrapper',
    'rgthree/rgthree-comfy': 'rgthree-comfy',
    'Hearmeman24/ComfyUI-HearmemanAI-Upscale': 'hearmemanai-upscale',
    'Kosinkadink/ComfyUI-VideoHelperSuite': 'ComfyUI-VideoHelperSuite',
    'aining2022/ComfyUI_Swwan': 'ComfyUI-KJNodes',
}


def workflow_nodes(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    nodes = list(data['nodes'])
    for group in (data.get('definitions') or {}).get('subgraphs', []):
        nodes += group['nodes']
    return data, nodes


def widget_models(nodes):
    used = set()

    def walk(value):
        if isinstance(value, str) and value.lower().endswith(
            ('.safetensors', '.onnx', '.pth', '.bin', '.pt', '.ckpt')
        ):
            used.add(value.rsplit('/', 1)[-1])
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    for node in nodes:
        walk(node.get('widgets_values'))
        walk(node.get('widgets_values_named'))
    used.discard(PLACEHOLDER)
    return used


def packs_for(data, nodes):
    subgraphs = {group['id'] for group in (data.get('definitions') or {}).get('subgraphs', [])}
    packs = set()
    for node in nodes:
        if node['type'] in subgraphs:
            continue
        props = node.get('properties') or {}
        cnr = props.get('cnr_id')
        aux = props.get('aux_id')
        if cnr in CNR:
            packs.add(CNR[cnr])
        elif aux in AUX:
            packs.add(AUX[aux])
        elif cnr not in (None, 'comfy-core'):
            raise AssertionError(f'unmapped node {node["type"]} cnr={cnr} aux={aux}')
    return packs


def test_new_packs_are_pinned_and_baked_with_the_krast_set():
    nodes = {node['name']: node for node in CATALOG['nodes']}
    upscale = nodes['hearmemanai-upscale']
    router = nodes['openrouter_node']
    assert upscale['repo'] == 'https://github.com/Hearmeman24/ComfyUI-HearmemanAI-Upscale.git'
    assert upscale['ref'] == '3ca96fddf8ba07094895d5aecec2438c85bb41ba'
    assert upscale['install_requirements'] is False
    assert router['repo'] == 'https://github.com/gabe-init/ComfyUI-Openrouter_node.git'
    assert router['ref'] == 'df18925e8d19f98987fe865fc0ed7b66015eeb05'
    assert router['requirements_packages'] == ['requests', 'tiktoken', 'Pillow']
    bake = (ROOT / 'docker/bake_comfyui.sh').read_text(encoding='utf-8')
    assert 'hearmemanai-upscale' in bake and 'openrouter_node' in bake


def test_umt5_fp8_uses_the_filename_the_graphs_load():
    item = CATALOG['files']['umt5-xxl-fp8-scaled']
    assert Path(item['destination']).name == 'umt5_xxl_fp8_e4m3fn_scaled.safetensors'
    assert item['destination'] == 'models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors'
    assert item['sha256'] == 'c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68'
    assert item['size_bytes'] == 6735906897
    # The motion-control catalog keeps the hyphenated name for the same bytes.
    workflows = json.loads((ROOT / 'catalog/workflows.json').read_text(encoding='utf-8'))
    blob = json.dumps(workflows)
    assert 'umt5-xxl-encoder-fp8-e4m3fn-scaled.safetensors' in blob
    assert 'umt5_xxl_fp8_e4m3fn_scaled.safetensors' not in blob


def test_wan_and_scail_catalog_entries_have_private_r2_keys():
    expected = {
        WAN: 15,
        SCAIL: 10,
    }
    seen = set()
    for profile, count in expected.items():
        keys = CATALOG[profile + '_files']
        assert len(keys) == count
        for key in keys:
            item = CATALOG['files'][key]
            assert re.fullmatch(r'[0-9a-f]{64}', item['sha256']), key
            assert item['size_bytes'] > 0 and item['auth'] == 'none', key
            assert item['destination'].startswith('models/'), key
            r2_key = item['destination'].removeprefix('models/')
            assert r2_key and not r2_key.startswith('models/'), key
            seen.add(Path(item['destination']).name)
    for name in (
        'wan2.2_animate_14B_int8_convrot.safetensors',
        'umt5_xxl_fp8_e4m3fn_scaled.safetensors',
        'wan_2.1_vae.safetensors',
        'lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors',
        'taew2_1.pth',
        'rife426.pth',
        'wan2.1_14B_SCAIL_2_int8_convrot.safetensors',
        'wan2.1_SCAIL_2_DPO_lora_bf16.safetensors',
        'sam3.1_multiplex_fp16.safetensors',
    ):
        assert name in seen
    assert CATALOG['files']['taew2-1']['destination'] == 'models/vae_approx/taew2_1.pth'
    assert CATALOG['files']['rife426']['destination'] == 'models/frame_interpolation/rife426.pth'
    assert CATALOG[WAN + '_links'] == CATALOG['6gsc_links']
    assert SCAIL + '_links' not in CATALOG


def test_wan_workflow_models_and_nodes_match_the_card():
    path = ROOT / 'selfism_workflows/hearmeman_wan_v5.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == WAN_SHA
    data, nodes = workflow_nodes(path)
    installed = {Path(CATALOG['files'][key]['destination']).name for key in CATALOG[WAN + '_files']}
    used = widget_models(nodes)
    assert used <= installed
    assert installed - used == {'vitpose_h_wholebody_data.bin'}
    assert packs_for(data, nodes) <= set(CATALOG[WAN + '_nodes'])
    assert 'OpenRouterNode' in {node['type'] for node in nodes}
    assert 'DWPreprocessor' in {node['type'] for node in nodes}
    attention = [
        node['widgets_values']
        for node in nodes
        if node['type'] == 'DiffusionModelLoaderKJ'
    ]
    assert attention and 'sageattn_qk_int8_pv_fp8_cuda++' in attention[0]


def test_scail_workflow_models_and_nodes_match_the_card():
    path = ROOT / 'selfism_workflows/hearmeman_scail2_v2.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SCAIL_SHA
    data, nodes = workflow_nodes(path)
    installed = {Path(CATALOG['files'][key]['destination']).name for key in CATALOG[SCAIL + '_files']}
    assert widget_models(nodes) == installed
    types = {node['type'] for node in nodes}
    assert 'DWPreprocessor' not in types and 'PoseAndFaceDetection' not in types
    assert 'WanSCAILToVideo' in types and 'SCAIL2ColoredMask' in types
    assert packs_for(data, nodes) <= set(CATALOG[SCAIL + '_nodes'])
    assert 'comfyui_controlnet_aux' not in CATALOG[SCAIL + '_nodes']
    assert 'ComfyUI-WanVideoWrapper' not in CATALOG[SCAIL + '_nodes']


def test_wan_and_scail_install_payloads_cards_and_downloads(monkeypatch):
    pages = [
        (ROOT / 'launcher/static/index.html').read_text(encoding='utf-8'),
        (ROOT / 'scripts/selfism-section.html').read_text(encoding='utf-8'),
    ]
    cases = (
        (WAN, 'Hearmeman Wan Animate V5 (SAM3)', 'Hearmeman_WanAnimate_V5_SAM3.json', 'sf-card-hearmeman-wan', 15),
        (SCAIL, 'Hearmeman SCAIL 2 God Mode V2', 'Hearmeman_SCAIL2_GodMode_V2.json', 'sf-card-hearmeman-scail', 10),
    )
    for profile, title, filename, css, count in cases:
        response, captured, html = start_install(monkeypatch, {'profile': profile, 'precision': 'int8'})
        assert response.status_code == 200, profile
        assert captured['selfism_profile'] == profile
        assert captured['precision'] == 'fp8'
        assert captured['selfism_repair'] is False
        assert captured['selfism_repair_paths_only'] is False
        assert [item['id'] for item in captured['files']] == CATALOG[profile + '_files']
        assert {node['name'] for node in captured['custom_nodes']} == set(CATALOG[profile + '_nodes'])
        if profile == WAN:
            assert captured['model_links'] == CATALOG[WAN + '_links']
            sources = {item['destination'] for item in captured['files']}
            assert {link['source'] for link in captured['model_links']} <= sources
        else:
            assert captured['model_links'] == []
        total = sum(CATALOG['files'][key]['size_bytes'] for key in CATALOG[profile + '_files'])
        size = f'{total / 1e9:.1f}'.replace('.', ',')
        for page in pages + [html]:
            card = page.split(css)[1].split('</article>')[0]
            assert f'<h2>{title}</h2>' in card
            assert f'data-sf-action="{profile}"' in card
            assert 'OpenRouter' in card and 'SageAttention' in card and 'sdpa' in card
            assert '48 GB' in card
            assert size in card
            assert f'{count} modela' in card
        with TestClient(m.app) as client:
            downloaded = client.get(f'/api/selfism/workflow/{profile}')
        shipped = (ROOT / 'selfism_workflows' / f'{profile}.json').read_bytes()
        assert downloaded.status_code == 200
        assert downloaded.content == shipped
        assert filename in downloaded.headers['content-disposition']

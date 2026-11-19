"""Matrix Dataset card: the published MATRIX Qwen Image 2.1 Dataset graph (32 images)."""
import asyncio
import hashlib
import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from test_full import CATALOG, ROOT, m, start_install
from test_hearmeman_video import widget_models, workflow_nodes
import launcher.selfism as s

PROFILE = 'matrix_qwen21_dataset'
WORKFLOW = ROOT / 'selfism_workflows/matrix_qwen21_dataset.json'
ORIGINAL = ROOT / 'selfism_workflows/sources/matrix_qwen21_dataset_original.json'
ORIGINAL_SHA = '05f2804cb6e568013dd146724abe3792b3ab7e34ad0ed4180c420dc71b909640'
OUT = 'Matrix_Qwen21_Dataset.json'
PRUNA_ORIGINAL = 'p_qwen_image_2.1_8step_v0.1.safetensors'
PRUNA_T8 = 'p_qwen_image_2.1_8step_v0.1-comfyui-T8.safetensors'
FILES = {
    'qwen21-int8': ('models/diffusion_models/qwen_image_2.1_int8_convrot.safetensors', 7256783064),
    'qwen3vl-8b-int8': ('models/text_encoders/qwen3vl_8b_int8_convrot.safetensors', 9350798360),
    'qwen21-vae': ('models/vae/qwen_image_2.1_vae_bf16.safetensors', 675509688),
    'bfs-head-qwen21': ('models/loras/bfs_head_v1.1_qwen_2.1.safetensors', 260096144),
    'pruna-qwen21-8step-t8': ('models/loras/' + PRUNA_T8, 335633936),
}
NODES = ['rgthree-comfy', 'MATRIX-LAB-Nodes', 'ComfyUI-Fantastic-MiniMaxH3-PromptBuilder']
REFS = {
    'MATRIX-LAB-Nodes': ('https://github.com/JsonMatrixLab/MATRIX-LAB-Nodes.git',
                         'f71fb38dea39f6052ff4e883c18fecc7ec07d199'),
    'ComfyUI-Fantastic-MiniMaxH3-PromptBuilder': (
        'https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder.git',
        '23038f5050acdcbb8e938bfd7c87a18d2cc84aab'),
    'rgthree-comfy': ('https://github.com/rgthree/rgthree-comfy.git',
                      '449c58fcdd612f7733e54c51f6758ead63fa180b'),
}
PACK_TYPES = {
    'Fast Groups Bypasser (rgthree)': 'rgthree-comfy', 'Fast Actions Button (rgthree)': 'rgthree-comfy',
    'Power Lora Loader (rgthree)': 'rgthree-comfy',
    'MATRIX_QwenImage21Resolution': 'MATRIX-LAB-Nodes',
    'MiniMaxH3MediaLoader': 'ComfyUI-Fantastic-MiniMaxH3-PromptBuilder',
    'MiniMaxH3ReferenceSplitter': 'ComfyUI-Fantastic-MiniMaxH3-PromptBuilder',
}
# Everything else is ComfyUI core in the baked v0.38.1 (MarkdownNote is a frontend note).
CORE = {'UNETLoader', 'CLIPLoader', 'VAELoader', 'EmptyLatentImage', 'PrimitiveInt', 'TextEncodeQwenImage21',
        'KSampler', 'VAEDecode', 'SaveImage', 'ComfyMathExpression', 'MarkdownNote'}


def _loras(data):
    return {row['lora']: (row['on'], row['strength']) for node in data['nodes']
            if node['type'] == 'Power Lora Loader (rgthree)'
            for row in node['widgets_values'] if isinstance(row, dict) and 'lora' in row}


def test_catalog_pins_every_model_the_workflow_loads():
    assert CATALOG[PROFILE + '_files'] == list(FILES)
    for key, (dest, size) in FILES.items():
        item = CATALOG['files'][key]
        assert item['id'] == key and item['destination'] == dest and item['size_bytes'] == size
        assert re.fullmatch(r'[0-9a-f]{64}', item['sha256']) and item['auth'] == 'none'
        assert re.search(r'^https://huggingface\.co/.+/resolve/[0-9a-f]{40}/', item['url'])
    # No new downloads: everything is shared with the Qwen Image 2.1 Swap card.
    assert set(FILES) <= set(CATALOG['qwen21_swap_files'])
    # The T8 conversion carries the original PrunaAI weights (rank 64, alpha 128) plus alpha tensors.
    assert CATALOG['files']['pruna-qwen21-8step-t8']['sha256'] == (
        '38a963b33ebd16d5eb9edfdf0254a3e7f55e3bda2d46bca55e5990e31ec43c83')


def test_node_packs_are_pinned_and_add_no_pip_work():
    assert CATALOG[PROFILE + '_nodes'] == NODES
    by_name = {node['name']: node for node in CATALOG['nodes']}
    for name, (repo, ref) in REFS.items():
        assert by_name[name]['repo'] == repo and by_name[name]['ref'] == ref
    for name in ('MATRIX-LAB-Nodes', 'ComfyUI-Fantastic-MiniMaxH3-PromptBuilder'):
        # torch/numpy/Pillow/scipy/aiohttp/av all come with ComfyUI v0.38.1; never reinstall torch.
        assert by_name[name]['requirements_packages'] == []
        assert sum(1 for node in CATALOG['nodes'] if node['name'] == name) == 1


def test_workflow_is_the_published_graph_with_only_the_pruna_rename():
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == ORIGINAL_SHA
    original = json.loads(ORIGINAL.read_text(encoding='utf-8'))
    data = json.loads(WORKFLOW.read_text(encoding='utf-8'))
    assert _loras(original)[PRUNA_ORIGINAL] == (False, 1.0)
    assert _loras(data) == {'bfs_head_v1.1_qwen_2.1.safetensors': (False, 0.9), PRUNA_T8: (False, 1.0)}
    restored = WORKFLOW.read_text(encoding='utf-8').replace(PRUNA_T8, PRUNA_ORIGINAL)
    assert json.loads(restored) == original
    assert len(data['nodes']) == 202 and not (data.get('definitions') or {}).get('subgraphs')


def test_every_node_and_model_is_provided_and_on_gpu():
    data, nodes = workflow_nodes(WORKFLOW)
    used_packs = set()
    for node in nodes:
        if node['type'] in PACK_TYPES:
            used_packs.add(PACK_TYPES[node['type']])
        else:
            assert node['type'] in CORE, node['type']
        for value in node.get('widgets_values') or []:
            assert value != 'cpu', node['type']
    assert used_packs == set(NODES)
    assert widget_models(nodes) == {Path(dest).name for dest, _ in FILES.values()}
    counts = {}
    for node in nodes:
        counts[node['type']] = counts.get(node['type'], 0) + 1
    assert counts['KSampler'] == counts['SaveImage'] == counts['TextEncodeQwenImage21'] == 32
    for node in nodes:
        if node['type'] == 'KSampler':
            assert node['widgets_values'][2:6] == [25, 1.0, 'euler', 'beta']
        if node['type'] == 'MiniMaxH3MediaLoader':
            # No LoadImage nodes: the user drags references into each section's loader.
            assert node['widgets_values'][0] == '[]'
    assert 'LoadImage' not in counts


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
        card = page.split('sf-card-matrix-qwen21-dataset"')[1].split('</article>')[0]
        assert '<h2>Matrix Dataset</h2>' in card
        assert f'data-sf-action="{PROFILE}"' in card and '5 modela' in card and '3 paketa' in card
        assert f'{total / 1e9:.1f}'.replace('.', ',') in card and '24 GB' in card
        assert f'/api/selfism/workflow/{PROFILE}' in card
        for other in ('sf-card-qwen21-dataset-generator', 'sf-card-qwen21-swap', 'sf-card-qwen21-auto-carousel',
                      'sf-card-krast-org-pose-millie'):
            assert other in page
    with TestClient(m.app) as client:
        downloaded = client.get(f'/api/selfism/workflow/{PROFILE}')
        assert downloaded.status_code == 200 and downloaded.content == WORKFLOW.read_bytes()
        assert OUT in downloaded.headers['content-disposition']
        other = client.get('/api/selfism/workflow/qwen21_dataset_generator')
        assert other.content == (ROOT / 'selfism_workflows/qwen21_dataset_generator.json').read_bytes()


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

import importlib
import json
import subprocess
from pathlib import Path

import pytest

int8 = importlib.import_module('launcher.selfism_int8')
ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT/'selfism_workflows').glob('*.json'))


def sh(cwd, *args):
    return subprocess.run(['git', '-C', str(cwd), '-c', 'user.email=t@t', '-c', 'user.name=t', *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def upstream_and_clone(tmp_path, monkeypatch):
    up = tmp_path/'up'; up.mkdir()
    sh(up, 'init', '-q', '-b', 'master')
    (up/'comfy').mkdir()
    (up/'requirements.txt').write_text('')
    (up/'comfy/quant_ops.py').write_text('# old\n')
    sh(up, 'add', '.'); sh(up, 'commit', '-qm', 'old')
    clone = tmp_path/'comfy'
    sh(tmp_path, 'clone', '-q', str(up), 'comfy')
    (up/'comfy/quant_ops.py').write_text('QUANT_ALGOS["int8_tensorwise"] = {}\n')
    sh(up, 'commit', '-qam', 'int8'); sh(up, 'tag', 'v9.9.9')
    monkeypatch.setattr(int8, 'UPSTREAM', str(up))
    monkeypatch.setattr(int8, 'TARGET_TAG', 'v9.9.9')
    monkeypatch.setattr(int8, 'TARGET_SHA', sh(up, 'rev-parse', 'v9.9.9^{commit}'))
    return up, clone


def run_main(monkeypatch, clone, *extra, kitchen='0.2.36', install=None):
    calls = []
    monkeypatch.setattr(int8, 'installed', lambda n: {'comfy-kitchen': kitchen, 'torch': '2.9.0'}.get(n, ''))
    monkeypatch.setattr(int8, 'fresh_version', lambda n: {'comfy-kitchen': '0.2.36', 'torch': '2.9.0'}.get(n, ''))
    def fake_install(comfy, constraints):
        calls.append(Path(constraints).read_text())
        if install: install()
    monkeypatch.setattr(int8, 'install_requirements', fake_install)
    monkeypatch.setattr('sys.argv', ['x', '--comfy-dir', str(clone), *extra])
    int8.main()
    return calls


def test_version_tuple():
    assert int8.version_tuple('0.2.36') >= int8.MIN_KITCHEN
    assert int8.version_tuple('0.2.9') < int8.MIN_KITCHEN
    assert int8.version_tuple('0.2.16+cu128') == (0, 2, 16)


def test_upgrade_is_recorded_constrained_and_idempotent(upstream_and_clone, monkeypatch):
    up, clone = upstream_and_clone
    old_head = sh(clone, 'rev-parse', 'HEAD')
    calls = run_main(monkeypatch, clone)
    assert len(calls) == 1 and 'torch==2.9.0' in calls[0]
    assert 'int8_tensorwise' in (clone/'comfy/quant_ops.py').read_text()
    assert json.loads((clone/'.selfism-int8-backup.json').read_text())['original']['head'] == old_head
    assert run_main(monkeypatch, clone) == []


def test_old_kitchen_with_new_source_only_reinstalls_requirements(upstream_and_clone, monkeypatch):
    up, clone = upstream_and_clone
    run_main(monkeypatch, clone)
    head = sh(clone, 'rev-parse', 'HEAD')
    assert len(run_main(monkeypatch, clone, kitchen='0.2.10')) == 1
    assert sh(clone, 'rev-parse', 'HEAD') == head


def test_dirty_checkout_is_refused_unless_stash_requested(upstream_and_clone, monkeypatch):
    up, clone = upstream_and_clone
    (clone/'comfy/quant_ops.py').write_text('# local edit\n')
    with pytest.raises(SystemExit):
        run_main(monkeypatch, clone)
    assert (clone/'comfy/quant_ops.py').read_text() == '# local edit\n'
    run_main(monkeypatch, clone, '--stash')
    assert 'selfism-int8' in sh(clone, 'stash', 'list')


def test_unexpected_tag_commit_is_refused(upstream_and_clone, monkeypatch):
    up, clone = upstream_and_clone
    monkeypatch.setattr(int8, 'TARGET_SHA', '0'*40)
    with pytest.raises(SystemExit):
        run_main(monkeypatch, clone)


def declared_outputs(workflow):
    scopes = [(workflow['nodes'], workflow.get('links', []))]
    scopes += [(g['nodes'], g.get('links', [])) for g in workflow.get('definitions', {}).get('subgraphs', [])]
    return scopes


@pytest.mark.parametrize('path', WORKFLOWS, ids=lambda p: p.name)
def test_every_link_origin_slot_exists_on_its_node(path):
    workflow = json.loads(path.read_text(encoding='utf-8'))
    for nodes, links in declared_outputs(workflow):
        by_id = {n['id']: n for n in nodes}
        for link in links:
            origin, slot = (link['origin_id'], link['origin_slot']) if isinstance(link, dict) else (link[1], link[2])
            node = by_id.get(origin)
            if node is not None and 'outputs' in node:
                assert slot < len(node['outputs']), (path.name, link, node['type'])


def test_full_resolution_node_uses_v3_layout():
    text = (ROOT/'selfism_workflows/full.json').read_text(encoding='utf-8')
    workflow = json.loads(text)
    node = next(n for n in workflow['nodes'] if n['id'] == 1021)
    assert [o['name'] for o in node['outputs']] == ['width', 'height', 'image', 'mask', 'latent', 'info']
    assert node['outputs'][4]['links'] == [1990]
    assert next(l for l in workflow['links'] if l[0] == 1990)[2] == 4
    assert json.dumps(workflow, ensure_ascii=False, indent=2) == text


def test_int8_repair_is_wired_before_the_runtime_repair():
    source = (ROOT/'launcher/selfism.py').read_text(encoding='utf-8')
    assert "INT8_PROFILES = ('simple','aio','full'" in source
    assert source.index('selfism_int8.py') < source.index('super()._install_workflow') < source.index('selfism_runtime.py')

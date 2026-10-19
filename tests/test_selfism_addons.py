import json
import re
from pathlib import Path

import pytest

from tests.test_full import CATALOG, ROOT, s, start_install

ADDONS = ['outfit', 'skin-contrast', 'llm-qwen35-9b', 'mmproj-qwen35-9b']


def test_addon_and_lut_metadata():
    files = CATALOG['files']
    assert CATALOG['selfism_addon_files'] == ADDONS
    assert CATALOG['selfism_simple_addon_files'] == ['llm-qwen35-9b', 'mmproj-qwen35-9b']
    for k in ADDONS + CATALOG['selfism_lut_files']:
        f = files[k]
        assert re.fullmatch(r'[0-9a-f]{64}', f['sha256']) and f['size_bytes'] > 0, k
    assert len(CATALOG['selfism_lut_files']) == 53
    assert files['bf16']['size_bytes'] == 25640190264
    dests = {files[k]['destination'] for k in CATALOG['selfism_lut_files']}
    assert len(dests) == 53 and all(d.startswith('models/luts/') and d.endswith('.cube') for d in dests)
    assert all(files[k]['r2_required'] for k in CATALOG['selfism_lut_files'])
    links = CATALOG['selfism_lut_links']
    assert {l['source'] for l in links} == dests
    for l in links:
        assert l['destination'] == 'custom_nodes/ComfyUI-OlmLUT/luts/' + Path(l['source']).name
    assert 'models/luts/Smartphone_13.cube' in dests
    # The Full + Refine and 6gsc cards are untouched.
    for profile in ('full_refine', '6gsc'):
        assert not set(ADDONS + CATALOG['selfism_lut_files']) & set(CATALOG[profile + '_files'])


@pytest.mark.parametrize('profile,r2', [('aio', True), ('aio', False), ('full', True), ('full', False), ('simple', True)])
def test_selfism_cards_get_addons_and_luts(monkeypatch, profile, r2):
    monkeypatch.setattr(s, 'r2_client', lambda: (object() if r2 else None, 'b', ''))
    body = {'profile': profile} if profile == 'full' else {'profile': profile, 'precision': 'fp8'}
    r, captured, _ = start_install(monkeypatch, body)
    assert r.status_code == 200
    ids = [f['id'] for f in captured['files']]
    assert len(ids) == len(set(ids))
    luts = set(CATALOG['selfism_lut_files'])
    if profile == 'simple':
        assert {'llm-qwen35-9b', 'mmproj-qwen35-9b'} <= set(ids) and not luts & set(ids)
        assert not {'outfit', 'skin-contrast'} & set(ids) and captured['model_links'] == []
        return
    assert set(ADDONS) <= set(ids)
    lut_links = [l for l in captured['model_links'] if l['destination'].startswith('custom_nodes/ComfyUI-OlmLUT/')]
    if r2:
        assert luts <= set(ids) and lut_links == CATALOG['selfism_lut_links']
    else:
        assert not luts & set(ids) and lut_links == []
    assert {l['source'] for l in captured['model_links']} <= {f['destination'] for f in captured['files']}


def test_full_refine_unchanged(monkeypatch):
    monkeypatch.setattr(s, 'r2_client', lambda: (object(), 'b', ''))
    r, captured, _ = start_install(monkeypatch, {'profile': 'full_refine'})
    ids = {f['id'] for f in captured['files']}
    assert not (set(ADDONS) | set(CATALOG['selfism_lut_files'])) & ids
    assert captured['model_links'] == CATALOG['full_links']

"""Full + Refine installation contract and graph regressions; no GPU required."""
import asyncio
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from test_full import ROOT, CATALOG, all_nodes, start_install, m, s

WORKFLOW = json.loads((ROOT/'selfism_workflows/full_refine.json').read_text(encoding='utf-8'))
NODES = {n['id']: n for n in WORKFLOW['nodes']}


def test_graph_links_are_bidirectional_and_refine_off_cuts_sampler_path():
    links = {l[0]: l for l in WORKFLOW['links']}
    for i, a, ao, b, bi, typ in links.values():
        assert NODES[b]['inputs'][bi]['link'] == i
        assert i in NODES[a]['outputs'][ao]['links']
    for node in NODES.values():
        for inp in node.get('inputs', []):
            if inp.get('link') is not None: assert inp['link'] in links
    switch = next(n for n in NODES.values() if n.get('title','').startswith('REFINE ON / OFF'))
    assert switch['type'] == 'LazySwitchKJ' and switch['widgets_values'] == [True]
    assert links[switch['inputs'][0]['link']][1] == 3119
    assert links[switch['inputs'][1]['link']][1] == 18
    # Model real executable data edges, excluding bypassed processing and optional UI connections.
    def ancestors(node_id, enabled, seen):
        if node_id in seen: return
        seen.add(node_id)
        node = NODES[node_id]
        if node['type'].startswith('Fast '): return
        inputs=node.get('inputs', [])
        if node_id == switch['id']: inputs=[inputs[1 if enabled else 0]]
        elif node.get('mode') == 4:
            # All bypassed paths used by the output chain pass through IMAGE.
            inputs=[i for i in inputs if i['type']=='IMAGE']
        for inp in inputs:
            if inp.get('link') is not None: ancestors(links[inp['link']][1], enabled, seen)
    for enabled in (False, True):
        seen=set()
        for n in NODES.values():
            if n['type'] in ('SaveImage','PreviewImage') and n.get('mode',0)==0:
                ancestors(n['id'],enabled,seen)
        assert (3118 in seen) == enabled
        assert (3120 in seen) == enabled
        assert 3117 in seen


def test_sampler_recipe_replaces_second_pass_and_leaves_base_recipe():
    sampler=NODES[3118]
    assert sampler['type']=='KSampler'
    assert sampler['widgets_values']==[40,'fixed',6,1.0,'euler','simple',0.18]
    assert [i['name'] for i in sampler['inputs']]==['model','positive','negative','latent_image']
    assert NODES[3120]['widgets_values']==['4xNMKDSuperscale_4xNMKDSuperscale.pt','nearest-exact',0.25]
    old=json.loads((ROOT/'selfism_workflows/full.json').read_text(encoding='utf-8'))
    old_first=next(n for n in old['nodes'] if n['id']==3117)
    assert NODES[3117]['widgets_values'][3:]==old_first['widgets_values'][3:]
    assert NODES[14]['widgets_values'][0]>=0
    assert NODES[866]['widgets_values'][10]=='fixed'
    assert NODES[866]['widgets_values'][15]==(ROOT/'selfism_workflows/presets/Recreate_SFW_prefix.txt').read_text(encoding='utf-8')


def test_all_eight_6gsc_loras_are_retained_off_with_latest_mystic():
    expected={'MysticXXX_KREA2_v3.safetensors':0.7,'famegrid_spicy.safetensors':0.45,
        'krea2-bloomgirls-realism-step00004000.safetensors':0.3,'Yumi_000002250.safetensors':0.65,
        'snofs_krea_v1_4.safetensors':1,'krea-smartphone-photo-slider.safetensors':0.25,
        'Photografic Scene Coherence V1.5.safetensors':0.3,'krea2_turbo_openpose_controlnet.safetensors':0.7}
    rows={r['lora']:r for r in NODES[70]['widgets_values'] if isinstance(r,dict) and 'lora' in r}
    for name,strength in expected.items():
        assert rows[name]['on'] is False and rows[name]['strength']==strength
    assert rows['millie_000002750.safetensors']['on'] is True
    names={Path(CATALOG['files'][k]['destination']).name for k in CATALOG['full_refine_files']}
    assert set(rows)-names=={'Yumi_000002250.safetensors'}
    for n in all_nodes(WORKFLOW):
        if n['type'] in ('UpscaleModelLoader','SAMLoader','Krea2ControlLoRALoader','CLIPLoader'):
            assert Path(n['widgets_values'][0]).name in names


@pytest.mark.parametrize('precision', [None,'int8','fp8'])
def test_new_card_installs_full_dependencies_without_civitai_lookup(monkeypatch, precision):
    body={'profile':'full_refine'}
    if precision: body['precision']=precision
    r, captured, html=start_install(monkeypatch,body)
    assert r.status_code==200
    assert captured['precision']==(precision or 'int8')
    assert {f['id'] for f in captured['files']}==set(CATALOG['full_refine_files'])|{precision or 'int8'}
    assert {n['name'] for n in captured['custom_nodes']}==set(CATALOG['full_nodes'])
    assert captured['model_links']==CATALOG['full_links']
    assert captured['selfism_repair'] and 'full_refine' in s.INT8_PROFILES
    assert 'data-sf-action="full_refine"' in html and 'sf-refine-precision' in html
    assert 'data-sf-action="full"' in html
    with TestClient(m.app) as client:
        assert client.get('/api/selfism/workflow/full_refine').json()==WORKFLOW


def test_new_card_rejects_bf16(monkeypatch):
    r,captured,_=start_install(monkeypatch,{'profile':'full_refine','precision':'bf16'})
    assert r.status_code==400 and not captured


def test_saving_refine_never_overwrites_existing_full_or_user_edits(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'COMFYUI_DIR',tmp_path)
    monkeypatch.setattr(m,'COMFYUI_VENV',tmp_path/'.venv')
    python=m.COMFYUI_VENV/'bin/python';python.parent.mkdir(parents=True);python.touch()
    ctrl=type(m.selfism_controller)()
    async def ready(*args): pass
    async def run(*args,**kwargs): return 0,'torch==2.10.0'
    async def sources(host,files): return files,[]
    monkeypatch.setattr(ctrl,'_wait_for_comfyui',ready)
    monkeypatch.setattr(ctrl,'_run_process',run)
    monkeypatch.setattr(m.JobController,'_install_workflow',ready)
    monkeypatch.setattr(s,'resolve_sources',sources)
    folder=tmp_path/'user/default/workflows/Selfism';folder.mkdir(parents=True)
    original=folder/'Selfism_FULL_int8_recreate_v1.json';original.write_text('original user edits')
    wf={'selfism_profile':'full_refine','precision':'fp8','files':[{'id':'x'}]}
    asyncio.run(ctrl._install_workflow(wf))
    saved=folder/'Selfism_FULL_fp8_Refine_v1.json'
    data=json.loads(saved.read_text(encoding='utf-8'))
    assert next(n for n in data['nodes'] if n['id']==881)['widgets_values'][0]==Path(CATALOG['files']['fp8']['destination']).name
    assert (tmp_path/'models/LLM/prompts/Recreate_SFW_prefix.txt').exists()
    saved.write_text('refine user edits')
    asyncio.run(ctrl._install_workflow(wf))
    assert saved.read_text()=='refine user edits' and original.read_text()=='original user edits'


def test_private_only_models_fail_before_fallback_and_use_verified_r2(monkeypatch):
    files=[copy.deepcopy(CATALOG['files']['6gsc-smartphone'])]
    monkeypatch.setattr(s,'r2_client',lambda:(None,'','not connected'))
    with pytest.raises(RuntimeError,match='Connect private R2'):
        asyncio.run(s.resolve_sources(Mock(),files))
    private=copy.deepcopy(files[0]);private['url']='https://example.test/signed'
    monkeypatch.setattr(s,'select_private',lambda *a:({0:private},[]))
    selected,_=asyncio.run(s.resolve_sources(Mock(),files))
    assert selected[0]['_selfism_source']=='R2' and selected[0]['url']==private['url']

"""The 6gsc card packages the original recipe with only documented portability edits."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_full import ROOT, CATALOG, start_install, m, s

SOURCE=ROOT/'selfism_workflows/sources/6gsc_original.json'
ORIGINAL=json.loads(SOURCE.read_text(encoding='utf-8'))
WORKFLOW=json.loads((ROOT/'selfism_workflows/6gsc.json').read_text(encoding='utf-8'))


def test_original_recipe_is_preserved_except_unused_loader_and_newer_disabled_mystic():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='16289367cb3a029b10bdd960887d05bfaab0f2c81eb34437da171c966f99c4a2'
    expected=copy.deepcopy(ORIGINAL)
    assert not any(1761 in (l[1],l[3]) for l in expected['links'])
    expected['nodes']=[n for n in expected['nodes'] if n['id']!=1761]
    lora=next(n for n in expected['nodes'] if n['id']==1633)
    for value in lora['widgets_values']+list(lora['widgets_values_named'].values()):
        if isinstance(value,dict) and value.get('lora')=='MysticXXX_KREA2_v1.safetensors':
            assert value['on'] is False
            value['lora']='MysticXXX_KREA2_v3.safetensors'
    expected['id']=WORKFLOW['id']
    assert expected==WORKFLOW


def test_graph_connections_and_model_coverage():
    nodes={n['id']:n for n in WORKFLOW['nodes']}
    for lid,a,ao,b,bi,typ in WORKFLOW['links']:
        assert nodes[b]['inputs'][bi]['link']==lid
        assert lid in nodes[a]['outputs'][ao]['links']
    names={Path(CATALOG['files'][k]['destination']).name for k in CATALOG['6gsc_files']}
    for n in nodes.values():
        if n['type'] in ('UNETLoader','CLIPLoader','VAELoader','UpscaleModelLoader','SAMLoader','UltralyticsDetectorProvider'):
            assert Path(n['widgets_values'][0]).name in names
        if n['type']=='DWPreprocessor':
            assert set(n['widgets_values'][4:6]) <= names
        if n['type']=='Power Lora Loader (rgthree)':
            for row in n['widgets_values']:
                if isinstance(row,dict) and 'lora' in row:
                    if row['lora']=='Yumi_000002250.safetensors': assert row['on'] is False
                    else: assert row['lora'] in names
    assert not {'llm','mmproj','int8','fp8','depth'} & set(CATALOG['6gsc_files'])
    assert len(CATALOG['6gsc_files'])==15
    assert len(set(CATALOG['6gsc_nodes']))==7
    ostris=next(n for n in CATALOG['nodes'] if n['name']=='comfyui-krea2-ostris-edit')
    assert ostris['repo']=='https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit.git'
    assert ostris['ref']=='7756566160c4a1b24bb1bd9f0ff3ced1a83d7547'


@pytest.mark.parametrize('precision',['int8','fp8','bf16'])
def test_install_is_complete_and_independent_of_selfora_precision(monkeypatch,precision):
    r,captured,html=start_install(monkeypatch,{'profile':'6gsc','precision':precision})
    assert r.status_code==200
    assert captured['precision']=='fp8' and captured['selfism_profile']=='6gsc'
    assert {f['id'] for f in captured['files']}==set(CATALOG['6gsc_files'])
    assert {n['name'] for n in captured['custom_nodes']}==set(CATALOG['6gsc_nodes'])
    assert captured['model_links']==CATALOG['6gsc_links'] and captured['selfism_repair']
    for item in captured['model_links']:
        assert item['source'] in {f['destination'] for f in captured['files']}
        assert item['destination']==item['source'].replace('models/controlnet_aux/','custom_nodes/comfyui_controlnet_aux/ckpts/')
    assert '<h2>6gsc</h2>' in html and 'data-sf-action="6gsc"' in html
    assert html.index('<h2>10sorLabs Reference + Depth</h2>') < html.index('<h2>6gsc</h2>') < html.index('<h2>Kompletan workflow (Selfora Full)</h2>')
    assert 'data-sf-action="full_refine"' in html
    with TestClient(m.app) as client:
        response=client.get('/api/selfism/workflow/6gsc')
        assert response.json()==WORKFLOW
        assert '6gsc_dashboard_v1.json' in response.headers['content-disposition']


def test_installer_saves_6gsc_without_touching_other_workflows_or_user_edits(monkeypatch,tmp_path):
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
    full=folder/'Selfism_FULL_int8_Refine_v1.json';full.write_text('existing Full + Refine')
    wf={'selfism_profile':'6gsc','precision':'fp8','files':[{'id':'x'}],'selfism_repair':True}
    asyncio.run(ctrl._install_workflow(wf))
    path=folder/'6gsc_dashboard_v1.json'
    assert path.read_bytes()==(ROOT/'selfism_workflows/6gsc.json').read_bytes()
    path.write_text('user edit')
    asyncio.run(ctrl._install_workflow(wf))
    assert path.read_text()=='user edit' and full.read_text()=='existing Full + Refine'

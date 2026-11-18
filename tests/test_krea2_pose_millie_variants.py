"""RawGirl and KRAST v2 variants of the Millie Pose Recreate card: only the model and sampler differ."""
import asyncio
import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_full import ROOT, CATALOG, start_install, m, s

BASE=json.loads((ROOT/'selfism_workflows/krea2_pose_recreate_millie.json').read_text(encoding='utf-8'))
VARIANTS={
    'krea2_pose_millie_rawgirl':('krea2_pose_recreate_millie_rawgirl.json','Krea2_PoseRecreate_Millie_RawGirl_v1.json',
                                 'rawgirl-krea2-v20-fp8','rawgirlKrea2_v20Fp8.safetensors','euler','beta','Millie Pose Recreate — RawGirl'),
    'krea2_pose_millie_krast':('krea2_pose_recreate_millie_krast.json','Krea2_PoseRecreate_Millie_KRAST_v1.json',
                               'krast-v20','krast_v20.safetensors','er_sde','beta','Millie Pose Recreate — KRAST v2'),
}


def _load(src): return json.loads((ROOT/'selfism_workflows'/src).read_text(encoding='utf-8'))


@pytest.mark.parametrize('profile',VARIANTS)
def test_only_model_and_sampler_differ_from_the_turbo_card(profile):
    src,_,_,unet,sampler,scheduler,_=VARIANTS[profile]
    wf=_load(src)
    expected=copy.deepcopy(BASE); expected['id']=wf['id']
    n={x['id']:x for x in expected['nodes']}
    n[1689]['widgets_values']=[unet,'default']
    for i in (1634,1790):
        n[i]['widgets_values'][4]=sampler; n[i]['widgets_values'][5]=scheduler
    got=copy.deepcopy(wf)
    for x in got['nodes']+expected['nodes']:
        if x['id'] in (1689,1634,1790): x.pop('title')
    assert got==expected and wf['id']!=BASE['id']
    k={x['id']:x for x in wf['nodes']}
    assert k[1634]['widgets_values'][2:4]==[8,1] and k[1634]['widgets_values'][6]==0.85
    assert k[1790]['widgets_values'][2:4]==[8,1] and k[1790]['widgets_values'][6]==0.2


@pytest.mark.parametrize('profile',VARIANTS)
def test_catalog_swaps_only_the_diffusion_model(profile):
    _,_,unet_key,unet,_,_,_=VARIANTS[profile]
    files=CATALOG[profile+'_files']
    assert files==[unet_key if f=='krea-turbo' else f for f in CATALOG['krea2_pose_millie_files']]
    assert Path(CATALOG['files'][unet_key]['destination']).name==unet
    assert CATALOG[profile+'_nodes']==CATALOG['krea2_pose_millie_nodes']
    assert CATALOG[profile+'_links']==CATALOG['krea2_pose_millie_links']
    names={Path(CATALOG['files'][f]['destination']).name for f in files}
    for n in _load(VARIANTS[profile][0])['nodes']:
        if n['type'] in ('UNETLoader','CLIPLoader','VAELoader','SAMLoader','UltralyticsDetectorProvider','LoraLoaderModelOnly'):
            assert n['widgets_values'][0].split('/')[-1] in names
        if n['type']=='Power Lora Loader (rgthree)':
            assert all(r['lora'] in names for r in n['widgets_values'] if isinstance(r,dict) and 'lora' in r)


@pytest.mark.parametrize('profile',VARIANTS)
def test_install_card_and_download(monkeypatch,profile):
    src,out,_,_,_,_,h2=VARIANTS[profile]
    r,captured,html=start_install(monkeypatch,{'profile':profile,'precision':'int8'})
    assert r.status_code==200 and captured['precision']=='fp8' and captured['selfism_profile']==profile
    assert {f['id'] for f in captured['files']}==set(CATALOG[profile+'_files'])
    assert {n['name'] for n in captured['custom_nodes']}==set(CATALOG[profile+'_nodes'])
    assert captured['model_links']==CATALOG[profile+'_links']
    assert captured['selfism_repair'] and captured['selfism_repair_paths_only']
    assert f'data-sf-action="{profile}"' in html
    assert html.index('<h2>Millie Pose Recreate (Krea2 Ostris + DWPose)</h2>') < html.index(f'<h2>{h2}</h2>')
    with TestClient(m.app) as client:
        response=client.get('/api/selfism/workflow/'+profile)
        assert response.json()==_load(src) and out in response.headers['content-disposition']


@pytest.mark.parametrize('profile',VARIANTS)
def test_installer_saves_without_overwriting_user_edits(monkeypatch,tmp_path,profile):
    src,out,*_=VARIANTS[profile]
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
    turbo=folder/'Krea2_PoseRecreate_Millie_v1.json';turbo.write_text('existing turbo')
    wf={'selfism_profile':profile,'precision':'fp8','files':[{'id':'x'}],'selfism_repair':True,'selfism_repair_paths_only':True}
    asyncio.run(ctrl._install_workflow(wf))
    path=folder/out
    assert path.read_bytes()==(ROOT/'selfism_workflows'/src).read_bytes()
    path.write_text('user edit')
    asyncio.run(ctrl._install_workflow(wf))
    assert path.read_text()=='user edit' and turbo.read_text()=='existing turbo'

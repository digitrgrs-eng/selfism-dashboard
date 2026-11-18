"""Krast ORG Pose Recreate (Millie): the Krast ORG graph plus the Bella DWPose + Ostris Edit pose technique in stage 1."""
import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient
from test_full import ROOT, CATALOG, start_install, m, s

PROFILE='krast_org_pose_millie'
SRC='krast_org_pose_recreate_millie.json'
OUT='Krast_ORG_PoseRecreate_Millie_v1.json'


def _load(name): return json.loads((ROOT/'selfism_workflows'/name).read_text(encoding='utf-8'))


def _all_nodes(w): return list(w['nodes'])+[n for sg in w['definitions']['subgraphs'] for n in sg['nodes']]


def test_links_are_consistent():
    w=_load(SRC); N={n['id']:n for n in w['nodes']}; ids=set()
    for i,o,os_,t,ts,_ in w['links']:
        assert i not in ids; ids.add(i)
        assert i in N[o]['outputs'][os_]['links'] and N[t]['inputs'][ts]['link']==i
    for n in w['nodes']:
        for o in n.get('outputs',[]): assert set(o.get('links') or [])<=ids
        for x in n.get('inputs',[]):
            assert x.get('link') is None or x['link'] in ids
            if x.get('link') is None and not x.get('widget'): assert x.get('shape')==7, (n['id'],x['name'])
    assert w['last_node_id']>=max(N) and w['last_link_id']>=max(ids)


def test_krast_org_parts_are_kept():
    org=_load('krast_org.json'); w=_load(SRC)
    O={n['id']:n for n in org['nodes']}; N={n['id']:n for n in w['nodes']}
    keep=[1619,1622,1624,1626,1627,1628,1629,1630,1634,1635,1636,1639,13421,13422,13424,13438,13451,13452,13453,13454,13455,13456,13465,272,117]
    for i in keep:
        assert N[i]['type']==O[i]['type'] and N[i].get('widgets_values')==O[i].get('widgets_values'), i
    org_sg={s['id']:s for s in org['definitions']['subgraphs']}
    for sg in w['definitions']['subgraphs']: assert sg==org_sg[sg['id']]
    assert 13448 not in N and 13464 not in N
    assert N[13438]['widgets_values']==['krast_v20.safetensors','Qwen3-VL-4B-Instruct-Heretic-int8-convrot.safetensors','krea2RealVae_v10.safetensors']
    assert N[13441]['widgets_values'][0]['loras']==[
        {'name':'millie_000002750.safetensors','on':True,'sm':1.0,'sc':1.0,'triggers':[]},
        {'name':'krea-smartphone-photo-slider.safetensors','on':True,'sm':1.0,'sc':1.0,'triggers':[]}]
    # the original card is untouched
    assert O[13441]['widgets_values'][0]['loras']==[{'name':'millie_000002750.safetensors','on':True,'sm':1.0,'sc':1.0,'triggers':[]}]
    pats={json.loads(n['properties']['saveImageState'])['pattern'] for n in w['nodes'] if n['type']=='PixaromaSaveImage'}
    assert pats=={'KRAST_ORG_POSE/final_%counter%','KRAST_ORG_POSE/face_%counter%','KRAST_ORG_POSE/stage2_%counter%'}


def test_pose_technique_wiring():
    w=_load(SRC); N={n['id']:n for n in w['nodes']}
    L={l[0]:l for l in w['links']}
    src=lambda nid,name: (lambda l: (N[l[1]],l[2]))(L[next(x for x in N[nid]['inputs'] if x['name']==name)['link']])
    by=lambda t: [n for n in w['nodes'] if n['type']==t]
    ks=N[170]
    assert ks['widgets_values']==[2,'fixed',8,1,'er_sde','simple',0.85]
    pose_lora,_=src(170,'model')
    assert pose_lora['type']=='LoraLoaderModelOnly' and pose_lora['widgets_values']==['krea2_turbo_openpose_controlnet.safetensors',0.75]
    patch,_=src(pose_lora['id'],'model')
    assert patch['type']=='Krea2OstrisEditModelPatch' and patch['widgets_values']==[True]
    assert src(patch['id'],'model')[0]['id']==13441
    # stage 2 and detailers keep the plain LoRA model (no Ostris patch, no pose LoRA)
    assert src(1635,'model')[0]['id']==13441 and src(13452,'model')[0]['id']==13441
    pos,_=src(170,'positive'); neg,_=src(170,'negative')
    assert pos['type']==neg['type']=='FluxKontextMultiReferenceLatentMethod'
    assert pos['widgets_values']==neg['widgets_values']==['index_timestep_zero']
    pe,_=src(pos['id'],'conditioning'); ne,_=src(neg['id'],'conditioning')
    assert pe['type']==ne['type']=='TextEncodeKrea2OstrisEdit' and ne['widgets_values']==['']
    for e in (pe,ne):
        assert src(e['id'],'clip')==(N[13438],1) and src(e['id'],'vae')==(N[13438],0)
        i1,_=src(e['id'],'image1'); i2,_=src(e['id'],'image2'); i3,_=src(e['id'],'image3')
        assert i1['type']==i2['type']==i3['type']=='FluxKontextImageScale'
        blur,_=src(i1['id'],'image'); assert blur['type']=='ImageBlur' and blur['widgets_values']==[2,1]
        dw,_=src(i2['id'],'image'); assert dw['type']=='DWPreprocessor'
        assert dw['widgets_values']==['enable','enable','enable',1024,'yolox_l.onnx','dw-ll_ucoco_384_bs5.torchscript.pt','disable']
        assert src(dw['id'],'image')[0] is src(blur['id'],'image')[0]
        assert src(i3['id'],'image')[0]['type']=='LoadImage'
    lat,_=src(170,'latent_image')
    assert lat['type']=='VAEEncode' and src(lat['id'],'vae')==(N[13438],0) and src(lat['id'],'pixels')[0] is i1
    prompt,_=src(pe['id'],'prompt')
    assert prompt['type']=='PrimitiveStringMultiline' and src(112,'text')[0] is prompt
    assert json.loads(prompt['widgets_values'][0])['image_roles'].startswith('The first image is the reference photo')
    assert len(by('LoadImage'))==2 and all(n['widgets_values'][0]=='example.png' for n in by('LoadImage'))


def test_catalog_extends_krast_org_with_the_pose_assets():
    assert CATALOG[PROFILE+'_files']==CATALOG['krast_org_files']+['dwpose-det','dwpose-pose','pose-lora','6gsc-smartphone']
    assert CATALOG[PROFILE+'_nodes']==CATALOG['krast_org_nodes']+['comfyui_controlnet_aux','comfyui-krea2-ostris-edit']
    assert CATALOG[PROFILE+'_links']==CATALOG['krea2_pose_millie_links']
    dests={CATALOG['files'][f]['destination'].split('/',1)[1] for f in CATALOG[PROFILE+'_files']}
    used={v for n in _all_nodes(_load(SRC)) if isinstance(n.get('widgets_values'),list) for v in n['widgets_values']
          if isinstance(v,str) and v.endswith(('.safetensors','.pt','.pth','.onnx'))}
    used|={r['name'] for n in _load(SRC)['nodes'] if n['type']=='PixaromaLoraLoader' for r in n['widgets_values'][0]['loras']}
    for name in used:
        if name in ('yolox_l.onnx','dw-ll_ucoco_384_bs5.torchscript.pt'): continue
        assert any(d.endswith(name) for d in dests), name
    names={n['name'] for n in CATALOG['nodes']}
    assert set(CATALOG[PROFILE+'_nodes'])<=names


def test_install_card_and_download(monkeypatch):
    r,captured,html=start_install(monkeypatch,{'profile':PROFILE,'precision':'int8'})
    assert r.status_code==200 and captured['precision']=='fp8' and captured['selfism_profile']==PROFILE
    assert {f['id'] for f in captured['files']}==set(CATALOG[PROFILE+'_files'])
    assert {n['name'] for n in captured['custom_nodes']}==set(CATALOG[PROFILE+'_nodes'])
    assert captured['model_links']==CATALOG[PROFILE+'_links']
    assert captured['selfism_repair'] and captured['selfism_repair_paths_only']
    assert f'data-sf-action="{PROFILE}"' in html
    assert html.index('<h2>Krast ORG</h2>') < html.index('<h2>Krast ORG Pose Recreate (Millie)</h2>')
    with TestClient(m.app) as client:
        response=client.get('/api/selfism/workflow/'+PROFILE)
        assert response.json()==_load(SRC) and OUT in response.headers['content-disposition']
        assert client.get('/api/selfism/workflow/krast_org').json()==_load('krast_org.json')


def test_installer_saves_without_overwriting_user_edits(monkeypatch,tmp_path):
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
    org=folder/'KRAST_ORG_Millie.json';org.write_text('existing krast org')
    wf={'selfism_profile':PROFILE,'precision':'fp8','files':[{'id':'x'}],'selfism_repair':True,'selfism_repair_paths_only':True}
    asyncio.run(ctrl._install_workflow(wf))
    path=folder/OUT
    assert path.read_bytes()==(ROOT/'selfism_workflows'/SRC).read_bytes()
    path.write_text('user edit')
    asyncio.run(ctrl._install_workflow(wf))
    assert path.read_text()=='user edit' and org.read_text()=='existing krast org'

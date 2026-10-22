import json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_krast_org_catalog_and_workflow():
    c=json.loads((ROOT/'catalog/selfism.json').read_text())
    files=[c['files'][k] for k in c['krast_org_files']]
    assert len(files)==12
    for f in files:
        assert re.fullmatch(r'[0-9a-f]{64}',f['sha256']) and f['size_bytes']>0
    names={n['name'] for n in c['nodes']}
    assert set(c['krast_org_nodes'])<=names and len(c['krast_org_nodes'])==12
    wf=(ROOT/'selfism_workflows/krast_org.json').read_text()
    assert 'blonde_epoch_8' not in wf and 'KRASTINT8' not in wf and 'millie_000002750.safetensors' in wf
    dests={f['destination'].split('/',1)[1] for f in files}
    w=json.loads(wf); ns=list(w['nodes'])+[n for s in w['definitions']['subgraphs'] for n in s['nodes']]
    used=[v for n in ns if isinstance(n.get('widgets_values'),list) for v in n['widgets_values'] if isinstance(v,str) and v.endswith(('.safetensors','.pt','.pth'))]
    assert used
    for m in used:
        assert any(d.endswith(m) for d in dests), m


def test_krast_org_node_packs_install_only_what_the_workflow_uses():
    c=json.loads((ROOT/'catalog/selfism.json').read_text())
    nodes={n['name']:n for n in c['nodes']}
    # Verified by loading each pack alone in ComfyUI v0.38.1 with only these packages:
    # the used nodes (Resolution, 1hew_AnySwitchBool, Eclipse toggles, Switch latent) register.
    expected={'CRT-Nodes':['opencv-python-headless','soundfile'],
              'ComfyUI-1hewNodes':[],
              'ComfyUI_Eclipse':['opencv-python-headless','pilgram'],
              'ComfyUI-Crystools':['deepdiff','pynvml','py-cpuinfo','piexif']}
    for name,packages in expected.items():
        assert nodes[name]['requirements_packages']==packages
    heavy={'openai-whisper','omnivoice','librosa','pedalboard','rembg','transparent-background','onnxruntime','torchaudio','torch','torchvision'}
    for name in c['krast_org_nodes']:
        assert not heavy & set(nodes[name].get('requirements_packages',[])), name
    w=json.loads((ROOT/'selfism_workflows/krast_org.json').read_text())
    used={n['type'] for n in list(w['nodes'])+[n for s in w['definitions']['subgraphs'] for n in s['nodes']]}
    for cls in ('Resolution','1hew_AnySwitchBool','Switch latent [Crystools]','Fast Mode Toggle [Eclipse]','Mute / Bypass Repeater [Eclipse]'):
        assert cls in used


def test_rawgirl_millie_card():
    c=json.loads((ROOT/'catalog/selfism.json').read_text())
    files=[c['files'][k] for k in c['rawgirl_millie_files']]
    dests={f['destination'] for f in files}
    assert dests=={'models/diffusion_models/rawgirlKrea2_v20Fp8.safetensors','models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors','models/vae/qwen_image_vae.safetensors','models/loras/millie_000002750.safetensors'}
    for f in files:
        assert re.fullmatch(r'[0-9a-f]{64}',f['sha256']) and f['size_bytes']>0
    assert c['rawgirl_millie_nodes']==['RES4LYF']
    w=json.loads((ROOT/'selfism_workflows/rawgirl_millie.json').read_text())
    core={'CLIPLoader','ConditioningZeroOut','EmptyLatentImage','VAEDecode','UNETLoader','VAELoader','PreviewImage','CLIPTextEncode','LoraLoaderModelOnly','Note'}
    types={n['type'] for n in w['nodes']}
    assert types-core=={'ClownsharKSampler_Beta'}
    used={v for n in w['nodes'] for v in (n.get('widgets_values') or []) if isinstance(v,str) and v.endswith('.safetensors')}
    assert used=={d.rsplit('/',1)[1] for d in dests}
    src=(ROOT/'launcher/selfism.py').read_text()
    assert "'rawgirl_millie'" in src and 'rawgirl_millie.json' in src
    for html in ('launcher/static/index.html','scripts/selfism-section.html'):
        page=(ROOT/html).read_text()
        assert 'data-sf-action="rawgirl_millie"' in page and '96 GB' not in page.split('sf-card-rawgirl-millie')[1].split('</article>')[0]


def test_krast_org_load_image_and_saves():
    w=json.loads((ROOT/'selfism_workflows/krast_org.json').read_text())
    top={n['id']:n for n in w['nodes']}
    loads=[n for n in w['nodes'] if n['type']=='LoadImage']
    assert len(loads)==1 and loads[0]['widgets_values'][0]=='example.png'
    assert not any(n['type']=='LoadImage' for s in w['definitions']['subgraphs'] for n in s['nodes'])
    assert '2026-07-17003' not in json.dumps(w)
    assert not any(p[0]=='13440' for p in top[13448]['properties']['proxyWidgets'])
    pats={json.loads(n['properties']['saveImageState'])['pattern'] for n in w['nodes'] if n['type']=='PixaromaSaveImage'}
    assert pats=={'KRAST_ORG/final_%counter%','KRAST_ORG/face_%counter%','KRAST_ORG/stage2_%counter%'}
    loras=top[13441]['widgets_values'][0]['loras']
    assert loras==[{'name':'millie_000002750.safetensors','on':True,'sm':1.0,'sc':1.0,'triggers':[]}]
    # every link is consistent at both ends (top level and inside subgraphs)
    def ok(nodes,links,io=None):
        N={n['id']:n for n in nodes}
        for l in links:
            i,o,os_,t,ts=(l['id'],l['origin_id'],l['origin_slot'],l['target_id'],l['target_slot']) if isinstance(l,dict) else l[:5]
            assert (i in io['inputs'][os_]['linkIds']) if o==-10 else i in (N[o]['outputs'][os_].get('links') or [])
            assert (i in io['outputs'][ts]['linkIds']) if t==-20 else N[t]['inputs'][ts].get('link')==i
    ok(w['nodes'],w['links'])
    for s in w['definitions']['subgraphs']: ok(s['nodes'],s['links'],s)


def _models(w):
    ns=list(w['nodes'])+[n for s in (w.get('definitions') or {}).get('subgraphs',[]) for n in s['nodes']]
    out=set()
    def sc(v):
        if isinstance(v,str) and re.search(r'\.(safetensors|pth|pt)$',v): out.add(v.rsplit('/',1)[-1])
        elif isinstance(v,list):
            for x in v: sc(x)
        elif isinstance(v,dict):
            for x in v.values(): sc(x)
    for n in ns: sc(n.get('widgets_values'))
    return out


def test_krast_org_depth_and_rawgirl_cards():
    c=json.loads((ROOT/'catalog/selfism.json').read_text())
    names={n['name'] for n in c['nodes']}
    ag=[n for n in c['nodes'] if n['name']=='ComfyUI-AutoGrid'][0]
    assert re.fullmatch(r'[0-9a-f]{40}',ag['ref'])
    kj=[n for n in c['nodes'] if n['name']=='ComfyUI-KJNodes'][0]
    assert kj.get('install_requirements',True) and 'requirements_packages' not in kj  # full requirements.txt -> color-matcher
    for prof,count in (('krast_org_depth',14),('rawgirl',7)):
        files=[c['files'][k] for k in c[prof+'_files']]
        assert len(files)==count and set(c[prof+'_nodes'])<=names
        for f in files: assert re.fullmatch(r'[0-9a-f]{64}',f['sha256']) and f['size_bytes']>0
        base={f['destination'].rsplit('/',1)[1] for f in files}
        wfs=list((ROOT/'selfism_workflows'/prof).glob('*.json'))
        used=set().union(*[_models(json.loads(p.read_text())) for p in wfs])
        assert used==base, (prof,used^base)
        assert "'"+prof+"'" in (ROOT/'launcher/selfism.py').read_text()
        for html in ('launcher/static/index.html','scripts/selfism-section.html'):
            assert f'data-sf-action="{prof}"' in (ROOT/html).read_text()
    enc=c['files']['encoder']
    assert enc['destination']=='models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors' and enc['size_bytes']==6020316384
    assert len(list((ROOT/'selfism_workflows/krast_org_depth').glob('*.json')))==8
    assert len(list((ROOT/'selfism_workflows/rawgirl').glob('*.json')))==4
    depth_users=[p for p in ROOT.joinpath('selfism_workflows/krast_org_depth').glob('*.json') if 'DepthAnythingV2Preprocessor' in p.read_text()]
    assert depth_users


AUX=['einops','huggingface-hub','pyyaml','scipy','scikit-image','matplotlib']
IMPACT=['segment-anything','scikit-image','piexif','opencv-python-headless','scipy','dill','matplotlib']
DEPTH_PROFILES=('krast','recreate_hybrid','krea_fluxdepth','krast_fluxdepth','krast_faithful_klein_v5','krast_org_depth','rawgirl')


def test_shared_packs_install_only_the_preprocessors_the_cards_use():
    c=json.loads((ROOT/'catalog/selfism.json').read_text())
    nodes={n['name']:n for n in c['nodes']}
    assert nodes['comfyui_controlnet_aux']['requirements_packages']==AUX
    assert nodes['ComfyUI-Impact-Pack']['requirements_packages']==IMPACT
    for name in ('comfyui_controlnet_aux','ComfyUI-Impact-Pack'):
        blob=' '.join(nodes[name]['requirements_packages']).lower()
        assert 'opencv-contrib' not in blob and 'sam2' not in blob and 'onnxruntime' not in blob
    users=[key[:-6] for key,value in c.items() if key.endswith('_nodes') and 'comfyui_controlnet_aux' in value]
    assert 'krast_org_depth' in users and len(users)>=11
    assert 'ComfyUI-Impact-Pack' in c['krast_org_nodes'] and 'sam2' not in IMPACT


def test_depth_anything_is_linked_for_every_card_that_downloads_it(monkeypatch):
    from test_full import start_install
    c=json.loads((ROOT/'catalog/selfism.json').read_text())
    link=c['depth_anything_links'][0]
    assert link==c['reference_links'][0]==c['full_links'][0]==c['full_refine_links'][0]
    assert link['destination']=='custom_nodes/comfyui_controlnet_aux/ckpts/depth-anything/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth'
    for prof in DEPTH_PROFILES:
        r,captured,_=start_install(monkeypatch,{'profile':prof})
        assert r.status_code==200, prof
        assert captured['model_links'].count(link)==1, prof
        assert any(f['destination']==link['source'] for f in captured['files']), prof
    for prof in ('full','full_refine','reference','carousel'):
        r,captured,_=start_install(monkeypatch,{'profile':prof})
        assert r.status_code==200, prof
        assert captured['model_links'].count(link)==1, prof
        assert captured['selfism_repair_paths_only'] is False
    r,captured,_=start_install(monkeypatch,{'profile':'krast_org'})
    assert captured['selfism_repair'] is True and captured['selfism_repair_paths_only'] is True
    assert link not in captured['model_links']
    r,captured,html=start_install(monkeypatch,{'profile':'krast_org_depth'})
    assert captured['selfism_repair'] is True and captured['selfism_repair_paths_only'] is True
    assert captured['model_links']==[link]
    for page in (html,(ROOT/'scripts/selfism-section.html').read_text(encoding='utf-8')):
        card=page.split('sf-card-krast-org-depth')[1].split('</article>')[0]
        assert '15 paketa custom nodova sa privatnog R2' not in card
        assert 'već u image-u' in card and 'controlnet_aux/ckpts' in card


def test_depth_anything_file_lands_in_the_preprocessor_ckpt_dir(tmp_path,monkeypatch):
    import asyncio
    from launcher import app as launcher_app
    c=json.loads((ROOT/'catalog/selfism.json').read_text())
    comfy=tmp_path/'ComfyUI'
    source=comfy/'models/controlnet_aux/depth-anything/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'depth-bytes')
    monkeypatch.setattr(launcher_app,'COMFYUI_DIR',comfy)
    asyncio.run(launcher_app.JobController()._apply_model_links(c['depth_anything_links']))
    dest=comfy/'custom_nodes/comfyui_controlnet_aux/ckpts/depth-anything/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth'
    assert dest.read_bytes()==b'depth-bytes'
    assert dest.stat().st_ino==source.stat().st_ino


def test_krast_org_repair_only_configures_cuda_paths(monkeypatch,tmp_path):
    import asyncio
    import importlib
    m=importlib.import_module('launcher.app')
    s=importlib.import_module('launcher.selfism')
    monkeypatch.setattr(m,'COMFYUI_DIR',tmp_path)
    monkeypatch.setattr(m,'COMFYUI_VENV',tmp_path/'.venv')
    python=m.COMFYUI_VENV/'bin/python'
    python.parent.mkdir(parents=True)
    python.touch()
    ctrl=type(m.selfism_controller)()
    calls=[]
    async def ready(*_a,**_k): pass
    async def run(*args,**_kwargs):
        calls.append(tuple(str(part) for part in args))
        return 0,'torch==2.10.0\ntorchvision==0\ntorchaudio==0\nnumpy==0\ntransformers==0\n'
    async def sources(_host,files): return files,[]
    monkeypatch.setattr(ctrl,'_wait_for_comfyui',ready)
    monkeypatch.setattr(ctrl,'_run_process',run)
    monkeypatch.setattr(m.JobController,'_install_workflow',ready)
    monkeypatch.setattr(s,'resolve_sources',sources)
    for prof in ('krast_org','krast_org_depth'):
        calls.clear()
        asyncio.run(ctrl._install_workflow({'selfism_profile':prof,'precision':'fp8','files':[{'id':'x'}],'custom_nodes':[{'name':'x'}],'selfism_repair':True,'selfism_repair_paths_only':True}))
        repair=[c for c in calls if any(part.endswith('selfism_runtime.py') for part in c)]
        assert repair and repair[0][-1]=='--paths-only', prof
    calls.clear()
    asyncio.run(ctrl._install_workflow({'selfism_profile':'6gsc','precision':'fp8','files':[{'id':'x'}],'selfism_repair':True,'selfism_repair_paths_only':False}))
    repair=[c for c in calls if any(part.endswith('selfism_runtime.py') for part in c)]
    assert repair and '--paths-only' not in repair[0]

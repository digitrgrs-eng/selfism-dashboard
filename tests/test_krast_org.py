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

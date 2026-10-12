"""Hearmeman ImageToImage card installs every model and node pack the workflow needs."""
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
from test_full import ROOT, CATALOG, start_install, m

PATH=ROOT/'selfism_workflows/hearmeman_i2i.json'
WORKFLOW=json.loads(PATH.read_text(encoding='utf-8'))
CORE={'BasicGuider','BasicScheduler','CLIPLoader','CLIPTextEncode','LoadImage','PrimitiveStringMultiline','RandomNoise',
      'RegexExtract','RegexReplace','ResolutionSelector','SamplerCustomAdvanced','SamplerEulerAncestral','SaveImage','String',
      'StringConcatenate','StringReplace','TextGenerate','UNETLoader','VAEDecode','VAEEncode','VAELoader'}
PACKS={'Display Any (rgthree)':'rgthree-comfy','Power Lora Loader (rgthree)':'rgthree-comfy','ImageResizeKJv2':'ComfyUI-KJNodes',
       'SimpleMathSlider+':'ComfyUI_essentials','DetailDaemonSamplerNode':'ComfyUI-Detail-Daemon',
       'SeedVR2LoadDiTModel':'ComfyUI-SeedVR2_VideoUpscaler','SeedVR2LoadVAEModel':'ComfyUI-SeedVR2_VideoUpscaler',
       'SeedVR2VideoUpscaler':'ComfyUI-SeedVR2_VideoUpscaler'}


def all_nodes():
    nodes=list(WORKFLOW['nodes'])
    for sg in WORKFLOW.get('definitions',{}).get('subgraphs',[]): nodes+=sg['nodes']
    return nodes


def test_workflow_is_shipped_unmodified():
    assert hashlib.sha256(PATH.read_bytes()).hexdigest()=='e3194094627f90c80a72523f14d6d8d852de6a46433a52aef0cc359cdbfa28ac'


def test_every_node_and_model_is_covered():
    subgraphs={sg['id'] for sg in WORKFLOW['definitions']['subgraphs']}
    dest={Path(CATALOG['files'][k]['destination']).as_posix() for k in CATALOG['hearmeman_i2i_files']}
    for n in all_nodes():
        t=n['type']
        assert t in CORE or t in subgraphs or PACKS.get(t) in CATALOG['hearmeman_i2i_nodes'], t
        folder={'UNETLoader':'diffusion_models','CLIPLoader':'text_encoders','VAELoader':'vae',
                'SeedVR2LoadDiTModel':'SEEDVR2','SeedVR2LoadVAEModel':'SEEDVR2'}.get(t)
        if folder: assert f'models/{folder}/{n["widgets_values"][0]}' in dest
    for k in CATALOG['hearmeman_i2i_files']:
        f=CATALOG['files'][k]; assert len(f['sha256'])==64 and f['size_bytes']>0


def test_install_and_download(monkeypatch):
    r,captured,html=start_install(monkeypatch,{'profile':'hearmeman_i2i','precision':'int8'})
    assert r.status_code==200 and captured['selfism_profile']=='hearmeman_i2i'
    assert {f['id'] for f in captured['files']}==set(CATALOG['hearmeman_i2i_files'])
    assert {n['name'] for n in captured['custom_nodes']}==set(CATALOG['hearmeman_i2i_nodes'])
    assert '<h2>Hearmeman ImageToImage</h2>' in html and 'data-sf-action="hearmeman_i2i"' in html
    with TestClient(m.app) as client:
        resp=client.get('/api/selfism/workflow/hearmeman_i2i')
        assert resp.content==PATH.read_bytes()
        assert 'Hearmeman_ImageToImage.json' in resp.headers['content-disposition']

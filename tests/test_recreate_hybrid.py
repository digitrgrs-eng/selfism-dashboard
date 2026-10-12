"""Selfism Recreate Hybrid card installs every model and node pack the workflow needs."""
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
from test_full import ROOT, CATALOG, start_install, m

PATH=ROOT/'selfism_workflows/recreate_hybrid.json'
WORKFLOW=json.loads(PATH.read_text(encoding='utf-8'))
CORE={'BasicGuider','BasicScheduler','CLIPLoader','CLIPTextEncode','LoadImage','PrimitiveStringMultiline','RandomNoise',
      'RegexReplace','ImageScaleToTotalPixels','SamplerCustomAdvanced','SamplerEulerAncestral','SaveImage',
      'StringConcatenate','TextGenerate','UNETLoader','VAEDecode','VAEEncode','VAELoader','Note'}
PACKS={'String':'ComfyLiterals','Display Any (rgthree)':'rgthree-comfy','Power Lora Loader (rgthree)':'rgthree-comfy','LazySwitchKJ':'ComfyUI-KJNodes',
       'DepthAnythingV2Preprocessor':'comfyui_controlnet_aux','Krea2ControlImageEncode':'comfyui-krea2-controlnet',
       'Krea2ControlLoRALoader':'comfyui-krea2-controlnet','Krea2ControlApply':'comfyui-krea2-controlnet',
       'DetailDaemonSamplerNode':'ComfyUI-Detail-Daemon',
       'SeedVR2LoadDiTModel':'ComfyUI-SeedVR2_VideoUpscaler','SeedVR2LoadVAEModel':'ComfyUI-SeedVR2_VideoUpscaler',
       'SeedVR2VideoUpscaler':'ComfyUI-SeedVR2_VideoUpscaler'}


def all_nodes():
    nodes=list(WORKFLOW['nodes'])
    for sg in WORKFLOW.get('definitions',{}).get('subgraphs',[]): nodes+=sg['nodes']
    return nodes


def test_workflow_is_shipped_unmodified():
    assert hashlib.sha256(PATH.read_bytes()).hexdigest()=='892d14ab2b0e9e0d32c4f3c016c82bb4dee9d5cd4327ca6b8f4a7fbeb0efad10'


def test_every_node_and_model_is_covered():
    subgraphs={sg['id'] for sg in WORKFLOW.get('definitions',{}).get('subgraphs',[])}
    dest={Path(CATALOG['files'][k]['destination']).as_posix() for k in CATALOG['recreate_hybrid_files']}
    for n in all_nodes():
        t=n['type']
        assert t in CORE or t in subgraphs or PACKS.get(t) in CATALOG['recreate_hybrid_nodes'], t
        folder={'UNETLoader':'diffusion_models','CLIPLoader':'text_encoders','VAELoader':'vae','Krea2ControlLoRALoader':'loras',
                'DepthAnythingV2Preprocessor':'controlnet_aux/depth-anything/Depth-Anything-V2-Large',
                'SeedVR2LoadDiTModel':'SEEDVR2','SeedVR2LoadVAEModel':'SEEDVR2'}.get(t)
        if folder: assert f'models/{folder}/{n["widgets_values"][0]}' in dest
        if t=='Power Lora Loader (rgthree)':
            for w in n['widgets_values']:
                if isinstance(w,dict) and 'lora' in w: assert f'models/loras/{w["lora"]}' in dest
    for k in CATALOG['recreate_hybrid_files']:
        f=CATALOG['files'][k]; assert len(f['sha256'])==64 and f['size_bytes']>0


def test_install_and_download(monkeypatch):
    r,captured,html=start_install(monkeypatch,{'profile':'recreate_hybrid','precision':'int8'})
    assert r.status_code==200 and captured['selfism_profile']=='recreate_hybrid'
    assert {f['id'] for f in captured['files']}==set(CATALOG['recreate_hybrid_files'])
    assert {n['name'] for n in captured['custom_nodes']}==set(CATALOG['recreate_hybrid_nodes'])
    assert '<h2>Selfism Recreate Hybrid</h2>' in html and 'data-sf-action="recreate_hybrid"' in html
    with TestClient(m.app) as client:
        resp=client.get('/api/selfism/workflow/recreate_hybrid')
        assert resp.content==PATH.read_bytes()
        assert 'Selfism_Recreate_Hybrid.json' in resp.headers['content-disposition']

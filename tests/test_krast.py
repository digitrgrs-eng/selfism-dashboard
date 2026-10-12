"""KRAST Recreate + Carousel card installs every model and node pack both workflows need."""
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
from test_full import ROOT, CATALOG, start_install, m

PATHS={'krast_recreate':(ROOT/'selfism_workflows/krast_recreate.json','487c9247292f44562a399624932187b9291ad0ad463cf068ad4babadb230f54d','KRAST_Recreate.json'),
       'krast_carousel':(ROOT/'selfism_workflows/krast_carousel.json','805956963ee62602814546faa9259ad93b2e657d85dfd7feec9f0c38eb69e568','KRAST_Carousel.json')}
CORE={'CLIPLoader','KSampler','LoadImage','ImageScaleToTotalPixels','SaveImage','UNETLoader','VAEDecode','VAEEncode','VAELoader','Note'}
PACKS={'Display Any (rgthree)':'rgthree-comfy','Power Lora Loader (rgthree)':'rgthree-comfy','LazySwitchKJ':'ComfyUI-KJNodes',
       'ArtfatLLMPrompter':'artfat-comfyui-llm-prompter',
       'DepthAnythingV2Preprocessor':'comfyui_controlnet_aux','Krea2ControlImageEncode':'comfyui-krea2-controlnet',
       'Krea2ControlLoRALoader':'comfyui-krea2-controlnet','Krea2ControlApply':'comfyui-krea2-controlnet',
       'SeedVR2LoadDiTModel':'ComfyUI-SeedVR2_VideoUpscaler','SeedVR2LoadVAEModel':'ComfyUI-SeedVR2_VideoUpscaler',
       'SeedVR2VideoUpscaler':'ComfyUI-SeedVR2_VideoUpscaler'}
BUNDLED={'KRASTV3PreserveFirstGrid':'ComfyUI-KRASTv3-Carousel'}


def test_workflows_shipped_unmodified():
    for path,digest,_ in PATHS.values(): assert hashlib.sha256(path.read_bytes()).hexdigest()==digest


def test_every_node_and_model_is_covered():
    dest={Path(CATALOG['files'][k]['destination']).as_posix() for k in CATALOG['krast_files']}
    assert 'RES4LYF' in CATALOG['krast_nodes']  # registers the res_2s sampler
    for path,_,_ in PATHS.values():
        for n in json.loads(path.read_text(encoding='utf-8'))['nodes']:
            t=n['type']; w=n['widgets_values']
            if t in BUNDLED:
                assert BUNDLED[t] in CATALOG['krast_bundled_nodes']
                assert (ROOT/'bundled_nodes'/BUNDLED[t]/'__init__.py').read_text().count(t)>0
                continue
            assert t in CORE or PACKS.get(t) in CATALOG['krast_nodes'], t
            folder={'UNETLoader':'diffusion_models','CLIPLoader':'text_encoders','VAELoader':'vae','Krea2ControlLoRALoader':'loras',
                    'DepthAnythingV2Preprocessor':'controlnet_aux/depth-anything/Depth-Anything-V2-Large',
                    'SeedVR2LoadDiTModel':'SEEDVR2','SeedVR2LoadVAEModel':'SEEDVR2'}.get(t)
            if folder: assert f'models/{folder}/{w[0]}' in dest
            if t=='ArtfatLLMPrompter': assert f'models/LLM/{w[0]}' in dest and f'models/LLM/{w[1]}' in dest
            if t=='SeedVR2VideoUpscaler': assert 0<=w[0]<=4294967295
            if t=='Power Lora Loader (rgthree)':
                for x in w:
                    if isinstance(x,dict) and 'lora' in x: assert f'models/loras/{x["lora"]}' in dest
    for k in CATALOG['krast_files']:
        f=CATALOG['files'][k]; assert len(f['sha256'])==64 and f['size_bytes']>0


def test_install_and_download(monkeypatch):
    r,captured,html=start_install(monkeypatch,{'profile':'krast','precision':'int8'})
    assert r.status_code==200 and captured['selfism_profile']=='krast' and captured['selfism_repair']
    assert {f['id'] for f in captured['files']}==set(CATALOG['krast_files'])
    assert {n['name'] for n in captured['custom_nodes']}==set(CATALOG['krast_nodes'])
    assert '<h2>KRAST Recreate + Carousel</h2>' in html and 'data-sf-action="krast"' in html
    with TestClient(m.app) as client:
        for profile,(path,_,name) in PATHS.items():
            resp=client.get('/api/selfism/workflow/'+profile)
            assert resp.content==path.read_bytes() and name in resp.headers['content-disposition']

"""Krea 2 / KRAST FluxDepth Millie cards install every model + node pack the workflow needs, plus the realism LoRA bundle."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_full import ROOT, CATALOG, start_install, m

CASES={'krea_fluxdepth':(ROOT/'selfism_workflows/krea_fluxdepth_millie.json','Krea2_FluxKlein_Millie_Artfat_Depth_v3.json'),
       'krast_fluxdepth':(ROOT/'selfism_workflows/krast_fluxdepth_millie.json','KRAST_FluxKlein_Millie_Artfat_Depth_v3.json')}
PACKS={'ArtfatLLMPrompter':'artfat-comfyui-llm-prompter','CR Text':'ComfyUI_Comfyroll_CustomNodes','ColorMatch':'ComfyUI-KJNodes','LazySwitchKJ':'ComfyUI-KJNodes',
       'DetailDaemonSamplerNode':'ComfyUI-Detail-Daemon','Display Any (rgthree)':'rgthree-comfy','Power Lora Loader (rgthree)':'rgthree-comfy',
       'FaceDetailer':'ComfyUI-Impact-Pack','SAMLoader':'ComfyUI-Impact-Pack','UltralyticsDetectorProvider':'ComfyUI-Impact-Subpack',
       'DepthAnythingV2Preprocessor':'comfyui_controlnet_aux','Krea2ControlLoRALoader':'comfyui-krea2-controlnet','Krea2ControlApply':'comfyui-krea2-controlnet'}
FOLDER={'UNETLoader':'diffusion_models','CLIPLoader':'text_encoders','VAELoader':'vae','LoraLoaderModelOnly':'loras','Krea2ControlLoRALoader':'loras',
        'SAMLoader':'sams','UltralyticsDetectorProvider':'ultralytics'}
REALISM={'lora-candid-snap','lora-krea-amateur-v4','lora-krea2-realism-v2','6gsc-snofs','lora-refusal-reduction-v2','lora-skin-tone','lora-lowres','lora-light','lora-detail'}


@pytest.mark.parametrize('profile',list(CASES))
def test_every_node_and_model_is_covered(profile):
    path,_=CASES[profile]
    dest={Path(CATALOG['files'][k]['destination']).as_posix() for k in CATALOG[profile+'_files']}
    w=json.loads(path.read_text(encoding='utf-8'))
    nodes=list(w['nodes'])+[n for sg in w.get('definitions',{}).get('subgraphs',[]) for n in sg.get('nodes',[])]
    for n in nodes:
        t=n['type']; v=n.get('widgets_values') or []
        if t in PACKS: assert PACKS[t] in CATALOG[profile+'_nodes'], t
        if t in FOLDER: assert f'models/{FOLDER[t]}/{v[0]}' in dest, (t,v[0])
        if t=='ArtfatLLMPrompter': assert f'models/LLM/{v[0]}' in dest and f'models/LLM/{v[1]}' in dest
    assert REALISM<=set(CATALOG[profile+'_files'])
    for k in CATALOG[profile+'_files']:
        f=CATALOG['files'][k]; assert len(f['sha256'])==64 and f['size_bytes']>0


@pytest.mark.parametrize('profile',list(CASES))
def test_install_and_download(monkeypatch,profile):
    r,captured,html=start_install(monkeypatch,{'profile':profile,'precision':'int8'})
    assert r.status_code==200 and captured['selfism_profile']==profile
    assert {f['id'] for f in captured['files']}==set(CATALOG[profile+'_files'])
    assert {n['name'] for n in captured['custom_nodes']}==set(CATALOG[profile+'_nodes'])
    assert f'data-sf-action="{profile}"' in html
    path,name=CASES[profile]
    with TestClient(m.app) as client:
        resp=client.get('/api/selfism/workflow/'+profile)
        assert resp.content==path.read_bytes() and name in resp.headers['content-disposition']

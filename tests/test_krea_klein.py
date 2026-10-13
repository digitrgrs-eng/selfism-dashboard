"""Krea 2 + Flux Klein card installs every model and node pack both workflows need (incl. bypassed nodes)."""
import json
from pathlib import Path

from fastapi.testclient import TestClient
from test_full import ROOT, CATALOG, start_install, m

PATHS={'krea_klein_artfat':(ROOT/'selfism_workflows/krea_klein_artfat.json','Krea2_FluxKlein_Artfat_Reference.json'),
       'krea_klein_famegrid':(ROOT/'selfism_workflows/krea_klein_famegrid.json','Krea2_FluxKlein_Famegrid.json')}
PACKS={'ArtfatLLMPrompter':'artfat-comfyui-llm-prompter','CR Text':'ComfyUI_Comfyroll_CustomNodes','ColorMatch':'ComfyUI-KJNodes',
       'DetailDaemonSamplerNode':'ComfyUI-Detail-Daemon','Display Any (rgthree)':'rgthree-comfy','Image Comparer (rgthree)':'rgthree-comfy',
       'FaceDetailer':'ComfyUI-Impact-Pack','SAMLoader':'ComfyUI-Impact-Pack','UltralyticsDetectorProvider':'ComfyUI-Impact-Subpack'}
FOLDER={'UNETLoader':'diffusion_models','CLIPLoader':'text_encoders','VAELoader':'vae','LoraLoaderModelOnly':'loras',
        'SAMLoader':'sams','UltralyticsDetectorProvider':'ultralytics'}


def test_every_node_and_model_is_covered():
    dest={Path(CATALOG['files'][k]['destination']).as_posix() for k in CATALOG['krea_klein_files']}
    for path,_ in PATHS.values():
        for n in json.loads(path.read_text(encoding='utf-8'))['nodes']:
            t=n['type']; w=n.get('widgets_values') or []
            if t in PACKS: assert PACKS[t] in CATALOG['krea_klein_nodes'], t
            if t in FOLDER: assert f'models/{FOLDER[t]}/{w[0]}' in dest, (t,w[0])
            if t=='ArtfatLLMPrompter': assert f'models/LLM/{w[0]}' in dest and f'models/LLM/{w[1]}' in dest
    for k in CATALOG['krea_klein_files']:
        f=CATALOG['files'][k]; assert len(f['sha256'])==64 and f['size_bytes']>0


def test_install_and_download(monkeypatch):
    r,captured,html=start_install(monkeypatch,{'profile':'krea_klein','precision':'int8'})
    assert r.status_code==200 and captured['selfism_profile']=='krea_klein'
    assert {f['id'] for f in captured['files']}==set(CATALOG['krea_klein_files'])
    assert {n['name'] for n in captured['custom_nodes']}==set(CATALOG['krea_klein_nodes'])
    assert 'data-sf-action="krea_klein"' in html
    with TestClient(m.app) as client:
        for profile,(path,name) in PATHS.items():
            resp=client.get('/api/selfism/workflow/'+profile)
            assert resp.content==path.read_bytes() and name in resp.headers['content-disposition']

"""Millie Pose Recreate card: the PNG-embedded Krea2 Ostris + DWPose recipe, adapted to Millie."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
from test_full import ROOT, CATALOG, start_install, m, s

SOURCE=ROOT/'selfism_workflows/sources/krea2_pose_recreate_original.json'
ORIGINAL=json.loads(SOURCE.read_text(encoding='utf-8'))
PATH=ROOT/'selfism_workflows/krea2_pose_recreate_millie.json'
WORKFLOW=json.loads(PATH.read_text(encoding='utf-8'))
PROFILE='krea2_pose_millie'


def _strip(wf):
    """Graph without the documented edits (titles, texts, image names, LoRA rows, id)."""
    wf=copy.deepcopy(wf)
    wf.pop('id')
    for g in wf['groups']: g.pop('title')
    for n in wf['nodes']:
        n.pop('title',None)
        if n['type'] in ('CLIPTextEncode','TextEncodeKrea2OstrisEdit','PrimitiveStringMultiline','LoadImage','Power Lora Loader (rgthree)'):
            n.pop('widgets_values')
    return wf


def test_source_is_the_png_workflow_and_only_documented_edits_were_made():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='29e70c0e6bb27c2619db91c781da5765973ecac5eb492d9db1b77d45457dc8a2'
    assert _strip(ORIGINAL)==_strip(WORKFLOW)
    assert WORKFLOW['links']==ORIGINAL['links'] and WORKFLOW['id']!=ORIGINAL['id']
    text=json.dumps(WORKFLOW,ensure_ascii=False)
    for gone in ('b3llaa22','bellasima','Bella','BELLA','krea2_edit_outfit_transfer','krea2_smartphone_photography_slider','csontvaz','MENET'):
        assert gone not in text


def test_millie_lora_rows_and_prompts():
    nodes={n['id']:n for n in WORKFLOW['nodes']}
    rows=[r for r in nodes[1633]['widgets_values'] if isinstance(r,dict) and 'lora' in r]
    assert [(r['lora'],r['on'],r['strength']) for r in rows]==[
        ('MysticXXX_KREA2_v3.safetensors',False,0.9),
        ('famegrid_spicy.safetensors',False,0.45),
        ('krea2-bloomgirls-realism-step00004000.safetensors',False,0.45),
        ('millie_000002750.safetensors',True,1),
        ('krea-smartphone-photo-slider.safetensors',True,1)]
    for nid in (1617,1764,1774,1793,1794):
        assert nodes[nid]['widgets_values'][0].startswith('m1lli3, young woman')
    # Both edit encoders take the manual prompt node; no LLM node exists.
    assert {l[3] for l in WORKFLOW['links'] if l[1]==1793}=={1764,1794}
    assert not any('llm' in n['type'].lower() or 'gguf' in n['type'].lower() for n in WORKFLOW['nodes'])
    assert nodes[1765]['widgets_values']==['krea2_turbo_openpose_controlnet.safetensors',0.75]
    assert nodes[1634]['widgets_values'][6]==0.85 and nodes[1790]['widgets_values'][6]==0.2


def test_graph_connections_and_model_coverage():
    nodes={n['id']:n for n in WORKFLOW['nodes']}
    for lid,a,ao,b,bi,typ in WORKFLOW['links']:
        assert nodes[b]['inputs'][bi]['link']==lid
        assert lid in nodes[a]['outputs'][ao]['links']
    names={Path(CATALOG['files'][k]['destination']).name for k in CATALOG[PROFILE+'_files']}
    for n in nodes.values():
        if n['type'] in ('UNETLoader','CLIPLoader','VAELoader','SAMLoader','UltralyticsDetectorProvider','LoraLoaderModelOnly'):
            assert Path(n['widgets_values'][0]).name in names
        if n['type']=='DWPreprocessor':
            assert set(n['widgets_values'][4:6]) <= names
        if n['type']=='Power Lora Loader (rgthree)':
            for row in n['widgets_values']:
                if isinstance(row,dict) and 'lora' in row: assert row['lora'] in names
    seg=CATALOG['files']['person-seg-m']
    assert seg['destination']=='models/ultralytics/segm/person_yolov8m-seg.pt' and seg['size_bytes']==54827683
    assert len(CATALOG[PROFILE+'_files'])==len(set(CATALOG[PROFILE+'_files']))==14
    assert all(len(CATALOG['files'][k]['sha256'])==64 for k in CATALOG[PROFILE+'_files'])
    known={n['name'] for n in CATALOG['nodes']}
    assert set(CATALOG[PROFILE+'_nodes'])<=known and len(CATALOG[PROFILE+'_nodes'])==5



def test_install_is_complete(monkeypatch):
    r,captured,html=start_install(monkeypatch,{'profile':PROFILE,'precision':'int8'})
    assert r.status_code==200
    assert captured['precision']=='fp8' and captured['selfism_profile']==PROFILE
    assert {f['id'] for f in captured['files']}==set(CATALOG[PROFILE+'_files'])
    assert {n['name'] for n in captured['custom_nodes']}==set(CATALOG[PROFILE+'_nodes'])
    assert captured['model_links']==CATALOG[PROFILE+'_links']
    assert captured['selfism_repair'] and captured['selfism_repair_paths_only']
    for item in captured['model_links']:
        assert item['source'] in {f['destination'] for f in captured['files']}
    assert f'data-sf-action="{PROFILE}"' in html
    assert html.index('<h2>6gsc</h2>') < html.index('<h2>Millie Pose Recreate (Krea2 Ostris + DWPose)</h2>')
    with TestClient(m.app) as client:
        response=client.get('/api/selfism/workflow/'+PROFILE)
        assert response.json()==WORKFLOW
        assert 'Krea2_PoseRecreate_Millie_v1.json' in response.headers['content-disposition']


def test_installer_saves_workflow_without_overwriting_user_edits(monkeypatch,tmp_path):
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
    other=folder/'6gsc_dashboard_v1.json';other.write_text('existing 6gsc')
    wf={'selfism_profile':PROFILE,'precision':'fp8','files':[{'id':'x'}],'selfism_repair':True,'selfism_repair_paths_only':True}
    asyncio.run(ctrl._install_workflow(wf))
    path=folder/'Krea2_PoseRecreate_Millie_v1.json'
    assert path.read_bytes()==PATH.read_bytes()
    path.write_text('user edit')
    asyncio.run(ctrl._install_workflow(wf))
    assert path.read_text()=='user edit' and other.read_text()=='existing 6gsc'

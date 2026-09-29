import asyncio
import importlib
import json
import sys
from pathlib import Path

import pytest
from launcher.selfism import prefer_rapidcache, resolve_rapidcache
from fastapi.testclient import TestClient

m=importlib.import_module('launcher.app')

@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setenv('SELFISM_AUTO_REPAIR','0')
    for c,attr in [(m.controller,'task'),(m.custom_model_controller,'worker_task'),
                   (m.custom_node_controller,'worker_task'),(m.comfy_service_controller,'task'),
                   (m.selfism_controller,'task')]: monkeypatch.setattr(c,attr,None)

def test_original_and_new_ui_available():
    with TestClient(m.app) as c:
        html=c.get('/').text
        for name in ('selfism','workflows','custom-models','custom-nodes','account','docs'):
            assert f'data-view="{name}"' in html
        assert c.get('/api/selfism').status_code==200
        assert c.get('/selfism.js').status_code==200
        assert c.get('/selfism.css').status_code==200

def test_packaged_workflows_and_trigger():
    with TestClient(m.app) as c:
        for profile in ('simple','aio'):
            w=c.get('/api/selfism/workflow/'+profile).json()
            llm=next(n for n in w['nodes'] if n['type']=='ArtfatLLMPrompter')
            assert llm['widgets_values'][12].startswith('m1lli3,')
            assert llm['widgets_values'][6] is True
            assert llm['widgets_values'][0]=='RVN-Q4_K_M-multilingual-mtp.gguf'
        assert c.get('/api/selfism/workflow/unknown').status_code==404

def test_no_arbitrary_shell_or_profile():
    with TestClient(m.app) as c:
        assert c.post('/api/selfism/install',json={'profile':'rm -rf /'}).status_code==422
        assert c.post('/api/selfism/install',json={'precision':'../../tmp'}).status_code==422

def test_running_selfism_blocks_original_installers(monkeypatch):
    class Running:
        def done(self): return False
    monkeypatch.setattr(m.selfism_controller,'task',Running())
    c=TestClient(m.app)
    for url,body in [('/api/install/krea-2-extended',None),
                     ('/api/custom-nodes',{'url':'https://github.com/rgthree/rgthree-comfy'}),
                     ('/api/custom-models',{'url':'https://huggingface.co/test/model','location':'vae'})]:
        assert c.post(url,json=body).status_code==409

def test_gpu_queue_blocks_mutation(monkeypatch):
    import httpx
    class Client:
        def __init__(self,**kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def get(self,url,**kwargs):
            return httpx.Response(200,json={'queue_running':[1],'queue_pending':[]},request=httpx.Request('GET',url))
    monkeypatch.setattr(httpx,'AsyncClient',Client)
    with TestClient(m.app) as c:
        r=c.post('/api/selfism/install',json={'profile':'repair'})
        assert r.status_code==409
        assert 'generation' in r.json()['detail']

def test_repair_request_starts_fixed_job(monkeypatch):
    import httpx
    class Client:
        def __init__(self,**kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def get(self,url,**kwargs):
            return httpx.Response(200,json={'queue_running':[],'queue_pending':[]},request=httpx.Request('GET',url))
    captured={}
    async def start(workflow): captured.update(workflow); return {'status':'running'}
    monkeypatch.setattr(httpx,'AsyncClient',Client)
    monkeypatch.setattr(m.selfism_controller,'start',start)
    with TestClient(m.app) as c:
        assert c.post('/api/selfism/install',json={'profile':'repair'}).status_code==200
    assert captured['selfism_repair'] is True
    assert captured['files']==[] and captured['custom_nodes']==[]

def test_streaming_runner_timeout_and_cancel():
    async def run():
        ctrl=type(m.selfism_controller)()
        rc,text=await ctrl._run_process(sys.executable,'-u','-c','print("probe output")',timeout=5)
        assert rc==0 and 'probe output' in text
        with pytest.raises(RuntimeError,match='timed out'):
            await ctrl._run_process(sys.executable,'-c','import time; time.sleep(20)',timeout=.1)
        ctrl.cancel_event.set()
        with pytest.raises(m.InstallCancelled):
            await ctrl._run_process(sys.executable,'-c','import time; time.sleep(20)',timeout=5)
    asyncio.run(run())

def test_git_remote_stdout_is_preserved_but_ui_log_is_redacted(tmp_path):
    import subprocess
    repo='https://github.com/rgthree/rgthree-comfy.git'
    subprocess.run(['git','init',str(tmp_path)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(tmp_path),'remote','add','origin',repo],check=True)
    async def run():
        rc, output=await m.selfism_controller._run_process(
            'git','-C',str(tmp_path),'remote','get-url','origin',timeout=5)
        assert rc==0
        assert m.normalized_git_remote(output)==m.normalized_git_remote(repo)
        with TestClient(m.app) as client:
            logs=client.get('/api/selfism').json()['log']
        assert logs[-1]==m.redacted_for_export(repo)
        assert repo not in logs[-1]
    asyncio.run(run())

def test_catalog_destinations_and_node_pins():
    data=json.loads((Path(__file__).resolve().parents[1]/'catalog/selfism.json').read_text())
    for f in data['files'].values():
        assert f['destination'].startswith('models/') and '..' not in f['destination']
        assert f['url'].startswith('https://')
        assert f.get('sha256') or f.get('civitai_version') or 'github.com/starinspace/' in f['url']
    for n in data['nodes']:
        assert len(n['ref'])==40 and int(n['ref'],16)>0

def test_rapidcache_preserves_workflow_path_and_enforces_hash():
    original = {'name':'encoder','url':'https://huggingface.co/original',
                'destination':'models/text_encoders/local.safetensors',
                'sha256':'a'*64,'size_bytes':123,'auth':'huggingface'}
    remote = dict(original, url='https://cache.example/file?signature=secret',
                  destination='models/wrong/path',auth='none',parallel=True,verify=False)
    result, logs = prefer_rapidcache([original], {'workflows':[{'files':[remote]}]})
    assert result[0]['url'] == remote['url']
    assert result[0]['destination'] == original['destination']
    assert result[0]['verify'] is True and result[0]['auth']=='none'
    assert result[0]['sha256'] == original['sha256']
    assert original['url']=='https://huggingface.co/original'
    assert 'RapidCache' in logs[0] and 'secret' not in str(logs)

@pytest.mark.parametrize('change', [
    {'sha256':'b'*64}, {'sha256':''}, {'size_bytes':124},
    {'parallel':False}, {'parallel':'true'}, {'url':'http://cache.example/file'},
    {'auth':'civitai'},
])
def test_rapidcache_mismatch_or_nonaccelerated_uses_original(change):
    original={'name':'same-name','url':'https://original.example/model',
              'destination':'models/vae/test','sha256':'a'*64,'size_bytes':123}
    remote=dict(original,parallel=True,auth='none',url='https://cache.example/model')
    remote.update(change)
    result, logs=prefer_rapidcache([original],{'workflows':[{'files':[remote]}]})
    assert result==[original]
    assert 'original source' in logs[0]

def test_rapidcache_outage_is_nonfatal_and_requests_fresh_links(monkeypatch):
    calls=[]
    def fetch(*,fresh):
        calls.append(fresh)
        raise RuntimeError('private signed URL must not be logged')
    monkeypatch.setattr(m.remote,'fetch_catalog',fetch)
    original=[{'url':'https://original.example/model','destination':'models/test'}]
    result,logs=asyncio.run(resolve_rapidcache(m,original))
    assert calls==[True] and result==original
    assert 'unavailable' in logs[0] and 'private signed' not in str(logs)
    asyncio.run(resolve_rapidcache(m,[]))
    assert calls==[True]


def test_reference_installer_uses_turbo_not_selfora_and_includes_depth_weights(monkeypatch):
    import httpx
    class Client:
        def __init__(self,**kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def get(self,url,**kwargs):
            assert url.endswith('/queue')  # No Civitai token required for Turbo.
            return httpx.Response(200,json={'queue_running':[],'queue_pending':[]},request=httpx.Request('GET',url))
    captured={}
    async def start(workflow): captured.update(workflow); return {'status':'running'}
    monkeypatch.setattr(httpx,'AsyncClient',Client)
    monkeypatch.setattr(m.selfism_controller,'start',start)
    with TestClient(m.app) as c:
        assert c.post('/api/selfism/install',json={'profile':'reference','precision':'bf16'}).status_code==200
        w=c.get('/api/selfism/workflow/reference').json()
        assert 'data-sf-action="reference"' in c.get('/').text
    files={f['id']:f for f in captured['files']}
    assert 'krea-turbo' in files and not {'bf16','int8','fp8'} & files.keys()
    assert {'depth','depth-anything','nmkd','sam','face','bloom','famegrid','qwen-vae','llm','mmproj','encoder'} <= files.keys()
    assert captured['selfism_repair'] is True
    assert captured['model_links'][0]['source']==files['depth-anything']['destination']
    assert any(n['name']=='ComfyUI-Artfat-Resolution' for n in captured['custom_nodes'])
    # All static loader files, including optional enhancement models, have catalog coverage.
    paths={Path(f['destination']).name for f in files.values()}
    for n in w['nodes']:
        if n['type'] in ('UNETLoader','VAELoader','CLIPLoader','UpscaleModelLoader','SAMLoader','UltralyticsDetectorProvider','Krea2ControlLoRALoader','DepthAnythingV2Preprocessor'):
            assert n['widgets_values'][0].split('/')[-1] in paths
        if n['type']=='Power Lora Loader (rgthree)':
            for row in n['widgets_values']:
                if isinstance(row,dict) and row.get('on'): assert row['lora'] in paths
    assert next(n for n in w['nodes'] if n['type']=='UNETLoader')['widgets_values'][0]=='krea2_turbo_fp8_scaled.safetensors'


def test_r2_inventory_includes_reference_models(tmp_path):
    from scripts.upload_models_r2 import inventory
    catalog=json.loads((Path(__file__).resolve().parents[1]/'catalog/selfism.json').read_text())
    expected=set()
    for key in catalog['reference_files']:
        p=tmp_path/catalog['files'][key]['destination']; p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'test');expected.add(p)
    found,missing=inventory(catalog,tmp_path)
    assert {p for p,spec in found}==expected
    assert len(found)==len(expected)

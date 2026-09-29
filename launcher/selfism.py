"""Selfism add-on using the original transfer, node and restart machinery."""
from __future__ import annotations
import asyncio
import copy
import json
import os
import signal
from collections import deque
from pathlib import Path
from typing import Literal

import httpx
from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

class Selection(BaseModel):
    profile: Literal['simple','aio','extras','repair','model','node'] = 'simple'
    precision: Literal['fp8','int8','bf16'] = 'fp8'
    item: str = ''

def register(host):
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root/'catalog/selfism.json').read_text(encoding='utf-8'))
    logs = deque(maxlen=1500)

    class SelfismController(host.JobController):
        async def _run_process(self, *command, timeout=None, env=None):
            # Stream bounded, redacted output for node installers and environment repair.
            process = await asyncio.create_subprocess_exec(*map(str,command),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                env=env, start_new_session=(os.name=='posix'))
            output=deque(maxlen=1000)
            async def reader():
                while line:=await process.stdout.readline():
                    text=host.redacted_for_export(line.decode(errors='replace').rstrip())
                    logs.append(text); output.append(text)
                return await process.wait()
            read_task=asyncio.create_task(reader())
            cancel_task=asyncio.create_task(self.cancel_event.wait())
            try:
                done,_=await asyncio.wait([read_task,cancel_task],timeout=timeout or 1800,
                                          return_when=asyncio.FIRST_COMPLETED)
                if read_task in done: return read_task.result(), '\n'.join(output)
                if os.name=='posix': os.killpg(process.pid,signal.SIGTERM)
                else: process.terminate()
                try: await asyncio.wait_for(process.wait(),10)
                except asyncio.TimeoutError:
                    if os.name=='posix': os.killpg(process.pid,signal.SIGKILL)
                    else: process.kill()
                    await process.wait()
                if self.cancel_event.is_set(): raise host.InstallCancelled()
                raise RuntimeError('Installation command timed out.')
            finally:
                if process.returncode is None:
                    if os.name=='posix': os.killpg(process.pid,signal.SIGKILL)
                    else: process.kill()
                    await process.wait()
                for task in (read_task,cancel_task):
                    if not task.done(): task.cancel()
                await asyncio.gather(read_task,cancel_task,return_exceptions=True)

        async def _install_workflow(self, workflow):
            await self._wait_for_comfyui()
            python=host.COMFYUI_VENV/'bin/python'
            if not python.exists(): raise RuntimeError('ComfyUI Python environment is not ready.')
            # Protect core packages while third-party requirements are installed.
            code=('import importlib.metadata as m; '
                  'print("\\n".join(n+"=="+m.version(n) for n in '
                  '["torch","torchvision","torchaudio","numpy","transformers"]))')
            rc,output=await self._run_process(python,'-c',code,timeout=30)
            if rc: raise RuntimeError('Could not record core package versions: '+output[-500:])
            constraints=host.COMFYUI_DIR/'.selfism-constraints.txt'
            constraints.write_text(output+'\n')
            previous=os.environ.get('PIP_CONSTRAINT')
            # Preserve base constraints as well as the snapshot.
            if previous and Path(previous).is_file():
                constraints.write_text(constraints.read_text()+Path(previous).read_text())
            os.environ['PIP_CONSTRAINT']=str(constraints)
            try:
                if workflow.get('files') or workflow.get('custom_nodes'):
                    await super()._install_workflow(workflow)
                if self.state.warnings:
                    raise RuntimeError('Some dependencies failed. See warnings and retry before running the workflow.')
                if workflow.get('selfism_repair'):
                    self.update(stage='installing',message='Repairing Qwen / CUDA environment…',percent=96)
                    rc,output=await self._run_process(python,'-u',root/'launcher/selfism_runtime.py',timeout=2100)
                    if rc: raise RuntimeError('Environment repair failed: '+output[-1500:])
                profile=workflow.get('selfism_profile')
                if profile in ('simple','aio'):
                    data=json.loads((root/'selfism_workflows'/f'{profile}.json').read_text(encoding='utf-8'))
                    for n in data['nodes']:
                        if n['type']=='UNETLoader': n['widgets_values'][0]=Path(catalog['files'][workflow['precision']]['destination']).name
                    folder=host.COMFYUI_DIR/'user/default/workflows/Selfism'
                    folder.mkdir(parents=True,exist_ok=True)
                    # Keep the user's previously edited workflow instead of overwriting it.
                    dest=folder/f'Selfism_{profile}_{workflow["precision"]}_m1lli3.json'
                    if not dest.exists(): dest.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
                    logs.append('Workflow saved: '+str(dest))
                    logs.append('Upload your reference image. Add your private Millie LoRA separately and enable its row.')
            finally:
                if previous is None: os.environ.pop('PIP_CONSTRAINT',None)
                else: os.environ['PIP_CONSTRAINT']=previous

    controller=SelfismController()
    host.selfism_controller=controller

    def busy():
        return any(t and not t.done() for t in (host.controller.task,
            host.custom_node_controller.worker_task,host.custom_model_controller.worker_task,
            host.comfy_service_controller.task,controller.task))

    async def ensure_idle():
        if busy(): raise HTTPException(409,'Wait for the current download, install or restart to finish.')
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response=await client.get(host.COMFYUI_LOCAL_URL+'/queue')
                response.raise_for_status(); q=response.json()
            if q.get('queue_running') or q.get('queue_pending'):
                raise HTTPException(409,'Finish or clear the ComfyUI generation queue first.')
        except httpx.HTTPError:
            raise HTTPException(409,'ComfyUI is not ready. Wait for startup and retry.')

    @host.app.get('/api/selfism')
    async def snapshot():
        return {'catalog':catalog,'job':controller.state.export(),'log':list(logs)}

    @host.app.post('/api/selfism/install')
    async def install(request:Selection):
        await ensure_idle()
        profile=request.profile
        file_keys=[]; nodes=[]
        if profile in ('simple','aio'):
            file_keys=[request.precision]+catalog['simple_files']
            nodes=[n for n in catalog['nodes'] if profile=='aio' or n['name'] in catalog['simple_nodes']]
            if profile=='aio': file_keys+=catalog['aio_files']
        elif profile=='extras': file_keys=catalog['extra_files']
        elif profile=='model':
            if request.item not in catalog['files']: raise HTTPException(404,'Unknown model.')
            file_keys=[request.item]
        elif profile=='node':
            nodes=[n for n in catalog['nodes'] if n['name']==request.item]
            if not nodes: raise HTTPException(404,'Unknown node.')
        files=[copy.deepcopy(catalog['files'][k]) for k in file_keys]
        # Resolve exact Civitai metadata before accepting a multi-GB transfer.
        async with httpx.AsyncClient(timeout=30,follow_redirects=True) as client:
            for f in files:
                if 'civitai_version' not in f: continue
                token=(os.getenv('CIVITAI_TOKEN') or os.getenv('CIVITAI_API_TOKEN') or '').strip()
                if not token:
                    raise HTTPException(400,'Add CIVITAI_TOKEN in the RunPod template environment for Selfora downloads. Do not paste the token into a public URL.')
                headers={'Authorization':'Bearer '+token} if token else {}
                try:
                    r=await client.get(f'https://civitai.com/api/v1/model-versions/{f["civitai_version"]}',headers=headers)
                    r.raise_for_status()
                    asset=next(x for x in r.json()['files'] if x['id']==f['civitai_file'])
                    f['sha256']=asset['hashes']['SHA256'].lower()
                except (httpx.HTTPError,KeyError,StopIteration):
                    raise HTTPException(502,'Cannot verify Selfora download metadata. Check Civitai access / CIVITAI_TOKEN and retry.')
        # Recheck after the network await so two simultaneous clicks cannot race.
        if busy(): raise HTTPException(409,'Another installation started; retry later.')
        logs.clear()
        workflow={'id':'selfism-'+profile,'title':'Selfora / Selfism — '+profile,
                  'files':files,'custom_nodes':nodes,'selfism_profile':profile,
                  'precision':request.precision,'selfism_repair':profile in ('simple','aio','repair','node')}
        return await controller.start(workflow)

    @host.app.post('/api/selfism/cancel')
    async def cancel(): return await controller.cancel()

    @host.app.get('/api/selfism/workflow/{profile}')
    async def workflow_file(profile:str):
        if profile not in ('simple','aio'): raise HTTPException(404)
        return FileResponse(root/'selfism_workflows'/f'{profile}.json',filename=f'Selfism_{profile}_m1lli3.json')

    async def startup_repair():
        # Called only by the opt-in image setting, after Comfy's base startup.
        for _ in range(120):
            try:
                await install(Selection(profile='repair'))
                return
            except HTTPException as exc:
                if exc.status_code != 409:
                    logs.append('Automatic repair: '+str(exc.detail)); return
            await asyncio.sleep(5)
        logs.append('Automatic repair deferred: ComfyUI stayed busy or unavailable. Use the repair button.')
    controller.startup_repair=startup_repair

    return controller

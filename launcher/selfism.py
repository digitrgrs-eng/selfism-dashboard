"""Selfism add-on using the original transfer, node and restart machinery."""
from __future__ import annotations
import asyncio
import copy
import json
import os
import signal
import re
import shutil
import uuid
import hashlib
from urllib.parse import urlsplit
from collections import deque
from pathlib import Path
from typing import Literal

import httpx
from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from launcher.private_r2 import r2_client, select_private

# Profiles that need a ComfyUI able to load Selfora INT8 (int8_tensorwise).
INT8_PROFILES = ('simple','aio','full','repair')
MINIMAX_WORKFLOW_NAME = 'Simply_Advanced_MiniMax_H3_v1.4.json'
MINIMAX_R2V_WORKFLOW_NAME = 'MiniMax_H3_R2V_Turbo_Hearmeman.json'

class Selection(BaseModel):
    profile: Literal['simple','aio','reference','carousel','full','minimax','minimax_r2v','extras','repair','model','node'] = 'simple'
    precision: Literal['fp8','int8','bf16'] = 'fp8'
    item: str = ''

def prefer_rapidcache(files, remote_catalog):
    """Use only exact, accelerated matches; keep local paths and verification."""
    candidates = {}
    for workflow in (remote_catalog or {}).get('workflows', []):
        for item in workflow.get('files', []):
            digest = str(item.get('sha256', '')).lower()
            url = str(item.get('url', ''))
            if (re.fullmatch(r'[0-9a-f]{64}', digest)
                    and item.get('parallel') is True
                    and urlsplit(url).scheme == 'https'
                    and item.get('auth', 'none') in ('none', '', None)
                    and type(item.get('size_bytes')) is int and item['size_bytes'] > 0):
                candidates.setdefault(digest, []).append(item)
    selected, messages = [], []
    for original in files:
        result = copy.deepcopy(original)
        matches = candidates.get(str(original.get('sha256', '')).lower(), [])
        match = next((item for item in matches if not original.get('size_bytes')
                      or item['size_bytes'] == original['size_bytes']), None)
        name = original.get('name') or Path(original['destination']).name
        if match:
            result.update(url=match['url'], auth='none', parallel=True,
                          size_bytes=match['size_bytes'], verify=True)
            messages.append(f'{name}: RapidCache (identical SHA256).')
        else:
            messages.append(f'{name}: original source (no verified accelerated match).')
        selected.append(result)
    return selected, messages

async def resolve_rapidcache(host, files):
    if not files:
        return files, []
    # Reuse the original RapidCache account; do not forward its token to downloads.
    try:
        data = await asyncio.to_thread(host.remote.fetch_catalog, fresh=True)
        if data is not None:
            host._validate_catalog(data)
    except Exception:
        data = None
    selected, messages = prefer_rapidcache(files, data)
    if data is None:
        messages.insert(0, 'RapidCache catalog unavailable; using original sources.')
    return selected, messages

async def resolve_sources(host, files):
    if not files:
        return files, []
    try:
        client,bucket,notice=await asyncio.to_thread(r2_client)
        preferred,messages=await asyncio.to_thread(select_private,files,client,bucket)
        if notice: messages.insert(0,notice)
    except Exception:
        preferred,messages={},['Private R2 unavailable; using fallback sources.']
    remaining=[i for i in range(len(files)) if i not in preferred]
    fallback,notes=await resolve_rapidcache(host,[files[i] for i in remaining])
    for i, selected in preferred.items():
        selected['_selfism_source']='R2'
        selected['_selfism_original']=copy.deepcopy(files[i])
    for i, selected in zip(remaining, fallback):
        if selected.get('url') != files[i].get('url'):
            selected['_selfism_source']='RapidCache'
            selected['_selfism_original']=copy.deepcopy(files[i])
    preferred.update(zip(remaining,fallback))
    return [preferred[i] for i in range(len(files))], messages+notes

def register(host):
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root/'catalog/selfism.json').read_text(encoding='utf-8'))
    logs = deque(maxlen=1500)

    class SelfismController(host.JobController):
        async def _download_file(self, client, file_spec, *args):
            original=file_spec.get('_selfism_original', file_spec)
            current=copy.deepcopy(file_spec)
            source=current.pop('_selfism_source', 'original')
            current.pop('_selfism_original', None)
            while True:
                self.check_cancelled()
                try:
                    return await super()._download_file(client, current, *args)
                except host.InstallCancelled:
                    raise
                except (httpx.HTTPError, RuntimeError) as exc:
                    self.check_cancelled()
                    # Keep signed URLs out of both UI errors and transfer diagnostics.
                    message=host.redacted_for_export(str(exc))
                    host.diagnostics.fail_in_flight(message)
                    if source == 'original':
                        raise RuntimeError(message) from None
                    # A corrupt partial must not be resumed against the next source.
                    # Network interruptions keep their valid partial data for resume.
                    if 'checksum' in str(exc).lower() or 'wrong size' in str(exc).lower():
                        self._discard_failed_partials(current)
                    logs.append(f"{original.get('name', 'Model')}: {source} transfer failed; trying next source.")
                    if source == 'R2':
                        candidates, notes=await resolve_rapidcache(host, [copy.deepcopy(original)])
                        logs.extend(notes)
                        current=candidates[0]
                        source='RapidCache' if current.get('url') != original.get('url') else 'original'
                    else:
                        current=copy.deepcopy(original)
                        source='original'
                    self.update(message=f"Retrying {original.get('name', 'model')} from {source}…",
                                bytes_per_second=0)

        def _discard_failed_partials(self, file_spec):
            destination=host.safe_destination(file_spec['destination'])
            paths=[destination.with_name(destination.name+'.part')]
            scratch=host.scratch_dir()
            if scratch is not None:
                stem=hashlib.sha256(str(destination).encode('utf-8')).hexdigest()[:16]
                paths.append(scratch/f'{stem}-{destination.name}.part')
            for path in paths:
                path.unlink(missing_ok=True)
                path.with_name(path.name+'.aria2').unlink(missing_ok=True)

        async def _install_carousel_helper(self, python):
            source=root/'bundled_nodes/ComfyUI-AIO-Carousel'
            self.update(stage='installing', message='Installing AIO Qwen Carousel helper nodes…', percent=95)
            rc, output=await self._run_process(python, '-m', 'pip', 'install',
                                             '-r', source/'requirements.txt', timeout=900)
            if rc: raise RuntimeError('Carousel dependencies failed: '+output[-1500:])
            rc, output=await self._run_process(python, '-c',
                'import color_matcher, ultralytics; print("Carousel dependencies ready")', timeout=120)
            if rc: raise RuntimeError('Carousel dependency import failed: '+output[-1500:])
            self.check_cancelled()
            target=host.COMFYUI_DIR/'custom_nodes/ComfyUI-AIO-Carousel'
            if target.exists():
                backup=host.COMFYUI_DIR/'user/carousel_backups'/uuid.uuid4().hex
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(target, backup)
                logs.append('Previous carousel helper saved: '+str(backup))
            shutil.copytree(source, target, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            logs.append('Bundled AIO Qwen Carousel helper installed in the ComfyUI environment.')

        async def _run_process(self, *command, timeout=None, env=None):
            # Stream bounded, redacted output for node installers and environment repair.
            process = await asyncio.create_subprocess_exec(*map(str,command),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                env=env, start_new_session=(os.name=='posix'))
            output=deque(maxlen=1000)
            async def reader():
                while line:=await process.stdout.readline():
                    text=line.decode(errors='replace').rstrip()
                    # The caller parses stdout (Git origin URLs, version pins).
                    # Redact only the UI copy, never the machine-readable result.
                    logs.append(host.redacted_for_export(text)); output.append(text)
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

        async def _install_python_packages(self, python, packages):
            # Some node packs import a package at load time without declaring it (comfyui-various does
            # `import soundfile` at the top of comfyui_sound.py and has no requirements.txt), so the whole pack fails
            # to import. PIP_CONSTRAINT (torch/numpy/...) is already active here, so this cannot move the core stack.
            safe=[p for p in packages if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*(==[A-Za-z0-9.*+-]+)?',str(p))]
            if len(safe)!=len(packages): raise RuntimeError('Unsafe Python package name in catalog.')
            self.update(stage='installing',message='Installing Python packages for node packs: '+', '.join(safe),percent=97)
            rc,output=await self._run_process(python,'-m','pip','install','--timeout','15','--retries','3',*safe,timeout=900)
            if rc: raise RuntimeError('Python packages failed ('+', '.join(safe)+'): '+output[-500:])
            for name in safe:
                module=name.split('==')[0].replace('-','_')
                rc,output=await self._run_process(python,'-c','import '+module,timeout=60)
                if rc: raise RuntimeError('Installed '+name+' but it cannot be imported: '+output[-500:])
            logs.append('Python packages installed for node packs: '+', '.join(safe)+'. Restart ComfyUI.')

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
                if workflow.get('selfism_profile') in INT8_PROFILES:
                    # First, so the restart after the install loads the new ComfyUI.
                    self.update(stage='installing',message='Checking ComfyUI INT8 support…',percent=1)
                    rc,output=await self._run_process(python,'-u',root/'launcher/selfism_int8.py',
                        '--comfy-dir',host.COMFYUI_DIR,'--stash',timeout=2400)
                    if rc: raise RuntimeError('ComfyUI INT8 upgrade failed: '+output[-1500:])
                    self.check_cancelled()
                if workflow.get('files') or workflow.get('custom_nodes'):
                    workflow = copy.deepcopy(workflow)
                    self.update(message='Checking private R2 and RapidCache for identical models…')
                    workflow['files'], source_messages = await resolve_sources(host, workflow.get('files', []))
                    logs.extend(source_messages)
                    self.check_cancelled()
                    await super()._install_workflow(workflow)
                if self.state.warnings:
                    raise RuntimeError('Some dependencies failed. See warnings and retry before running the workflow.')
                if workflow.get('pip_packages'):
                    await self._install_python_packages(python,workflow['pip_packages'])
                if workflow.get('selfism_profile') == 'carousel':
                    await self._install_carousel_helper(python)
                if workflow.get('selfism_repair'):
                    self.update(stage='installing',message='Repairing Qwen / CUDA environment…',percent=96)
                    rc,output=await self._run_process(python,'-u',root/'launcher/selfism_runtime.py',timeout=2100)
                    if rc: raise RuntimeError('Environment repair failed: '+output[-1500:])
                profile=workflow.get('selfism_profile')
                if profile=='minimax':
                    # Third-party workflow shipped byte-for-byte: no patching and no re-serialisation.
                    folder=host.COMFYUI_DIR/'user/default/workflows/Selfism'
                    folder.mkdir(parents=True,exist_ok=True)
                    dest=folder/MINIMAX_WORKFLOW_NAME
                    if not dest.exists(): dest.write_bytes((root/'selfism_workflows/minimax_h3_simply_advanced.json').read_bytes())
                    logs.append('Workflow saved: '+str(dest))
                    logs.append('Backend Attention defaults to "sage attention"; if SageAttention is not installed, choose "pytorch attention" in that subgraph.')
                if profile=='minimax_r2v':
                    # HearmemanAI R2V Turbo workflow: shipped as-is (only the UNETLoader file name was made to match the installed file).
                    folder=host.COMFYUI_DIR/'user/default/workflows/Selfism'
                    folder.mkdir(parents=True,exist_ok=True)
                    dest=folder/MINIMAX_R2V_WORKFLOW_NAME
                    if not dest.exists(): dest.write_bytes((root/'selfism_workflows/minimax_h3_r2v_turbo_hearmeman.json').read_bytes())
                    logs.append('Workflow saved: '+str(dest))
                    logs.append('The MiniMax References Manager node writes the prompt through OpenRouter by default: set OPENROUTER_API_KEY (or LLM_KEY) in the pod environment, or choose prompt_provider "none" in that node. The Power Lora row "hmmotion_minimax-h3_epoch40" is on by default but optional and not installed; it is skipped.')
                if profile in ('simple','aio','reference','carousel','full'):
                    data=json.loads((root/'selfism_workflows'/f'{profile}.json').read_text(encoding='utf-8'))
                    for n in data['nodes']:
                        if n['type']=='UNETLoader' and profile in ('simple','aio','full'): n['widgets_values'][0]=Path(catalog['files'][workflow['precision']]['destination']).name
                        if profile in ('reference','carousel') and n['type']=='Power Lora Loader (rgthree)':
                            for row in n.get('widgets_values', []):
                                if isinstance(row,dict) and row.get('lora')=='millie_000002750.safetensors':
                                    for name in ('millie_000002750.safetensors','millie.safetensors'):
                                        if (host.COMFYUI_DIR/'models/loras'/name).is_file():
                                            row.update(lora=name,on=True); break
                    folder=host.COMFYUI_DIR/'user/default/workflows/Selfism'
                    folder.mkdir(parents=True,exist_ok=True)
                    # Keep the user's previously edited workflow instead of overwriting it.
                    filename={'reference':'10sorlabs_MILLIE_REFERENCE_DEPTH_v1.json',
                              'carousel':'Selfism_AIO_m1lli3_CAROUSEL_QWEN_2511_v1.json',
                              'full':f'Selfism_FULL_{workflow["precision"]}_recreate_v1.json'}.get(profile,
                              f'Selfism_{profile}_{workflow["precision"]}_m1lli3.json')
                    dest=folder/filename
                    if not dest.exists(): dest.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
                    logs.append('Workflow saved: '+str(dest))
                    if profile=='full':
                        # The Recreate workflow selects these LLM system presets by file name. A preset file on the pod
                        # overrides the text baked into the node, so keep the shipped files in sync with the workflow.
                        # Recreate_{SFW,NSFW}_{prefix,noprefix}.txt (v2, identity-safe) and
                        # Recreate_FirstFrame_{SFW,NSFW}_{prefix,noprefix}.txt (reel first frame) match this glob.
                        prompts=host.COMFYUI_DIR/'models/LLM/prompts'
                        prompts.mkdir(parents=True,exist_ok=True)
                        for preset in sorted((root/'selfism_workflows/presets').glob('Recreate_*.txt')):
                            (prompts/preset.name).write_bytes(preset.read_bytes())
                        logs.append('Recreate + Recreate FirstFrame LLM presets copied to '+str(prompts)+'. Restart ComfyUI if the prompter does not list them.')
                    if profile=='full': logs.append('Upload your reference image. Millie LoRA is installed (first row of the Power Lora Loader).')
                    else: logs.append('Upload your reference image. Add your private Millie LoRA separately and enable its row.')
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
        elif profile=='reference':
            file_keys=catalog['reference_files']
            nodes=[n for n in catalog['nodes'] if n['name'] in catalog['reference_nodes']]
        elif profile=='carousel':
            file_keys=catalog['carousel_files']
            nodes=[n for n in catalog['nodes'] if n['name'] in catalog['carousel_nodes']]
        elif profile=='full':
            # Kompletan workflow: only INT8 (default) or FP8; BF16 is intentionally not offered.
            precision=request.precision if 'precision' in request.model_fields_set else 'int8'
            if precision not in ('int8','fp8'):
                raise HTTPException(400,'Kompletan workflow podržava samo Selfora INT8 ili FP8.')
            request=request.model_copy(update={'precision':precision})
            file_keys=[precision]+catalog['full_files']
            nodes=[n for n in catalog['nodes'] if n['name'] in catalog['full_nodes']]
        elif profile=='minimax':
            # MiniMax H3 Simply Advanced: exactly the default-active models of the workflow (R2 first, HF fallback).
            file_keys=catalog['minimax_files']
            nodes=[n for n in catalog['nodes'] if n['name'] in catalog['minimax_nodes']]
        elif profile=='minimax_r2v':
            # MiniMax H3 R2V Turbo (HearmemanAI): INT8 Ref2VA + INT8 encoder + FP16/audio VAEs + turbo LoRA + TAEH3 preview decoder.
            file_keys=catalog['r2v_files']
            nodes=[n for n in catalog['nodes'] if n['name'] in catalog['r2v_nodes']]
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
                  'precision':'fp8' if profile=='carousel' else request.precision,
                  'selfism_repair':profile in ('simple','aio','reference','carousel','full','repair','node'),
                  'pip_packages':list(catalog.get('minimax_pip',[])) if profile=='minimax' else [],
                  'model_links':(copy.deepcopy(catalog['full_links']) if profile=='full' else copy.deepcopy(catalog['reference_links'])) if profile in ('reference','carousel','full') else []}
        return await controller.start(workflow)

    @host.app.post('/api/selfism/cancel')
    async def cancel(): return await controller.cancel()

    @host.app.get('/api/selfism/workflow/{profile}')
    async def workflow_file(profile:str):
        if profile not in ('simple','aio','reference','carousel','full','minimax','minimax_r2v'): raise HTTPException(404)
        if profile=='minimax_r2v':
            return FileResponse(root/'selfism_workflows/minimax_h3_r2v_turbo_hearmeman.json',filename=MINIMAX_R2V_WORKFLOW_NAME)
        if profile=='minimax':
            return FileResponse(root/'selfism_workflows/minimax_h3_simply_advanced.json',filename=MINIMAX_WORKFLOW_NAME)
        return FileResponse(root/'selfism_workflows'/f'{profile}.json',filename=('Selfism_FULL_workflow.json' if profile=='full' else f'Selfism_{profile}_m1lli3.json'))

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

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
INT8_PROFILES = ('simple','aio','full','full_refine','repair')
GOD_MODE_WORKFLOW_NAME = 'Wan22_Animate_GOD_Mode.json'

class Selection(BaseModel):
    profile: Literal['simple','aio','reference','6gsc','hearmeman_i2i','hearmeman_wan_v5','hearmeman_scail2_v2','qwen21_edit_full','recreate_hybrid','krast','krea_klein','krea_fluxdepth','krast_fluxdepth','krast_faithful_klein_v5','krast_org','rawgirl_millie','krast_org_depth','rawgirl','carousel','full','full_refine','god_mode','extras','repair','model','node'] = 'simple'
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
    missing_private=[files[i]['name'] for i in remaining if files[i].get('r2_required')]
    if missing_private:
        raise RuntimeError('Connect private R2 and verify these files before retrying: '+', '.join(missing_private))
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
                    if source == 'original' or original.get('r2_required'):
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

        async def _install_krast_nodes(self):
            # KRASTv3 Carousel grid node is local (not on git): shipped in the image under bundled_nodes/.
            self.update(stage='installing', message='Installing KRASTv3 Carousel node…', percent=95)
            for name in catalog['krast_bundled_nodes']:
                source=root/'bundled_nodes'/name
                target=host.COMFYUI_DIR/'custom_nodes'/name
                shutil.copytree(source, target, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            logs.append('KRASTv3 Carousel node installed: '+', '.join(catalog['krast_bundled_nodes'])+'.')

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
                if workflow.get('selfism_profile') == 'krast':
                    await self._install_krast_nodes()
                if workflow.get('selfism_repair'):
                    # Krast ORG graphs have no llama.cpp / GGUF node. Keep the CUDA library
                    # path fix and skip the 313 MB llama-cpp wheel and its GPU probe.
                    repair_args=[python,'-u',root/'launcher/selfism_runtime.py']
                    repair_timeout=2100
                    repair_message='Repairing Qwen / CUDA environment…'
                    if workflow.get('selfism_repair_paths_only'):
                        repair_args.append('--paths-only')
                        repair_timeout=180
                        repair_message='Configuring CUDA library paths…'
                    self.update(stage='installing',message=repair_message,percent=96)
                    rc,output=await self._run_process(*repair_args,timeout=repair_timeout)
                    if rc: raise RuntimeError('Environment repair failed: '+output[-1500:])
                profile=workflow.get('selfism_profile')
                if profile=='recreate_hybrid':
                    folder=host.COMFYUI_DIR/'user/default/workflows/Selfism'
                    folder.mkdir(parents=True,exist_ok=True)
                    dest=folder/'Selfism_Recreate_Hybrid.json'
                    if not dest.exists(): dest.write_bytes((root/'selfism_workflows/recreate_hybrid.json').read_bytes())
                    logs.append('Workflow saved: '+str(dest))
                    logs.append('Selfism Recreate Hybrid: load the reference scene image. Krea 2 Turbo BF16 + Millie LoRA, Qwen 3.5 9B scene prompt writer, Depth control (Depth Anything V2), Detail Daemon and optional SeedVR2 7B upscale are installed.')
                if profile=='krast':
                    folder=host.COMFYUI_DIR/'user/default/workflows/Selfism'
                    folder.mkdir(parents=True,exist_ok=True)
                    for src,name in (('krast_recreate.json','KRAST_Recreate.json'),('krast_carousel.json','KRAST_Carousel.json')):
                        dest=folder/name
                        if not dest.exists(): dest.write_bytes((root/'selfism_workflows'/src).read_bytes())
                        logs.append('Workflow saved: '+str(dest))
                    logs.append('KRAST Recreate + Carousel: KRAST-V2 + Millie LoRA, Artfat RVN prompter, Depth control, res_2s (RES4LYF), KRASTv3 Carousel grid node and optional SeedVR2 7B upscale are installed. Not GPU-validated yet.')
                if profile=='krea_klein':
                    folder=host.COMFYUI_DIR/'user/default/workflows/Selfism'
                    folder.mkdir(parents=True,exist_ok=True)
                    for src,name in (('krea_klein_artfat.json','Krea2_FluxKlein_Artfat_Reference.json'),('krea_klein_famegrid.json','Krea2_FluxKlein_Famegrid.json')):
                        dest=folder/name
                        if not dest.exists(): dest.write_bytes((root/'selfism_workflows'/src).read_bytes())
                        logs.append('Workflow saved: '+str(dest))
                    logs.append('Krea 2 + Flux Klein: Krea 2 Turbo BF16 + Famegrid Standard, Detail Daemon, Flux 2 Klein 9B skin pass, ColorMatch, Artfat RVN prompter (Artfat Reference variant), optional UltraReal LoRA and FaceDetailer (bypassed) are installed. Not GPU-validated yet.')
                if profile in ('krea_fluxdepth','krast_fluxdepth'):
                    folder=host.COMFYUI_DIR/'user/default/workflows/Selfism'
                    folder.mkdir(parents=True,exist_ok=True)
                    src,name=(('krea_fluxdepth_millie.json','Krea2_FluxKlein_Millie_Artfat_Depth_v3.json') if profile=='krea_fluxdepth' else ('krast_fluxdepth_millie.json','KRAST_FluxKlein_Millie_Artfat_Depth_v3.json'))
                    dest=folder/name
                    if not dest.exists(): dest.write_bytes((root/'selfism_workflows'/src).read_bytes())
                    logs.append('Workflow saved: '+str(dest))
                    logs.append(('Krea 2 Turbo BF16' if profile=='krea_fluxdepth' else 'KRAST-V2')+' FluxDepth Millie: Millie + Famegrid, Artfat RVN prompter, Krea2 depth control LoRA + Depth Anything V2, Detail Daemon, Flux 2 Klein 9B skin pass, ColorMatch, plus all Krea 2 realism/slider LoRAs (default OFF; add them in Power Lora Loader). Not GPU-validated yet.')
                if profile=='krast_org':
                    folder=host.COMFYUI_DIR/'user/default/workflows/Selfism'
                    folder.mkdir(parents=True,exist_ok=True)
                    dest=folder/'KRAST_ORG_Millie.json'
                    if not dest.exists(): dest.write_bytes((root/'selfism_workflows/krast_org.json').read_bytes())
                    logs.append('Workflow saved: '+str(dest))
                    logs.append('Krast ORG: author KRAST workflow (krast_v20 INT8, Qwen3-VL-4B Heretic INT8 ConvRot, krea2RealVae; er_sde 8 steps + 1.5x DPM++ SDE/Beta/APG pass; Millie LoRA 1.0 in the main Pixaroma LoRA loader; Load Image on the main graph (example.png default, i2i switch off); saves to output/KRAST_ORG/stage2, face, final; MoP Mix Lustify SDXL NSFW detailers). Not GPU-validated yet.')
                if profile=='rawgirl_millie':
                    folder=host.COMFYUI_DIR/'user/default/workflows/Selfism'
                    folder.mkdir(parents=True,exist_ok=True)
                    dest=folder/'RawGirl_Autor_Millie.json'
                    if not dest.exists(): dest.write_bytes((root/'selfism_workflows/rawgirl_millie.json').read_bytes())
                    logs.append('Workflow saved: '+str(dest))
                    logs.append('RawGirl Millie: author RawGirl v2.0 FP8 workflow (qwen3vl_4b_fp8_scaled krea2 encoder, qwen_image_vae, 1080x1920, ClownsharKSampler_Beta linear/euler + beta 8 steps, eta 0.5, CFG 1) plus Millie LoRA 0.8. Not GPU-validated yet.')
                if profile in ('krast_org_depth','rawgirl'):
                    folder=host.COMFYUI_DIR/'user/default/workflows/Selfism'/('KRAST_ORG_Depth' if profile=='krast_org_depth' else 'RawGirl')
                    folder.mkdir(parents=True,exist_ok=True)
                    for src in sorted((root/'selfism_workflows'/profile).glob('*.json')):
                        dest=folder/src.name
                        if not dest.exists(): dest.write_bytes(src.read_bytes())
                        logs.append('Workflow saved: '+str(dest))
                    logs.append(('Krast ORG + Depth: author KRAST ORG workflow, AutoRatio Depth (v1, IMG2IMG, SoftFace v2, each with ColorMatch) and KRAST Millie Carousel 2x2; Krea2 depth control LoRA + Depth Anything V2, KJNodes ColorMatch, AutoGrid.' if profile=='krast_org_depth' else 'RawGirl: author RawGirl Millie, NMKD Upscale v1, NMKD Depth ColorMatch v2 and RawGirl Millie Carousel 2x2; 4x NMKD Superscale, Krea2 depth control LoRA + Depth Anything V2, KJNodes ColorMatch, AutoGrid.')+' Not GPU-validated yet.')
                if profile=='krast_faithful_klein_v5':
                    folder=host.COMFYUI_DIR/'user/default/workflows/Selfism'
                    folder.mkdir(parents=True,exist_ok=True)
                    dest=folder/'KRAST_FaithfulPose_Millie_Klein_Amateur_v5.json'
                    if not dest.exists(): dest.write_bytes((root/'selfism_workflows/krast_faithful_klein_v5.json').read_bytes())
                    logs.append('Workflow saved: '+str(dest))
                    logs.append('KRAST FaithfulPose + Klein v5: KRAST-V2 img2img (res_2s/beta, 6 steps, denoise 0.65) + Millie/candid_snap/Krea Amateur V4/PornMaster Low Res, Artfat RVN, Krea2 depth 0.6, optional Flux 2 Klein 9B edit (K0 OFF), ColorMatch/Blend/Sharpen/FilmGrain, plus all Krea 2 realism LoRAs. Not GPU-validated yet.')
                if profile=='hearmeman_i2i':
                    folder=host.COMFYUI_DIR/'user/default/workflows/Selfism'
                    folder.mkdir(parents=True,exist_ok=True)
                    dest=folder/'Hearmeman_ImageToImage.json'
                    if not dest.exists(): dest.write_bytes((root/'selfism_workflows/hearmeman_i2i.json').read_bytes())
                    logs.append('Workflow saved: '+str(dest))
                    logs.append('Hearmeman ImageToImage: load the character image and the image to recreate. Krea 2 Turbo BF16, Qwen 3.5 9B prompt writer (6 sentences) and SeedVR2 7B upscale are installed. Detail Daemon is in the graph and bypassed. Add your Millie LoRA in the Power Lora Loader.')
                if profile=='hearmeman_wan_v5':
                    folder=host.COMFYUI_DIR/'user/default/workflows/Selfism'
                    folder.mkdir(parents=True,exist_ok=True)
                    dest=folder/'Hearmeman_WanAnimate_V5_SAM3.json'
                    if not dest.exists(): dest.write_bytes((root/'selfism_workflows/hearmeman_wan_v5.json').read_bytes())
                    logs.append('Workflow saved: '+str(dest))
                    logs.append('Hearmeman Wan Animate V5 (SAM3): load a character image and a driving video. OpenRouter auto-prompt needs an API key (node field, LLM_KEY, or openrouter_api_key.json next to the node). SageAttention 2.2.0 fp8 (sageattn_qk_int8_pv_fp8_cuda++, fp32+fp16) is baked in the image for sm_89 (L40S, RTX 4090), sm_90 (H100/H200) and sm_120 (RTX 5090, RTX PRO 6000). DWPose uses the image CUDA onnxruntime (CUDAExecutionProvider). About 48 GB+ VRAM, not GPU-validated.')
                if profile=='hearmeman_scail2_v2':
                    folder=host.COMFYUI_DIR/'user/default/workflows/Selfism'
                    folder.mkdir(parents=True,exist_ok=True)
                    dest=folder/'Hearmeman_SCAIL2_GodMode_V2.json'
                    if not dest.exists(): dest.write_bytes((root/'selfism_workflows/hearmeman_scail2_v2.json').read_bytes())
                    logs.append('Workflow saved: '+str(dest))
                    logs.append('Hearmeman SCAIL 2 God Mode V2: load a character image and a driving video. OpenRouter auto-prompt needs an API key (node field, LLM_KEY, or openrouter_api_key.json next to the node). SageAttention 2.2.0 fp8 (sageattn_qk_int8_pv_fp8_cuda++, fp32+fp16) is baked in the image for sm_89 (L40S, RTX 4090), sm_90 (H100/H200) and sm_120 (RTX 5090, RTX PRO 6000). About 48 GB+ VRAM, not GPU-validated.')
                if profile=='qwen21_edit_full':
                    folder=host.COMFYUI_DIR/'user/default/workflows/Selfism'
                    folder.mkdir(parents=True,exist_ok=True)
                    dest=folder/'AusBoss_Qwen21_GenerateEdit_Full_v2.json'
                    if not dest.exists(): dest.write_bytes((root/'selfism_workflows/qwen21_edit_full.json').read_bytes())
                    logs.append('Workflow saved: '+str(dest))
                    logs.append('Qwen Image 2.1 Edit (Full v2): AusBoss Generate + Edit v2.0 workflow as published (Edit a picture ON, Auto prompt OFF, Upscale OFF, Fast OFF, Steps 25, Consistency LoRA 0.5). Qwen Image 2.1 INT8 ConvRot, Qwen3-VL 8B INT8 ConvRot encoder, Qwen Image 2.1 VAE, Consistency/Restore/Viggle Turbo LoRAs and AusBoss nodes 2.20.0 are installed. About 20-27 GB VRAM, not GPU-validated. Qwen Image 2.1 weights are under the Qwen research licence (non-commercial).')
                if profile=='6gsc':
                    folder=host.COMFYUI_DIR/'user/default/workflows/Selfism'
                    folder.mkdir(parents=True,exist_ok=True)
                    dest=folder/'6gsc_dashboard_v1.json'
                    if not dest.exists(): dest.write_bytes((root/'selfism_workflows/6gsc.json').read_bytes())
                    logs.append('Workflow saved: '+str(dest))
                    logs.append('6gsc: upload the source image and edit the text prompt. Original Krea2 Turbo, DWPose/Ostris, sampling, refine and FaceDetailer settings are preserved. Mystic v3 replaces disabled v1; Yumi remains OFF and is not installed. The disconnected Flux GGUF loader was removed. No LLM prompter or presets are needed.')
                if profile=='god_mode':
                    # GOD Mode: Wan 2.2 Animate character replacement / animation (Kijai WanVideoWrapper + preprocess + SAM2 + RIFE).
                    folder=host.COMFYUI_DIR/'user/default/workflows/Selfism'
                    folder.mkdir(parents=True,exist_ok=True)
                    dest=folder/GOD_MODE_WORKFLOW_NAME
                    if not dest.exists(): dest.write_bytes((root/'selfism_workflows/wan22_animate_god_mode.json').read_bytes())
                    logs.append('Workflow saved: '+str(dest))
                    logs.append('GOD Mode: load a reference image (start frame) and a driving video. Pose/face detection (ViTPose+YOLO), SAM2 mask, Wan 2.2 Animate 14B + LoRAs, RIFE 2x. Relight + LightX2V + Pusa + Fun MPS LoRAs are selected in the graph. Needs a large GPU and lots of VRAM/RAM.')
                if profile in ('simple','aio','reference','carousel','full','full_refine'):
                    data=json.loads((root/'selfism_workflows'/f'{profile}.json').read_text(encoding='utf-8'))
                    for n in data['nodes']:
                        if n['type']=='UNETLoader' and profile in ('simple','aio','full','full_refine'): n['widgets_values'][0]=Path(catalog['files'][workflow['precision']]['destination']).name
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
                              'full':f'Selfism_FULL_{workflow["precision"]}_recreate_v1.json',
                              'full_refine':f'Selfism_FULL_{workflow["precision"]}_Refine_v1.json'}.get(profile,
                              f'Selfism_{profile}_{workflow["precision"]}_m1lli3.json')
                    dest=folder/filename
                    if not dest.exists(): dest.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
                    logs.append('Workflow saved: '+str(dest))
                    if profile in ('full','full_refine'):
                        # The Recreate workflow selects these LLM system presets by file name. A preset file on the pod
                        # overrides the text baked into the node, so keep the shipped files in sync with the workflow.
                        # Recreate_{SFW,NSFW}_{prefix,noprefix}.txt (v2, identity-safe) and
                        # Recreate_FirstFrame_{SFW,NSFW}_{prefix,noprefix}.txt (reel first frame) match this glob.
                        prompts=host.COMFYUI_DIR/'models/LLM/prompts'
                        prompts.mkdir(parents=True,exist_ok=True)
                        for preset in sorted((root/'selfism_workflows/presets').glob('Recreate_*.txt')):
                            (prompts/preset.name).write_bytes(preset.read_bytes())
                        logs.append('Recreate + Recreate FirstFrame LLM presets copied to '+str(prompts)+'. Restart ComfyUI if the prompter does not list them.')
                    if profile in ('full','full_refine'): logs.append('Upload your reference image. Millie LoRA is installed (first row of the Power Lora Loader).')
                    else: logs.append('Upload your reference image. Add your private Millie LoRA separately and enable its row.')
                    if profile=='full_refine':
                        logs.append('Refine uses Euler/simple, 6 steps, CFG 1, denoise 0.18. The REFINE switch selects base/refined output. All imported 6gsc LoRAs are OFF; Mystic v3 replaces v1. Yumi is an unavailable disabled placeholder. OpenPose row requires its separate control path before use.')
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
        elif profile=='6gsc':
            file_keys=catalog['6gsc_files']
            nodes=[n for n in catalog['nodes'] if n['name'] in catalog['6gsc_nodes']]
        elif profile in ('hearmeman_i2i','hearmeman_wan_v5','hearmeman_scail2_v2','qwen21_edit_full'):
            file_keys=catalog[profile+'_files']
            nodes=[n for n in catalog['nodes'] if n['name'] in catalog[profile+'_nodes']]
        elif profile=='recreate_hybrid':
            file_keys=catalog['recreate_hybrid_files']
            nodes=[n for n in catalog['nodes'] if n['name'] in catalog['recreate_hybrid_nodes']]
        elif profile=='krast':
            file_keys=catalog['krast_files']
            nodes=[n for n in catalog['nodes'] if n['name'] in catalog['krast_nodes']]
        elif profile=='krea_klein':
            file_keys=catalog['krea_klein_files']
            nodes=[n for n in catalog['nodes'] if n['name'] in catalog['krea_klein_nodes']]
        elif profile in ('krea_fluxdepth','krast_fluxdepth','krast_faithful_klein_v5','krast_org','rawgirl_millie','krast_org_depth','rawgirl'):
            file_keys=catalog[profile+'_files']
            nodes=[n for n in catalog['nodes'] if n['name'] in catalog[profile+'_nodes']]
        elif profile=='carousel':
            file_keys=catalog['carousel_files']
            nodes=[n for n in catalog['nodes'] if n['name'] in catalog['carousel_nodes']]
        elif profile in ('full','full_refine'):
            # Kompletan workflow: only INT8 (default) or FP8; BF16 is intentionally not offered.
            precision=request.precision if 'precision' in request.model_fields_set else 'int8'
            if precision not in ('int8','fp8'):
                raise HTTPException(400,'Kompletan workflow podržava samo Selfora INT8 ili FP8.')
            request=request.model_copy(update={'precision':precision})
            file_keys=[precision]+catalog[profile+'_files']
            nodes=[n for n in catalog['nodes'] if n['name'] in catalog[profile+'_nodes']]
        elif profile=='god_mode':
            # GOD Mode: Wan 2.2 Animate 14B bf16 + VAE/UMT5/CLIP/SAM2/ViTPose/YOLO + LoRAs + RIFE; WanVideoWrapper stack.
            file_keys=catalog['god_mode_files']
            nodes=[n for n in catalog['nodes'] if n['name'] in catalog['god_mode_nodes']]
        elif profile=='extras': file_keys=catalog['extra_files']
        elif profile=='model':
            if request.item not in catalog['files']: raise HTTPException(404,'Unknown model.')
            file_keys=[request.item]
        elif profile=='node':
            nodes=[n for n in catalog['nodes'] if n['name']==request.item]
            if not nodes: raise HTTPException(404,'Unknown node.')
        # Selfism Simple / AIO / Full add-ons (R2 first, HF fallback). The OlmLUT .cube files exist
        # only on private R2, so they are added (and hard-linked into ComfyUI-OlmLUT/luts) only when
        # the R2 connection is configured; without it the install stays as before.
        lut_links=[]; lut_note=''
        if profile in ('simple','aio','full'):
            file_keys=file_keys+[k for k in catalog['selfism_simple_addon_files' if profile=='simple' else 'selfism_addon_files'] if k not in file_keys]
            if profile in ('aio','full'):
                try: r2_ready=(await asyncio.to_thread(r2_client))[0] is not None
                except Exception: r2_ready=False
                if r2_ready:
                    file_keys=file_keys+catalog['selfism_lut_files']
                    lut_links=copy.deepcopy(catalog['selfism_lut_links'])
                    lut_note=f"OlmLUT: {len(lut_links)} LUT files from private R2 are linked into custom_nodes/ComfyUI-OlmLUT/luts/."
                else:
                    lut_note='OlmLUT: private R2 not configured, so the extra LUT files were skipped.'
        files=[copy.deepcopy(catalog['files'][k]) for k in file_keys]
        # Resolve exact Civitai metadata before accepting a multi-GB transfer.
        async with httpx.AsyncClient(timeout=30,follow_redirects=True) as client:
            for f in files:
                # This profile ships pinned size + SHA256 for every model; downloads verify those bytes.
                # Do not require Civitai access merely to install verified private-R2 copies.
                if profile in ('full_refine','6gsc','hearmeman_i2i','hearmeman_wan_v5','hearmeman_scail2_v2','qwen21_edit_full','recreate_hybrid','krast','krea_klein','krea_fluxdepth','krast_fluxdepth','krast_faithful_klein_v5','krast_org','rawgirl_millie','krast_org_depth','rawgirl') and re.fullmatch(r'[0-9a-f]{64}', f.get('sha256','')) and f.get('size_bytes',0)>0:
                    continue
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
        if lut_note: logs.append(lut_note)
        # Depth Anything V2 is downloaded under models/controlnet_aux, but the preprocessor
        # only looks in custom_nodes/comfyui_controlnet_aux/ckpts (or AUX_ANNOTATOR_CKPTS_PATH).
        # full/full_refine/reference/carousel already carry that link; the other depth cards did not.
        model_links=[]
        if profile in ('full','full_refine','6gsc','god_mode','hearmeman_wan_v5'):
            model_links=copy.deepcopy(catalog[profile+'_links'])
        elif profile in ('reference','carousel'):
            model_links=copy.deepcopy(catalog['reference_links'])
        if profile in ('krast','recreate_hybrid','krea_fluxdepth','krast_fluxdepth','krast_faithful_klein_v5','krast_org_depth','rawgirl'):
            model_links=model_links+copy.deepcopy(catalog['depth_anything_links'])
        workflow={'id':'selfism-'+profile,'title':'Selfora / Selfism — '+profile,
                  'files':files,'custom_nodes':nodes,'selfism_profile':profile,
                  'precision':'fp8' if profile in ('carousel','6gsc','hearmeman_i2i','hearmeman_wan_v5','hearmeman_scail2_v2','qwen21_edit_full','recreate_hybrid','krast','krea_klein','krea_fluxdepth','krast_fluxdepth','krast_faithful_klein_v5','krast_org','rawgirl_millie','krast_org_depth','rawgirl') else request.precision,
                  'selfism_repair':profile in ('simple','aio','reference','6gsc','hearmeman_i2i','recreate_hybrid','krast','krea_klein','krea_fluxdepth','krast_fluxdepth','krast_faithful_klein_v5','krast_org','rawgirl_millie','krast_org_depth','rawgirl','carousel','full','full_refine','repair','node'),
                  'selfism_repair_paths_only':profile in ('krast_org','krast_org_depth'),
                  'pip_packages':(list(catalog.get('god_mode_pip',[])) if profile=='god_mode' else []),
                  'model_links':model_links+lut_links}
        return await controller.start(workflow)

    @host.app.post('/api/selfism/cancel')
    async def cancel(): return await controller.cancel()

    @host.app.get('/api/selfism/workflow/{profile}')
    async def workflow_file(profile:str):
        if profile=='krea_fluxdepth':
            return FileResponse(root/'selfism_workflows/krea_fluxdepth_millie.json',filename='Krea2_FluxKlein_Millie_Artfat_Depth_v3.json')
        if profile=='krast_fluxdepth':
            return FileResponse(root/'selfism_workflows/krast_fluxdepth_millie.json',filename='KRAST_FluxKlein_Millie_Artfat_Depth_v3.json')
        if profile=='krast_org_depth':
            return FileResponse(root/'selfism_workflows/krast_org_depth/KRAST_ORG_Millie_AutoRatio_Depth_v1.json',filename='KRAST_ORG_Millie_AutoRatio_Depth_v1.json')
        if profile=='rawgirl':
            return FileResponse(root/'selfism_workflows/rawgirl/RawGirl_Autor_Millie_NMKD_Upscale_v1.json',filename='RawGirl_Autor_Millie_NMKD_Upscale_v1.json')
        if profile=='rawgirl_millie':
            return FileResponse(root/'selfism_workflows/rawgirl_millie.json',filename='RawGirl_Autor_Millie.json')
        if profile=='krast_org':
            return FileResponse(root/'selfism_workflows/krast_org.json',filename='KRAST_ORG_Millie.json')
        if profile=='krast_faithful_klein_v5':
            return FileResponse(root/'selfism_workflows/krast_faithful_klein_v5.json',filename='KRAST_FaithfulPose_Millie_Klein_Amateur_v5.json')
        if profile=='krea_klein_artfat':
            return FileResponse(root/'selfism_workflows/krea_klein_artfat.json',filename='Krea2_FluxKlein_Artfat_Reference.json')
        if profile=='krea_klein_famegrid':
            return FileResponse(root/'selfism_workflows/krea_klein_famegrid.json',filename='Krea2_FluxKlein_Famegrid.json')
        if profile=='krast_recreate':
            return FileResponse(root/'selfism_workflows/krast_recreate.json',filename='KRAST_Recreate.json')
        if profile=='krast_carousel':
            return FileResponse(root/'selfism_workflows/krast_carousel.json',filename='KRAST_Carousel.json')
        if profile=='recreate_hybrid':
            return FileResponse(root/'selfism_workflows/recreate_hybrid.json',filename='Selfism_Recreate_Hybrid.json')
        if profile=='hearmeman_i2i':
            return FileResponse(root/'selfism_workflows/hearmeman_i2i.json',filename='Hearmeman_ImageToImage.json')
        if profile=='hearmeman_wan_v5':
            return FileResponse(root/'selfism_workflows/hearmeman_wan_v5.json',filename='Hearmeman_WanAnimate_V5_SAM3.json')
        if profile=='hearmeman_scail2_v2':
            return FileResponse(root/'selfism_workflows/hearmeman_scail2_v2.json',filename='Hearmeman_SCAIL2_GodMode_V2.json')
        if profile=='qwen21_edit_full':
            return FileResponse(root/'selfism_workflows/qwen21_edit_full.json',filename='AusBoss_Qwen21_GenerateEdit_Full_v2.json')
        if profile=='6gsc':
            return FileResponse(root/'selfism_workflows/6gsc.json',filename='6gsc_dashboard_v1.json')
        if profile=='full_refine':
            return FileResponse(root/'selfism_workflows/full_refine.json',filename='Selfism_FULL_int8_Refine_v1.json')
        if profile not in ('simple','aio','reference','carousel','full','god_mode'): raise HTTPException(404)
        if profile=='god_mode':
            return FileResponse(root/'selfism_workflows/wan22_animate_god_mode.json',filename=GOD_MODE_WORKFLOW_NAME)
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

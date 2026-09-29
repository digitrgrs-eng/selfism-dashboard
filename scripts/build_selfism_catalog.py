"""Build the pinned add-on catalog from recorded upstream metadata (developer tool)."""
import concurrent.futures
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT.parent / '.research'
base = json.loads((ROOT / 'catalog/workflows.json').read_text())['workflows'][0]
files = {}
def add(key, spec):
    files[key] = dict(spec, id=key)
def hf(key, repo, filename, destination, metadata=None):
    if metadata is None:
        with urllib.request.urlopen('https://huggingface.co/api/models/'+repo+'?blobs=true', timeout=40) as r:
            metadata = json.load(r)
    item = next(x for x in metadata['siblings'] if x['rfilename'] == filename)
    add(key, dict(name=Path(filename).name, url=f'https://huggingface.co/{repo}/resolve/{metadata["sha"]}/{filename}',
        destination=destination, size_bytes=item['size'], sha256=item['lfs']['sha256'], auth='none'))
for key, name in [('encoder','Qwen3-VL 4B Scaled FP8 text encoder'),('sam','SAM ViT-B'),('face','Face YOLOv8m'),('edit','Krea 2 Identity Edit v1.2 LoRA')]:
    add(key,next(f for f in base['files'] if f['name']==name))
for key,repo,filename,dest,meta in [
 ('vae','Kijai/WanVideo_comfy','Wan2_1_VAE_fp32.safetensors','vae/Wan2_1_VAE_fp32.safetensors','WanVideo_comfy'),
 ('edit-vae','LS110824/vae','krea2RealVae_v10.safetensors','vae/krea2RealVae_v10.safetensors','vae'),
 ('eyes','marduk191/Ultralytics_models','bbox/Eyes.pt','ultralytics/bbox/Eyes.pt','Ultralytics_models'),
 ('hair','marduk191/Ultralytics_models','segm/hair_yolov8n-seg_60.pt','ultralytics/segm/hair_yolov8n-seg_60.pt','Ultralytics_models'),
 ('depth','Patil/Krea-2-depth-controlnet','depth-control-lora.safetensors','loras/depth-control-lora.safetensors','Krea-2-depth-controlnet'),
]: hf(key,repo,filename,'models/'+dest)
hf('skin','JCTN/ESRGAN','1x-ITF-SkinDiffDetail-Lite-v1.pth','models/upscale_models/1x-ITF-SkinDiffDetail-Lite-v1.pth')
repo='0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF'
with urllib.request.urlopen('https://huggingface.co/api/models/'+repo+'?blobs=true',timeout=40) as r: meta=json.load(r)
for key,name in [('llm','RVN-Q4_K_M-multilingual-mtp.gguf'),('mmproj','mmproj-Qwen3.8-27B-Q8_0.gguf')]:
    hf(key,repo,name,'models/LLM/'+name,meta)
with urllib.request.urlopen('https://api.github.com/repos/starinspace/StarinspaceUpscale/releases/tags/Models',timeout=40) as r: assets=json.load(r)['assets']
for key,name,dest in [('upscale','4xPurePhoto-RealPLSKR.pth','4xPurePhoto-RealPLSKR.pth'),('span','4xPurePhoto-Span.pth','4xPurePhoto-Span.pth')]:
    item=next(x for x in assets if x['name']==name)
    add(key,dict(name=name,url=item['browser_download_url'],destination='models/upscale_models/'+dest,size_bytes=item['size'],sha256=(item.get('digest') or '').removeprefix('sha256:'),auth='none'))
for key,version,fileid,name in [('fp8',3231658,3114237,'selforaV21NightFix_selforaV21Fp8.safetensors'),('int8',3231439,3113969,'selforaV21NightFix_selfora21Int8.safetensors'),('bf16',3231887,3114599,'Selfora_v2_1_bf16.safetensors')]:
    add(key,dict(name='Selfora v2.1 '+key.upper(),url=f'https://civitai.com/api/download/models/{version}?fileId={fileid}',destination='models/diffusion_models/'+name,size_bytes=0,sha256='',auth='civitai',civitai_version=version,civitai_file=fileid))

repos = [
 'rgthree/rgthree-comfy','ClownsharkBatwing/RES4LYF','TinyTerra/ComfyUI_tinyterraNodes',
 'artfat-creator/artfat-comfyui-llm-prompter','pythongosssss/ComfyUI-Custom-Scripts',
 'kijai/ComfyUI-KJNodes','ltdrdata/ComfyUI-Impact-Pack','ltdrdata/ComfyUI-Impact-Subpack',
 'artfat-creator/ComfyUI-Face-Style-Preset-for-FaceDetailer-',
 'BlackSnowSkill/ComfyUI-Krea2-Projector-Tuner','facok/comfyui-krea2-controlnet',
 'o-l-l-i/ComfyUI-OlmLUT','spacepxl/ComfyUI-VAE-Utils','lbouaraba/comfyui-krea2edit',
 'yolain/ComfyUI-Easy-Use','EllangoK/ComfyUI-post-processing-nodes','r-vage/ComfyUI-RvTools_v2',
 'Fannovel16/comfyui_controlnet_aux','cubiq/ComfyUI_essentials',
 'fearnworks/ComfyUI_FearnworksNodes','chflame163/ComfyUI_LayerStyle','ssitu/ComfyUI_UltimateSDUpscale','calcuis/gguf',
]
def node(repo):
    url='https://github.com/'+repo+'.git'
    result=subprocess.run(['git','ls-remote',url,'HEAD'],check=True,capture_output=True,text=True,timeout=60)
    ref=result.stdout.split()[0]
    return dict(name=repo.split('/')[1],repo=url,ref=ref,install_requirements=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool: nodes=list(pool.map(node,repos))
data=dict(version=1,files=files,nodes=nodes,
    simple_files=['encoder','vae','llm','mmproj','upscale'],
    aio_files=['edit-vae','face','sam'],
    extra_files=['edit','eyes','hair','depth','skin','span'],
    simple_nodes=[r.split('/')[1] for r in repos[:5]]+['gguf'])
# Reference profile has separately verified metadata; retain it on regeneration.
previous=json.loads((ROOT/'catalog/selfism.json').read_text(encoding='utf-8'))
for key in ('reference_files','reference_nodes','reference_links'):
    if key in previous: data[key]=previous[key]
for key in previous.get('reference_files',[]):
    if key not in data['files']: data['files'][key]=previous['files'][key]
known={n['name'] for n in data['nodes']}
data['nodes'] += [n for n in previous['nodes'] if n['name'] in previous.get('reference_nodes',[]) and n['name'] not in known]
(ROOT/'catalog/selfism.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
print('Catalog written:',len(files),'models,',len(nodes),'pinned nodes')


"""Millie Recreate Pro: Setup card (models + nodes for the Turbo / RawGirl / KRAST ORG Recreate Pro
workflows) and the Krea2 Turbo A/B test: A = reference photo + skeleton + Millie in the Ostris encoder,
B = skeleton only as image1 (reference stays the img2img start latent)."""
import asyncio
import json
import re

from fastapi.testclient import TestClient
from test_full import ROOT, CATALOG, start_install, m, s

PROFILE='millie_recreate_pro_setup'
FOLDER=ROOT/'selfism_workflows/millie_recreate_pro'
A='MillieRecreatePro_AB_A_Turbo_ReferenceInEncoder.json'
B='MillieRecreatePro_AB_B_Turbo_SkeletonOnly.json'
TURBO='krea2_pose_recreate_millie.json'
REFINE=(1788,1789,1790,1791,1699,1794,1795)
FACE=(1611,1617,1692,1703)
BODY=(1774,1775,1776,1777,1778,1779,1780,1781,1782)
ENC=(1764,1769)

# Node type -> node pack (resolved by grepping each pack at its pinned catalog ref and ComfyUI v0.38.1).
CORE={'BetaSamplingScheduler','CFGGuider','CLIPLoader','CLIPTextEncode','CheckpointLoaderSimple','ConditioningZeroOut',
      'EmptyLatentImage','EmptySD3LatentImage','FluxKontextImageScale','FluxKontextMultiReferenceLatentMethod','GetImageSize',
      'GrowMask','ImageBlur','ImageCompositeMasked','ImageFromBatch','ImageScaleBy','ImageScaleToTotalPixels','ImageToMask',
      'ImageUpscaleWithModel','KSampler','LatentUpscaleBy','LoadImage','LoraLoaderModelOnly','MaskToImage','ModelSamplingAuraFlow',
      'PreviewImage','PrimitiveFloat','PrimitiveStringMultiline','SamplerCustomAdvanced','SamplerDPMPP_SDE','SaveImage',
      'UNETLoader','UpscaleModelLoader','VAEDecode','VAEDecodeTiled','VAEEncode','VAEEncodeTiled','VAELoader','TextGenerate',
      'Note','MarkdownNote','PrimitiveNode','Reroute'}
PACK={'1hew_AnySwitchBool':'ComfyUI-1hewNodes','AdaptiveProjectedGuidance':'ComfyUI-Adaptive-Guidance','AdvancedNoise':'RES4LYF',
      'Sigmas Rescale':'RES4LYF','SigmasPreview':'RES4LYF','ClownsharKSampler_Beta':'RES4LYF',
      'BOOLConstant':'ComfyUI-KJNodes','ColorMatch':'ComfyUI-KJNodes','GrowMaskWithBlur':'ComfyUI-KJNodes','LazySwitchKJ':'ComfyUI-KJNodes',
      'ImageGridtoBatch':'ComfyUI-KJNodes','JoinStrings':'ComfyUI-KJNodes','StringConstantMultiline':'ComfyUI-KJNodes',
      'BboxDetectorCombined_v2':'ComfyUI-Impact-Pack','DenoiseSchedulerDetailerHookProvider':'ComfyUI-Impact-Pack',
      'DetailerForEachDebug':'ComfyUI-Impact-Pack','FaceDetailer':'ComfyUI-Impact-Pack','FromBasicPipe_v2':'ComfyUI-Impact-Pack',
      'MaskToSEGS':'ComfyUI-Impact-Pack','SAMLoader':'ComfyUI-Impact-Pack','SEGSPreview':'ComfyUI-Impact-Pack',
      'SegmDetectorCombined_v2':'ComfyUI-Impact-Pack','ToBasicPipe':'ComfyUI-Impact-Pack','UltralyticsDetectorProvider':'ComfyUI-Impact-Subpack',
      'Fast Mode Toggle [Eclipse]':'ComfyUI_Eclipse','Mute / Bypass Repeater [Eclipse]':'ComfyUI_Eclipse',
      'Switch latent [Crystools]':'ComfyUI-Crystools','NormalizedAttentionGuidance':'sd-perturbed-attention',
      'PixaromaLoraLoader':'ComfyUI-Pixaroma','PixaromaSaveImage':'ComfyUI-Pixaroma','Resolution':'CRT-Nodes',
      'Image Comparer (rgthree)':'rgthree-comfy','Seed (rgthree)':'rgthree-comfy','Power Lora Loader (rgthree)':'rgthree-comfy',
      'DWPreprocessor':'comfyui_controlnet_aux','DepthAnythingV2Preprocessor':'comfyui_controlnet_aux',
      'Krea2ControlApply':'comfyui-krea2-controlnet','Krea2ControlImageEncode':'comfyui-krea2-controlnet',
      'Krea2ControlLoRALoader':'comfyui-krea2-controlnet','Krea2OstrisEditModelPatch':'comfyui-krea2-ostris-edit',
      'TextEncodeKrea2OstrisEdit':'comfyui-krea2-ostris-edit',
      'InpaintCropImproved':'ComfyUI-Inpaint-CropAndStitch','InpaintStitchImproved':'ComfyUI-Inpaint-CropAndStitch'}
# Explicit-content parts of the Krast ORG recipe that this card deliberately does not install.
EXCLUDED_MODELS={'MOP-MIX-LUSTIFY.safetensors','Nipples.pt','pussy.pt','CockAndBallYolo8x.pt','Eyeful_v2-Paired.pt','Hands.pt'}
EXCLUDED_KEYS={'krast-org-lustify','krast-org-nipples-seg','krast-org-pussy-bbox','krast-org-cockandball','krast-org-eyeful',
               'krast-org-hands-seg','lora-mystic-xxx','famegrid'}
# LoRA rows that are OFF in the pose cards (not used by Recreate Pro; dropped from the A/B Power Lora Loader).
OFF_ROW_KEYS={'lora-mystic-xxx','famegrid','bloom'}
RECIPES=[TURBO,'krea2_pose_recreate_millie_rawgirl.json','krea2_pose_recreate_millie_krast.json','krast_org.json',
         'krast_org_pose_recreate_millie.json']+['krast_org_depth/'+p.name for p in sorted((ROOT/'selfism_workflows/krast_org_depth').glob('*.json')) if 'Carousel' not in p.name]


def _load(name): return json.loads((ROOT/'selfism_workflows'/name).read_text(encoding='utf-8'))
def _wf(name): return json.loads((FOLDER/name).read_text(encoding='utf-8'))
def _all_nodes(w): return list(w['nodes'])+[n for sg in (w.get('definitions') or {}).get('subgraphs',[]) for n in sg['nodes']]
def _types(w):
    sg={x['id'] for x in (w.get('definitions') or {}).get('subgraphs',[])}
    return {n['type'] for n in _all_nodes(w)}-sg


def _src(w,nid,name):
    N={n['id']:n for n in w['nodes']}; L={l[0]:l for l in w['links']}
    link=next(x for x in N[nid]['inputs'] if x['name']==name)['link']
    return None if link is None else (N[L[link][1]],L[link][2])


def test_folder_has_exactly_the_two_ab_workflows():
    assert sorted(p.name for p in FOLDER.glob('*'))==[A,B]


def test_links_are_consistent():
    for name in (A,B):
        w=_wf(name); N={n['id']:n for n in w['nodes']}; ids=set()
        for i,o,os_,t,ts,typ in w['links']:
            assert i not in ids; ids.add(i)
            assert i in N[o]['outputs'][os_]['links'] and N[t]['inputs'][ts]['link']==i
            assert N[o]['outputs'][os_]['type']==N[t]['inputs'][ts]['type']==typ
        for n in w['nodes']:
            for o in n.get('outputs',[]): assert set(o.get('links') or [])<=ids
            for x in n.get('inputs',[]): assert x.get('link') is None or x['link'] in ids
        assert w['last_node_id']>=max(N) and w['last_link_id']>=max(ids)
        assert len(N)==len(w['nodes'])


def test_ab_is_the_turbo_card_with_only_the_documented_edits():
    t=_load(TURBO); T={n['id']:n for n in t['nodes']}
    for name in (A,B):
        w=_wf(name); N={n['id']:n for n in w['nodes']}
        assert set(N)-set(T)=={t['last_node_id']+1,t['last_node_id']+2} and set(T)<=set(N)
        for i,o in T.items():
            n=N[i]
            assert n['type']==o['type'], i
            if i!=1633: assert n.get('widgets_values')==o.get('widgets_values'), i
            assert n['mode']==(4 if i in REFINE+FACE+BODY else 0), i
        rows=[r for r in N[1633]['widgets_values'] if isinstance(r,dict) and 'lora' in r]
        assert rows==[{'on':True,'lora':'millie_000002750.safetensors','strength':1,'strengthTwo':None},
                      {'on':True,'lora':'krea-smartphone-photo-slider.safetensors','strength':1,'strengthTwo':None}]
        # every original LoRA row that was ON is still ON with the same strength
        orig_on=[r for r in T[1633]['widgets_values'] if isinstance(r,dict) and r.get('on')]
        assert orig_on==rows
        extra=[l for l in w['links'] if l[0] not in {x[0] for x in t['links']}]
        save=N[t['last_node_id']+1]
        assert extra==[[t['last_link_id']+1,1622,0,save['id'],0,'IMAGE']]
        assert save['type']=='SaveImage' and save['mode']==0
        assert save['widgets_values']==['MillieRecreatePro/AB_'+('A' if name==A else 'B')]
        note=N[t['last_node_id']+2]; assert note['type']=='Note' and 'A/B TEST' in note['widgets_values'][0]
        assert all(n['widgets_values'][0]=='example.png' for n in w['nodes'] if n['type']=='LoadImage')
    a=_wf(A)
    assert [l for l in a['links'] if l[3]!=t['last_node_id']+1]==t['links']


def test_a_and_b_share_every_setting_except_the_encoder_images():
    a=_wf(A); b=_wf(B); NA={n['id']:n for n in a['nodes']}; NB={n['id']:n for n in b['nodes']}
    assert set(NA)==set(NB)
    for i in NA:
        if NA[i]['type']=='Note' or (NA[i]['type']=='SaveImage' and NA[i]['mode']==0): continue
        assert NA[i].get('widgets_values')==NB[i].get('widgets_values'), i
        assert NA[i]['mode']==NB[i]['mode'], i
    for N in (NA,NB):
        assert N[1634]['widgets_values']==[1,'fixed',8,1,'euler','simple',0.85]
        assert N[1765]['widgets_values']==['krea2_turbo_openpose_controlnet.safetensors',0.75]
        assert N[1689]['widgets_values'][0]=='krea2_turbo_fp8_scaled.safetensors'
        assert N[1687]['widgets_values']==['qwen3vl_4b_fp8_scaled.safetensors','krea2','default']
        assert N[1688]['widgets_values']==['qwen_image_vae.safetensors']
    diff_a={tuple(l) for l in a['links']}-{tuple(l) for l in b['links']}
    diff_b={tuple(l) for l in b['links']}-{tuple(l) for l in a['links']}
    assert {l[3] for l in diff_a|diff_b}<=set(ENC)


def test_a_layout_reference_skeleton_millie():
    w=_wf(A)
    for e in ENC:
        i1,_=_src(w,e,'image1'); i2,_=_src(w,e,'image2'); i3,_=_src(w,e,'image3')
        assert (i1['id'],i2['id'],i3['id'])==(1783,1767,1787)


def test_b_skeleton_only_and_reference_stays_the_start_latent():
    w=_wf(B)
    for e in ENC:
        i1,slot=_src(w,e,'image1'); assert (i1['id'],i1['type'],slot)==(1767,'FluxKontextImageScale',0)
        assert _src(w,e,'image2') is None and _src(w,e,'image3') is None
        assert _src(w,e,'vae')[0]['id']==1688 and _src(w,e,'clip')[0]['id']==1633
        assert _src(w,i1['id'],'image')[0]['type']=='DWPreprocessor'
    assert _src(w,1764,'prompt')[0]['id']==1793
    # reference photo -> blur -> resize -> VAEEncode -> PASS 1 latent, but never into the encoder
    lat,_=_src(w,1634,'latent_image'); assert lat['id']==1785
    ref,_=_src(w,1785,'pixels'); assert ref['id']==1783
    assert _src(w,1783,'image')[0]['id']==1784 and _src(w,1784,'image')[0]['id']==1761
    assert not [l for l in w['links'] if l[1] in (1783,1787) and l[3] in ENC]
    model,_=_src(w,1634,'model'); assert model['id']==1765
    assert _src(w,1765,'model')[0]['id']==1766 and _src(w,1766,'model')[0]['id']==1633


def test_every_node_type_resolves_to_core_or_a_setup_pack():
    packs=set(CATALOG[PROFILE+'_nodes'])
    assert packs<= {n['name'] for n in CATALOG['nodes']}
    for name in RECIPES:
        for t in _types(_load(name)):
            assert t in CORE or PACK.get(t) in packs, (name,t)
    for name in (A,B):
        for t in _types(_wf(name)): assert t in CORE or PACK.get(t) in packs, (name,t)
    assert set(PACK.values())<=packs


def test_every_model_used_by_the_recipes_is_installed_except_explicit_ones():
    files=CATALOG[PROFILE+'_files']
    assert len(files)==len(set(files)) and not set(files)&(EXCLUDED_KEYS|OFF_ROW_KEYS)
    dests={CATALOG['files'][f]['destination'] for f in files}
    names={d.rsplit('/',1)[1] for d in dests}
    for name in RECIPES+[A,B]:
        w=_load(name) if name in RECIPES else _wf(name)
        used=set()
        for n in _all_nodes(w):
            for v in (n.get('widgets_values') or []) if isinstance(n.get('widgets_values'),list) else []:
                if isinstance(v,str) and v.endswith(('.safetensors','.pt','.pth','.onnx')): used.add(v.rsplit('/',1)[-1])
                if isinstance(v,dict) and v.get('on') and isinstance(v.get('lora'),str): used.add(v['lora'])
                if isinstance(v,dict) and isinstance(v.get('loras'),list):
                    used|={r['name'] for r in v['loras'] if r.get('on')}
        for u in used-EXCLUDED_MODELS:
            assert u in names, (name,u)
    # auto prompter (core TextGenerate) uses Qwen3-VL 8B BF16, depth branch the depth LoRA + Depth Anything V2
    assert {'qwen3vl_8b_bf16.safetensors','depth-control-lora.safetensors','depth_anything_v2_vitl.pth',
            'face_yolov8m.pt','Eyes.pt','hand_yolov8s.pt','person_yolov8m-seg.pt','sam_vit_b_01ec64.pth'}<=names


def test_catalog_entries_are_pinned_and_verified():
    for k in CATALOG[PROFILE+'_files']:
        f=CATALOG['files'][k]
        assert re.fullmatch(r'[0-9a-f]{64}',f['sha256']) and f['size_bytes']>0, k
    for n in CATALOG['nodes']:
        if n['name'] in CATALOG[PROFILE+'_nodes']: assert re.fullmatch(r'[0-9a-f]{40}',n['ref']), n['name']
    cs=next(n for n in CATALOG['nodes'] if n['name']=='ComfyUI-Inpaint-CropAndStitch')
    assert cs=={'name':'ComfyUI-Inpaint-CropAndStitch','repo':'https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git',
                'ref':'6005efc02f7745649118edd89cdd084c2b9bd3ae','install_requirements':False}
    assert CATALOG[PROFILE+'_links']==CATALOG['krea2_pose_millie_links']+CATALOG['depth_anything_links']
    # superset of the three existing pose cards and Krast ORG (minus the explicit SDXL detailer parts)
    for p in ('krea2_pose_millie','krea2_pose_millie_rawgirl','krea2_pose_millie_krast','krast_org_pose_millie','krast_org_depth'):
        assert set(CATALOG[p+'_files'])-EXCLUDED_KEYS-OFF_ROW_KEYS<=set(CATALOG[PROFILE+'_files']), p
        assert set(CATALOG[p+'_nodes'])-{'ComfyUI-AutoGrid'}<=set(CATALOG[PROFILE+'_nodes']), p


def test_install_card_and_downloads(monkeypatch):
    r,captured,html=start_install(monkeypatch,{'profile':PROFILE,'precision':'int8'})
    assert r.status_code==200 and captured['precision']=='fp8' and captured['selfism_profile']==PROFILE
    assert [f['id'] for f in captured['files']]==CATALOG[PROFILE+'_files']
    assert {n['name'] for n in captured['custom_nodes']}==set(CATALOG[PROFILE+'_nodes'])
    assert captured['model_links']==CATALOG[PROFILE+'_links']
    assert captured['selfism_repair'] and captured['selfism_repair_paths_only'] and captured['pip_packages']==[]
    assert f'data-sf-action="{PROFILE}"' in html
    assert '<h2>Millie Recreate Pro: Setup (modeli + nodovi + A/B test)</h2>' in html
    assert html.index('<h2>Krast ORG Pose Recreate (Millie)</h2>')<html.index('<h2>Millie Recreate Pro: Setup')
    section=(ROOT/'scripts/selfism-section.html').read_text(encoding='utf-8')
    card=next(l for l in html.splitlines() if 'sf-card-millie-recreate-pro-setup' in l)
    assert card in section.splitlines()
    with TestClient(m.app) as client:
        for path,name in (('millie_recreate_pro_ab_a',A),('millie_recreate_pro_ab_b',B),(PROFILE,A)):
            response=client.get('/api/selfism/workflow/'+path)
            assert response.json()==_wf(name) and name in response.headers['content-disposition']
        assert client.get('/api/selfism/workflow/krea2_pose_millie').json()==_load(TURBO)


def test_installer_saves_folder_without_overwriting_user_edits(monkeypatch,tmp_path):
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
    base=tmp_path/'user/default/workflows/Selfism';base.mkdir(parents=True)
    other=base/'Krea2_PoseRecreate_Millie_v1.json';other.write_text('existing pose card')
    wf={'selfism_profile':PROFILE,'precision':'fp8','files':[{'id':'x'}],'selfism_repair':True,'selfism_repair_paths_only':True}
    asyncio.run(ctrl._install_workflow(wf))
    folder=base/'MillieRecreatePro'
    for name in (A,B): assert (folder/name).read_bytes()==(FOLDER/name).read_bytes()
    (folder/A).write_text('user edit')
    asyncio.run(ctrl._install_workflow(wf))
    assert (folder/A).read_text()=='user edit' and (folder/B).read_bytes()==(FOLDER/B).read_bytes()
    assert other.read_text()=='existing pose card'

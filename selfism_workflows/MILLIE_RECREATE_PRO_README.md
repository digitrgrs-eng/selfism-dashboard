# Millie Recreate Pro: Setup (card `millie_recreate_pro_setup`)

One install card for the upcoming Millie Recreate Pro workflows (three separate cards later: Krea2 Turbo, RawGirl, KRAST ORG) and for the A/B test that comes first. Every existing card and workflow is unchanged.

Workflows are copied (never overwritten) into ComfyUI `user/default/workflows/Selfism/MillieRecreatePro/`: every `*.json` in `selfism_workflows/millie_recreate_pro/`, so the final Recreate Pro workflows only need to be added to that folder later.

## Models (20, about 72.2 GB, all existing catalog entries shared with other cards)

| Role | Catalog key | File |
| --- | --- | --- |
| Base model Turbo | `krea-turbo` | `krea2_turbo_fp8_scaled.safetensors` |
| Base model RawGirl | `rawgirl-krea2-v20-fp8` | `rawgirlKrea2_v20Fp8.safetensors` |
| Base model KRAST v2 | `krast-v20` | `krast_v20.safetensors` |
| Text encoder (Turbo, RawGirl) | `encoder` | `qwen3vl_4b_fp8_scaled.safetensors` |
| Text encoder (KRAST ORG) | `krast-org-heretic-te` | `Qwen3-VL-4B-Instruct-Heretic-int8-convrot.safetensors` |
| Auto prompter (core `TextGenerate`, CLIPLoader type `qwen_image`, as the Qwen 2.1 Dataset Generator card) | `qwen3vl-8b-bf16` | `qwen3vl_8b_bf16.safetensors` |
| VAE (Turbo, RawGirl) | `qwen-vae` | `qwen_image_vae.safetensors` |
| VAE (KRAST ORG) | `edit-vae` | `krea2RealVae_v10.safetensors` |
| Millie LoRA (m1lli3) | `millie` | `millie_000002750.safetensors` |
| Smartphone slider (private R2) | `6gsc-smartphone` | `krea-smartphone-photo-slider.safetensors` |
| Pose LoRA (thedeoxen/Krea-2-pose-controlnet) | `pose-lora` | `krea2_turbo_openpose_controlnet.safetensors` |
| Depth LoRA (`Krea2ControlLoRALoader`, facok/comfyui-krea2-controlnet) | `depth` | `depth-control-lora.safetensors` |
| Depth preprocessor (`DepthAnythingV2Preprocessor`) | `depth-anything` | `depth_anything_v2_vitl.pth` |
| DWPose | `dwpose-det`, `dwpose-pose` | `yolox_l.onnx`, `dw-ll_ucoco_384_bs5.torchscript.pt` |
| Detectors + SAM | `face`, `eyes`, `hand`, `person-seg-m`, `sam` | `face_yolov8m.pt`, `Eyes.pt`, `hand_yolov8s.pt`, `person_yolov8m-seg.pt`, `sam_vit_b_01ec64.pth` |

DWPose and Depth Anything V2 are also linked into `custom_nodes/comfyui_controlnet_aux/ckpts/` (as on the pose and depth cards).

Not installed on purpose: the MoP Mix Lustify SDXL checkpoint and the explicit SDXL detailer models of the Krast ORG file, and the OFF LoRA rows of the pose cards (Mystic XXX, FameGrid Spicy, Bloomgirls).

## Node packs (16, all pinned)

rgthree-comfy, RES4LYF, ComfyUI-KJNodes (ColorMatch), ComfyUI-Impact-Pack, ComfyUI-Impact-Subpack, ComfyUI-Pixaroma, ComfyUI-Adaptive-Guidance, ComfyUI-Crystools, ComfyUI-1hewNodes, CRT-Nodes, ComfyUI_Eclipse, sd-perturbed-attention, comfyui_controlnet_aux, comfyui-krea2-ostris-edit, comfyui-krea2-controlnet and the new **ComfyUI-Inpaint-CropAndStitch** (lquesada, `6005efc02f7745649118edd89cdd084c2b9bd3ae`, v3.0.20, no extra pip requirements) for the background mask.

Every node type in the Krast ORG files (`krast_org.json`, `krast_org_depth/*` including the AutoRatio Depth IMG2IMG ColorMatch file, `krast_org_pose_recreate_millie.json`), the three Pose Recreate cards and the A/B files resolves to ComfyUI core (v0.38.1) or one of these packs (`tests/test_millie_recreate_pro_setup.py`). No SAM3 or extra Differential Diffusion pack (`DifferentialDiffusion` is core).

## A/B test (Krea2 Turbo)

Both files are the `krea2_pose_recreate_millie.json` Turbo card with only these edits:

- PASS 2 (2x upscale + 4 MP detailing), FACE FIX and BODY FIX nodes are kept but bypassed (mode 4). Turn a group back on with right-click → Bypass Group Nodes (or Ctrl+B on the nodes).
- New `A/B RESULT` SaveImage after the PASS 1 decode: `output/MillieRecreatePro/AB_A_*` and `AB_B_*`. Group 5 is split into 5 (PASS 1) and 5b (PASS 2).
- The OFF LoRA rows (Mystic XXX, FameGrid, Bloomgirls) are removed from the Power Lora Loader; Millie 1.0 and the smartphone slider 1.0 stay ON.
- A READ ME note.

Same in A and B: reference image chain (blur 2/1 → FluxKontextImageScale → VAEEncode = PASS 1 start latent), prompt node, seed 1 fixed, 8 steps euler/simple CFG 1, denoise 0.85, Ostris patch, OpenPose LoRA 0.75.

- **A** (`MillieRecreatePro_AB_A_Turbo_ReferenceInEncoder.json`): Ostris encoder image1 = reference photo, image2 = DWPose skeleton, image3 = Millie reference (current layout).
- **B** (`MillieRecreatePro_AB_B_Turbo_SkeletonOnly.json`): Ostris encoder (positive and negative) gets only the DWPose skeleton as image1. The reference photo is not in the encoder but is still the img2img start latent. The Millie reference loader is kept but not connected. The prompt text is identical to A, so its image1/image3 words have no image behind them in B.

The Ostris encoder numbers only the connected images ("Picture 1", "Picture 2"...), so in B the skeleton is Picture 1, as the pose LoRA author documents.

VRAM: Turbo FP8 13.1 GB + Qwen3-VL 4B FP8 6 GB + VAE/LoRAs ~1.5 GB, plus 1 MP activations with up to three reference latents: fits a 48 GB GPU (A has one more reference latent than B). Estimate only. Not GPU-validated.

# Millie Pose Recreate (Krea2 Ostris + DWPose)

Source: the ComfyUI workflow embedded in the user's PNG renders (workflow id `aa23637a-…`, 48 nodes, 79 links). The original bytes are kept in `sources/krea2_pose_recreate_original.json`. All four PNGs that carried metadata use this same graph (identical links); they differ only in input image names, seeds, smartphone-slider on/off, DWPose body toggle and pose-LoRA strength.

## What changed for Millie

- Character LoRA `bellasima.safetensors` (trigger `b3llaa22`) → `millie_000002750.safetensors` 1.0 (trigger `m1lli3`).
- `krea2_smartphone_photography_slider.safetensors` → `krea-smartphone-photo-slider.safetensors` (same Civitai 2796343 file, identical SHA256), ON 1.0.
- `MysticXXX_KREA2_v1` (OFF) → `MysticXXX_KREA2_v3` (OFF), as on the 6gsc card. famegrid and bloomgirls stay OFF.
- The OFF `krea2_edit_outfit_transfer.safetensors` row was removed (unverifiable file, not on R2).
- Prompts rewritten for Millie (22, oval face, dark-brown almond eyes, long dark-brown center-parted hair, slim hourglass). The prompt node stays manual; there is no LLM prompter.
- Hungarian node and group titles translated to English. Input image names reset to `example.png`.

Everything else (links, samplers, denoise 0.85 / 0.2, 2× 4 MP pass, pose LoRA 0.75, Ostris patch, FaceDetailer 0.28, BodyDetailer 0.1, blurred face composite) is unchanged and is checked by `tests/test_krea2_pose_millie.py`.

## Inputs

1. `REFERENCE IMAGE` (image1): the photo or video first frame to recreate. It is also the start latent.
2. `MILLIE REFERENCE` (image3): a Millie photo showing her chest/body; used only for body shape and bust size.
3. `WRITE YOUR PROMPT HERE`: describe the reference scene (camera, pose, outfit, room, light). Keep the `m1lli3` look prefix.

## Installation

14 models (~23.2 GB, mostly shared with 6gsc/Krea cards) and 5 pinned node packs (rgthree, Impact Pack, Impact Subpack, controlnet_aux, Ostris Edit `7756566`). Models come from private R2 when size + SHA256 metadata match, otherwise from the pinned public source. DWPose weights are linked into `custom_nodes/comfyui_controlnet_aux/ckpts`. New catalog asset: `person-seg-m` (`ultralytics/segm/person_yolov8m-seg.pt`, Bingsu/adetailer, 54,827,683 bytes). GPU execution has not been tested.

## RawGirl and KRAST v2 variants

`krea2_pose_recreate_millie_rawgirl.json` (card `krea2_pose_millie_rawgirl`) and `krea2_pose_recreate_millie_krast.json` (card `krea2_pose_millie_krast`) are the same graph with only these changes:

| | Turbo card | RawGirl | KRAST v2 |
|---|---|---|---|
| UNETLoader | `krea2_turbo_fp8_scaled` | `rawgirlKrea2_v20Fp8` (default dtype) | `krast_v20` (default dtype) |
| Pass 1 + pass 2 KSampler | euler / simple | euler / beta | er_sde / beta |

Steps (8), CFG (1), denoise (0.85 / 0.2), seeds, LoRAs, prompts, Ostris patch, pose LoRA, detailers (euler/normal) and upscale are unchanged. Both finetunes are step-distilled like Turbo: the existing RawGirl Millie cards run 8 steps, CFG 1, euler/beta (ClownsharKSampler), and the KRAST cards run 6–8 steps, CFG 1, er_sde or res_2s with beta. Only the scheduler/sampler was aligned with those cards.

Risk: the Ostris edit patch and `krea2_turbo_openpose_controlnet` were trained on Krea 2 (Turbo). The finetunes share the architecture (same Qwen3-VL 4B encoder, Qwen VAE and Millie LoRA as on their own cards), so they load, but pose adherence and multi-reference editing may be weaker or different. Not GPU-validated.

## Krast ORG Pose Recreate (card `krast_org_pose_millie`)

`krast_org_pose_recreate_millie.json` (saved in ComfyUI as `Selfism/Krast_ORG_PoseRecreate_Millie_v1.json`) is a copy of `krast_org.json` with the Bella pose technique added to stage 1. `krast_org.json` and every other card are unchanged.

Kept from Krast ORG: the `Model loader` subgraph (KRAST-V2 INT8 `krast_v20`, Qwen3-VL-4B Heretic INT8 ConvRot, `krea2RealVae_v10`), Pixaroma LoRA loader with Millie 1.0, stage 1 KSampler er_sde/simple 8 steps CFG 1 seed from `Seed (rgthree)`, stage 2 (1.5× latent upscale, DPM++ SDE + Beta 6 steps, sigma rescale, Adaptive Projected Guidance, text-only conditioning), FaceDetailer subgraph, MoP Mix Lustify SDXL detailers and the three PixaromaSaveImage nodes (now `output/KRAST_ORG_POSE/`).

Removed: the img2img `LATENT` subgraph and its `Load Image` (the reference chain replaces them).

Added (values from the Bella original workflow):

- `REFERENCE IMAGE (image1)` → `DWPose - skeleton` (hand/body/face on, 1024, yolox_l + dw-ll_ucoco_384_bs5) → `Skeleton resize (image2)`.
- Reference → `Reference denoise blur` (2, 1) → `Reference resize (image1)` → `VAEEncode` with krea2RealVae → stage 1 latent, denoise 0.85.
- `OPTIONAL image3` LoadImage → resize → image3 of both encode nodes. Not needed: delete its two links into `PROMPT STAGE 1` and `NEGATIVE STAGE 1`.
- Stage 1 model: Pixaroma LoRA (Millie 1.0 + `krea-smartphone-photo-slider` 1.0) → `Krea2OstrisEditModelPatch` (kv_cache on) → `POSE CONTROLNET LORA` 0.75. Stage 2 and the detailers keep the plain LoRA model.
- `PROMPT STAGE 1` / `NEGATIVE STAGE 1 (leave empty!)`: `TextEncodeKrea2OstrisEdit` with the Heretic CLIP and krea2RealVae, followed by `FluxKontextMultiReferenceLatentMethod` index_timestep_zero.
- `WRITE YOUR JSON PROMPT HERE` feeds stage 1 and the stage 2 `CLIPTextEncode` (`PROMPT STAGE 2`). Put `image_roles` first.

Compatibility (checked in source, not on a GPU):

- Ostris text encode calls `clip.tokenize(..., images=...)` and needs the Qwen3-VL vision tower. ComfyUI v0.38.1 loads the Heretic file as the core `krea2` CLIP (`Qwen3VL` with `visual`). By size accounting the Heretic INT8 file (5,246,908,680 bytes) = Comfy FP8 file minus `lm_head` (embeddings 0.778 GB + 3.634 GB INT8 linears + 0.831 GB F16 vision tower), so the vision weights are present. The header itself could not be read (Civitai needs a token here).
- `Krea2OstrisEditModelPatch` only calls the DiT submodules (`wq/wk/wv/gate/wo`, `mlp`, `first`, `tmlp`, ...), so the INT8 `comfy_quant` layers of `krast_v20` run through the normal quantized ops; the same file is already patched on the `krea2_pose_millie_krast` card.
- `krea2RealVae_v10` has the same 194 tensors/shapes as `qwen_image_vae` (F32), encoder included; Krast ORG already uses it for VAEEncode.

No fallback encoder/VAE was needed. About 48 GB VRAM (as Krast ORG; three 1 MP reference latents add ~12k tokens, cached once per run with kv_cache). Not GPU-validated.

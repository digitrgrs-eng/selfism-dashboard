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

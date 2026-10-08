# Selfism Full + Refine v1

Separate workflow/profile derived from Selfism_FULL_int8_recreate_v1 and the second sampler in 6gsc. The original Full card and JSON are preserved.

## Use

1. Install **Selfism Full + Refine** in the dashboard, choose INT8 or FP8, then restart ComfyUI after installation. Private R2 environment settings must be configured; never put keys in workflow files.
2. Open `Selfism_FULL_int8_Refine_v1.json` (or the FP8 file created by the installer), then upload the reference image as in Full.
3. Run with the initial settings. `Selfism_Full_Refine/base` saves the first pass; `Selfism_Full_Refine/selected` saves the selected final output.
4. `REFINE ON / OFF`: true selects the refined image; false selects the base and skips the upscale/second sampler. Keep other optional processing bypassed for this comparison.
5. Keep the input, prompt and base seed unchanged while comparing. Base seed is fixed at 1803, refine seed at 40, and the LLM seed control is fixed. Change the base seed to request a new base image. Inspect the actual LLM text if you change references or prompter settings.

The refine recipe is NMKD x4 -> nearest-exact x0.25 -> Qwen VAE encode -> KSampler (6 steps, Euler, simple, CFG 1, denoise 0.18) -> Qwen VAE decode. **40 is its seed, not its step count.** At x0.25 the output retains the base dimensions. Keep this factor for the initial test: the existing Depth control is aligned to the base latent. Higher resolution requires checking/re-aligning that control.

This replaces Full's previous second sampler; it does not add a third pass. Selfora, Millie, the original Skin Tone row, base sampling and Depth remain. The baked LLM instructions match the current dashboard's Recreate_SFW_prefix preset.

Both Full and Full + Refine install `krea2RealVae_v10.safetensors` and `krea2_identity_edit_v1_2.safetensors` even while Identity Edit is bypassed, so its retained loaders have their required files. This does not enable Edit or change the main Qwen VAE.

## Optional 6gsc LoRAs

All eight rows are present and **OFF**. The installer includes the seven available models:

| Row | Stored strength | Availability |
|---|---:|---|
| MysticXXX KREA2 v3 | 0.7 | Existing newer version replaces v1 |
| famegrid spicy | 0.45 | Installed |
| bloomgirls realism | 0.3 | Installed |
| Yumi 000002250 | 0.65 | Not in R2; disabled placeholder, not installed |
| SNOFS Krea v1.4 | 1.0 | Installed |
| smartphone photo slider | 0.25 | Private R2 required |
| Photografic Scene Coherence V1.5 | 0.3 | Private R2 required |
| Krea2 Turbo OpenPose control | 0.7 | Installed, OFF; requires a separate Ostris/DWPose control path to use correctly |

Do not use Toggle All. Test style LoRAs individually. An available LoRA is not a guarantee of compatibility or improved identity/realism with Selfora. The disabled Yumi row is ignored by rgthree's loader; enabling it requires its actual model first.

Models use verified SHA256 and size, with private R2 preferred. Two R2-only assets fail with an explicit connection message if verification/access fails. Custom nodes use the existing Full installer's pinned GitHub revisions and requirements; they are not restored from the archived R2 node bundles. Existing user-edited workflow files are not overwritten.

## Validation limits

Automated checks cover graph links, lazy branch routing, sampler settings, LoRA defaults/model coverage, installer dependency selection, precision selection, private-R2 failure handling and preservation of existing Full files. No GPU image generation has been performed for this version. Visual realism, identity and lighting require an actual RunPod comparison.

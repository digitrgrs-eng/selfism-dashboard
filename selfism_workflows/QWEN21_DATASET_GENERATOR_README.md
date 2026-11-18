# Qwen 2.1 Dataset Generator (24 images)

Card id: `qwen21_dataset_generator`. Installs `Qwen21_Dataset_Generator_24.json` into
`user/default/workflows/Selfism`. Existing user edits are never overwritten.

Source: the published 1050R / 10sorLabs Qwen Image 2.1 Dataset Generator
(`sources/qwen21_dataset_generator_original.json`, 924 nodes including subgraphs).

Card edits (everything else is the published graph):
- SeedVR2 DiT / VAE loaders and the upscaler: `offload_device` `cpu` -> `cuda:0` (GPU only, 72 nodes).
- `Model description` replaced with the neutral `Adult woman.`; write your own physical description.
- `Load Face` / `Load Body+Face` default to `example.png`.

Inputs: a sharp head-and-shoulders portrait facing the camera (Load Face) and a full-body photo of
the same person in fitted clothing (Load Body+Face). Test with quality 0.30-0.50 and the Enhancer
off, then 1.00 with the Enhancer on.

Models (11, about 62.4 GB): REDQW21 UNLOCKED v2 FP8, Qwen3-VL 8B BF16, Qwen Image 2.1 VAE,
NSFW Qwen by TheseAlpacas V2 LoRA, Lenovo UltraReal Qwen 2.1 LoRA, BFS Head v1.1 LoRA,
Krea 2 Turbo FP8, Qwen3-VL 4B FP8 scaled, krea2RealVae, SeedVR2 EMA 7B FP16 + EMA VAE FP16.
Nodes: rgthree, KJNodes, Easy-Use, Custom-Scripts, SeedVR2 (all ComfyUI core nodes exist in v0.38.1).

VRAM: 48 GB minimum (A40 / L40S / A6000), 80 GB recommended (A100 / H100) because SeedVR2 keeps
everything on the GPU. Not GPU-validated.

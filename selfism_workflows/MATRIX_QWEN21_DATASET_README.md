# Matrix Dataset (Qwen Image 2.1, 32 images)

Card id: `matrix_qwen21_dataset`. Installs `Matrix_Qwen21_Dataset.json` into
`user/default/workflows/Selfism`. Existing user edits are never overwritten.

Source: MATRIX LAB "MATRIX Qwen Image 2.1 Dataset Workflow - Normal"
(`sources/matrix_qwen21_dataset_original.json`, 202 nodes, no subgraphs).

Card edits (everything else is the published graph):
- Power Lora Loader Pruna row: `p_qwen_image_2.1_8step_v0.1.safetensors` ->
  `p_qwen_image_2.1_8step_v0.1-comfyui-T8.safetensors` (still off, strength 1.0). The T8 file is
  the same PrunaAI weights (source sha256 f0865d68...) with the per-module alpha tensors added. The
  original has rank 64 / lora_alpha 128 but no alpha keys, so ComfyUI would apply it at half
  strength. The T8 file is already in the catalog (Qwen Image 2.1 Swap card).

Inputs: no LoadImage nodes. Each section has a Fantastic H3 Media Loader; drag the pictures in, in
order, without gaps (picture 1 face; picture 2 upper body with outfit; picture 3 full body with
outfit and shoes). The input aspect must match the section's MATRIX QWEN IMAGE 2.1 RESOLUTION node.

Models (5, about 17.9 GB, all shared with the Qwen Image 2.1 Swap card): Qwen Image 2.1 INT8
ConvRot, Qwen3-VL 8B INT8 ConvRot, Qwen Image 2.1 VAE bf16, BFS Head v1.1 LoRA (off), Pruna 8-step
LoRA T8 (off). INT8 ConvRot loads with the core UNETLoader / CLIPLoader of the baked ComfyUI
v0.38.1. Nodes: rgthree (baked), MATRIX-LAB-Nodes, ComfyUI-Fantastic-MiniMaxH3-PromptBuilder. The
two new packs install no pip packages: everything they import (torch, numpy, Pillow, scipy, aiohttp,
av) is already in the image through ComfyUI v0.38.1, so `requirements_packages` is empty and
torch is never touched.

VRAM: about 24 GB minimum, 32-48 GB recommended (25 steps at about 4.3 MP). Everything on GPU.
Not GPU-validated.

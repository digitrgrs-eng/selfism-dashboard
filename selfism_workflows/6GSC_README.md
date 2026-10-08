# 6gsc dashboard package

This card installs the supplied `6gsc workflow.json` recipe, separately from Reference + Depth, Full and Full + Refine. Upload a source image in `SOURCE / INSPIRATION IMAGE`, edit the positive prompt manually, and run. The original workflow does not use an LLM prompter or TXT presets and does not automatically apply Millie identity.

Preserved: Krea2 Turbo FP8, Qwen3-VL encoder, Qwen image VAE, DWPose/Ostris conditioning and model patch, every active graph connection, original positive/negative prompts, base sampler (10 steps, res_2s/beta, denoise 0.6, seed 1803/increment), refine (6 steps, Euler/simple, denoise 0.18, seed 40/fixed), NMKD x4 then nearest-exact x0.25, FaceDetailer and the original enabled/disabled LoRA states.

Only portability changes:

- Remove node 1761 (`UnetLoaderGGUF`, `flux2-dev-Q4_K_M.gguf`). It has no connections in the original graph and has no effect on its output.
- Replace disabled MysticXXX KREA2 v1 with the user's existing v3; its strength and disabled state stay the same.
- Give the packaged workflow its own ID. The original bytes are retained under `sources/6gsc_original.json` for comparison.

Yumi remains a disabled row, is unavailable in the audited R2 inventory and is not installed. Do not enable it without its actual model. Other optional 6gsc LoRAs are installed along with active dependencies. The four original active rows remain active: famegrid, smartphone photo slider, Scene Coherence and OpenPose control.

## Installation

The **6gsc** card installs 15 model assets and seven pinned custom-node packages (rgthree, RES4LYF, KJNodes, Impact Pack/Subpack, controlnet_aux and Ostris Edit). Node requirements use the existing constrained installer. DWPose detection and pose weights are downloaded through the normal verified model pipeline and linked into the exact controlnet_aux `ckpts` subdirectories; no first-run model download is intended.

R2 is preferred for models with matching size and SHA256 metadata, then the existing source fallback applies. Smartphone and Scene Coherence are private-R2-only assets and require configured R2 access. Nodes install from pinned GitHub revisions. No R2 credentials are stored in the workflow or repository. The Selfora precision selector does not alter this Krea2 FP8 profile.

The packaged file is installed as `ComfyUI/user/default/workflows/Selfism/6gsc_dashboard_v1.json`. Existing edited files and other workflows are not overwritten. The downloadable JSON is the same file the installer saves.

Automated checks compare the entire graph with the supplied original, permitting only the three changes above, and cover model dependencies, DWPose paths, card selection and non-destructive installation. GPU execution and image quality have not been tested.

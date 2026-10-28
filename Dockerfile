# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS r2deps
RUN pip install --no-cache-dir --no-compile --target=/r2deps boto3==1.43.104
# Verified linux/amd64 manifest of 10sorllabs/comfyui-workflow-launcher:2.0.
# Reuse its already-built CUDA/runtime layers. The only RUN is the ComfyUI bake below:
# BuildKit must unpack the base rootfs (~11 GiB) to execute it, so build-image.yml's
# existing cleanup (frees ~114 GiB on the hosted runner) is required. Every other step stays COPY --link.
FROM 10sorllabs/comfyui-workflow-launcher@sha256:d01908958aa33cc9117b478d81845d14ed53c135f173e3db3eafe14e780e8846

# Bake ComfyUI v0.38.1 (git checkout, origin = upstream) into /opt/comfyui-baked.
# bake_comfyui.sh first upgrades the base cu128 torch stack to 2.10.0+cu130 (matching
# torchvision/torchaudio) so comfy_kitchen CUDA is not disabled, then installs ComfyUI
# requirements with torch/torchvision/torchaudio/numpy/transformers/pillow/opencv pinned
# to that cu130 freeze. RunPod's start.sh copies /opt/comfyui-baked to
# /workspace/runpod-slim/ComfyUI on first boot and its venv (--system-site-packages; path
# may still be named .venv-cu128) sees these packages, so the boot hook (selfism_boot ->
# selfism_int8.py) finds INT8 support and comfy-kitchen >= 0.2.16 and does nothing.
# Keep COMFYUI_TAG/COMFYUI_SHA identical to TARGET_TAG/TARGET_SHA in launcher/selfism_int8.py
# (tests/test_bake.py checks this). Launcher files are untouched.
ARG COMFYUI_TAG=v0.38.1
ARG COMFYUI_SHA=20ca544ee0436721d8eb5f544665e490609f72c8
# The same RUN first installs OpenGL/EGL/GLES userspace for ComfyUI core GLSLShader
# (Artfat Colorgrade grain); with it GLSLShader initializes via ANGLE on NVIDIA Vulkan.
# The same RUN also bakes the Krast ORG + Depth node packs (and the two Hearmeman
# packs and the AusBoss pack) from catalog/selfism.json into /opt/10sorlabs/node_cache, then installs
# the SageAttention 2.2.0 cu130/cp312 wheel (--no-deps, so onnxruntime-gpu stays
# 1.30.0). Keep it one RUN: BuildKit unpacks the base rootfs only for this step.
RUN --mount=type=bind,source=docker/bake_comfyui.sh,target=/tmp/bake_comfyui.sh \
    --mount=type=bind,source=catalog/selfism.json,target=/tmp/selfism-catalog.json \
    apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 libopengl0 libglu1-mesa libegl1 libegl-mesa0 libgl1-mesa-dri libgles2 \
       libportaudio2 \
    && rm -rf /var/lib/apt/lists/* \
    && COMFYUI_TAG="${COMFYUI_TAG}" COMFYUI_SHA="${COMFYUI_SHA}" bash /tmp/bake_comfyui.sh

ARG IMAGE_VERSION=dev
LABEL org.opencontainers.image.title="Selfora / Selfism dashboard" \
      org.opencontainers.image.description="10sorLabs launcher with Selfism installation and runtime repair" \
      org.opencontainers.image.source="https://github.com/digitrgrs-eng/selfism-dashboard" \
      org.opencontainers.image.version="${IMAGE_VERSION}"

# controlnet_aux reads AUX_ORT_PROVIDERS (else its default list, which also starts
# with CUDA). CUDA first, CPU if the provider cannot be selected. The bake installs
# onnxruntime-gpu==1.30.0 and checks CUDAExecutionProvider is registered.
ENV LAUNCHER_AUTO_UPDATE=0 \
    SELFISM_AUTO_REPAIR=1 \
    SELFISM_AUTO_COMFY_UPDATE=1 \
    HF_TOKEN_FILE=/dev/null \
    PYTHONPATH=/opt/r2deps:/opt/10sorlabs \
    AUX_ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider

# User credentials must be supplied via RunPod environment variables.
# Do not use credentials possibly baked into the upstream image.
COPY --link launcher/ /opt/10sorlabs/launcher/
COPY --link --from=r2deps /r2deps/ /opt/r2deps/
COPY --link scripts/upload_models_r2.py /opt/10sorlabs/scripts/upload_models_r2.py
COPY --link catalog/ /opt/10sorlabs/catalog/
COPY --link selfism_workflows/ /opt/10sorlabs/selfism_workflows/
COPY --link bundled_nodes/ /opt/10sorlabs/bundled_nodes/
COPY --link --chmod=755 docker/entrypoint.sh /start.sh

# Inherit the original services, ports, working directory and entrypoint.

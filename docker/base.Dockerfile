# syntax=docker/dockerfile:1.7-labs
# Selfism base image: everything heavy and rarely changing. Built by
# .github/workflows/build-base.yml, tagged ghcr.io/digitrgrs-eng/selfism-dashboard:base-<hash>
# (scripts/base_image.py hashes this file and every file it reads). The app Dockerfile is
# FROM that tag pinned by digest and only adds the launcher, catalog and workflows.
#
# Stage "fat" is exactly the bake of the old single-stage Dockerfile (same base digest,
# same apt packages, same bake_comfyui.sh), pinned to the old image's package versions,
# plus the llama-cpp-python wheel the environment repair used to download on each pod.
# The final stage copies fat's finished filesystem into a fresh image, so the cu128 torch
# and the other files the bake replaced or deleted (about 4.9 GB that the old image
# still shipped in its lower layers) are gone. Paths, owners, modes, the system
# python3.12 and every package version stay the same.

FROM python:3.12-slim AS r2deps
# boto3 for scripts/upload_models_r2.py and the launcher's private R2 source; the
# dependency versions are the ones the 5dcaf8e image shipped.
RUN pip install --no-cache-dir --no-compile --target=/r2deps \
    boto3==1.43.104 botocore==1.43.120 jmespath==1.1.0 python-dateutil==2.9.0.post0 \
    s3transfer==0.19.3 six==1.17.0 urllib3==2.8.0

# Verified linux/amd64 manifest of 10sorllabs/comfyui-workflow-launcher:2.0.
FROM 10sorllabs/comfyui-workflow-launcher@sha256:d01908958aa33cc9117b478d81845d14ed53c135f173e3db3eafe14e780e8846 AS fat

# Bake ComfyUI v0.38.1 (git checkout, origin = upstream) into /opt/comfyui-baked.
# bake_comfyui.sh first upgrades the base cu128 torch stack to 2.10.0+cu130 (matching
# torchvision/torchaudio) so comfy_kitchen CUDA is not disabled, then installs ComfyUI
# requirements with torch/torchvision/torchaudio/numpy/transformers/pillow/opencv pinned
# to that cu130 freeze. RunPod's start.sh copies /opt/comfyui-baked to
# /workspace/runpod-slim/ComfyUI on first boot and its venv (--system-site-packages; path
# may still be named .venv-cu128) sees these packages, so the boot hook (selfism_boot ->
# selfism_int8.py) finds INT8 support and comfy-kitchen >= 0.2.16 and does nothing.
# Keep COMFYUI_TAG/COMFYUI_SHA identical to TARGET_TAG/TARGET_SHA in launcher/selfism_int8.py
# (tests/test_bake.py checks this).
ARG COMFYUI_TAG=v0.38.1
ARG COMFYUI_SHA=20ca544ee0436721d8eb5f544665e490609f72c8
# The same RUN first installs OpenGL/EGL/GLES userspace for ComfyUI core GLSLShader
# (Artfat Colorgrade grain); with it GLSLShader initializes via ANGLE on NVIDIA Vulkan.
# It also bakes the Krast ORG + Depth node packs (and the two Hearmeman packs and the
# AusBoss pack) listed in docker/base-nodes.json into /opt/10sorlabs/node_cache, then
# installs the SageAttention 2.2.0 cu130/cp312 wheel (--no-deps, so onnxruntime-gpu stays
# 1.30.0). The ComfyUI startup check (main.py --quick-test-for-ci) runs inside the bake.
RUN --mount=type=bind,source=docker/bake_comfyui.sh,target=/tmp/bake_comfyui.sh \
    --mount=type=bind,source=docker/base-nodes.json,target=/tmp/selfism-catalog.json \
    --mount=type=bind,source=docker/base-constraints.txt,target=/tmp/base-constraints.txt \
    apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 libopengl0 libglu1-mesa libegl1 libegl-mesa0 libgl1-mesa-dri libgles2 \
       libportaudio2 \
    && rm -rf /var/lib/apt/lists/* \
    && COMFYUI_TAG="${COMFYUI_TAG}" COMFYUI_SHA="${COMFYUI_SHA}" \
       SELFISM_BASE_CONSTRAINTS=/tmp/base-constraints.txt bash /tmp/bake_comfyui.sh

RUN --mount=type=bind,source=docker/install_llama_cpp.sh,target=/tmp/install_llama_cpp.sh \
    bash /tmp/install_llama_cpp.sh

# Flatten: one copy of every live file, in a few layers so a pull downloads them in
# parallel. Order and content do not matter for the result; together they are fat's /.
FROM scratch
COPY --from=fat --exclude=usr/local/lib/python3.12 / /
# --exclude paths are relative to the copied source directory (here python3.12/).
COPY --from=fat --exclude=dist-packages/nvidia \
     --exclude=dist-packages/torch \
     --exclude=dist-packages/triton \
     /usr/local/lib/python3.12/ /usr/local/lib/python3.12/
COPY --from=fat /usr/local/lib/python3.12/dist-packages/torch/ /usr/local/lib/python3.12/dist-packages/torch/
COPY --from=fat /usr/local/lib/python3.12/dist-packages/triton/ /usr/local/lib/python3.12/dist-packages/triton/
COPY --from=fat /usr/local/lib/python3.12/dist-packages/nvidia/ /usr/local/lib/python3.12/dist-packages/nvidia/
COPY --from=r2deps /r2deps/ /opt/r2deps/

# Image config of 10sorllabs/comfyui-workflow-launcher@sha256:d0190895... (a fresh image
# starts empty). The app Dockerfile sets the Selfism ENV on top, as before.
ENV PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
    FILEBROWSER_CONFIG=/workspace/runpod-slim/.filebrowser.json \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    NVIDIA_REQUIRE_CUDA="" \
    NVIDIA_DISABLE_REQUIRE=true \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    LAUNCHER_PORT=3000 \
    LAUNCHER_AUTO_UPDATE=1 \
    LAUNCHER_GITHUB_REPO=10sorlabs/AI1-Model-Grabber \
    LAUNCHER_GITHUB_REF=main \
    LCT_API_BASE=https://rapidcache.10sorlabs.com \
    COMFYUI_DIR=/workspace/runpod-slim/ComfyUI \
    HF_TOKEN_FILE=/opt/10sorlabs/secrets/hf_token
USER root
WORKDIR /workspace/runpod-slim
EXPOSE 22/tcp 3000/tcp 8080/tcp 8188/tcp 8888/tcp
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:3000/api/health || exit 1
ENTRYPOINT ["/start.sh"]
LABEL org.opencontainers.image.title="Selfora / Selfism dashboard base" \
      org.opencontainers.image.description="CUDA 13 PyTorch, ComfyUI bake and node packs for the Selfism dashboard" \
      org.opencontainers.image.source="https://github.com/digitrgrs-eng/selfism-dashboard"

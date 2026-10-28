# syntax=docker/dockerfile:1.7
# Thin app image. Everything heavy (CUDA 13 PyTorch, ComfyUI v0.38.1 bake, baked node
# packs, SageAttention, onnxruntime-gpu, sam2, mediapipe, llama-cpp-python, boto3 for R2)
# is in the base image built by .github/workflows/build-base.yml from
# docker/base.Dockerfile. The tag is base-<hash of the base inputs> and is pinned by digest
# here; tests/test_base_image.py checks the hash matches the current inputs. When a base
# input changes, build the base, then put its tag and digest below.
# This file has no RUN step, so a card-only change rebuilds only the COPY layers below
# and RunPod pulls only those (a few MB) when it already has the base.
FROM ghcr.io/digitrgrs-eng/selfism-dashboard:BASE_TAG@BASE_DIGEST

ARG IMAGE_VERSION=dev
LABEL org.opencontainers.image.title="Selfora / Selfism dashboard" \
      org.opencontainers.image.description="10sorLabs launcher with Selfism installation and runtime repair" \
      org.opencontainers.image.source="https://github.com/digitrgrs-eng/selfism-dashboard" \
      org.opencontainers.image.version="${IMAGE_VERSION}"

# controlnet_aux reads AUX_ORT_PROVIDERS (else its default list, which also starts
# with CUDA). CUDA first, CPU if the provider cannot be selected. The base bake installs
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
COPY --link scripts/upload_models_r2.py /opt/10sorlabs/scripts/upload_models_r2.py
COPY --link catalog/ /opt/10sorlabs/catalog/
COPY --link selfism_workflows/ /opt/10sorlabs/selfism_workflows/
COPY --link bundled_nodes/ /opt/10sorlabs/bundled_nodes/
COPY --link --chmod=755 docker/entrypoint.sh /start.sh

# Inherit the original services, ports, working directory and entrypoint.
